"""Responses API (POST /v1/responses) utilities for the OAI server.

The Responses API is translated to the chat completions pipeline: requests
are normalized into a ChatCompletionRequest and generation runs through
_chat_stream_collector, so reasoning parsing, guided decoding and usage
accounting behave the same as /v1/chat/completions.
"""

import asyncio
import json
import pathlib
from asyncio import CancelledError
from time import time
from typing import List, Optional
from uuid import uuid4

from fastapi import HTTPException, Request

from common.errors import ContextLengthExceededError, ContextLengthHTTPException
from common.logger import xlogger
from common.multimodal import MultimodalEmbeddingWrapper
from common.networking import (
    get_context_length_generator_error,
    get_generator_error,
    handle_request_error,
    request_tag,
    DisconnectHandler,
)
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.types.common import UsageStats
from endpoints.OAI.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)
from endpoints.OAI.utils.chat_completion import (
    _chat_stream_collector,
    _resolve_start_in_reasoning,
)
from endpoints.OAI.utils.common_ import aggregate_usage_stats, get_usage_stats
from endpoints.OAI.utils.completion import _gen_label


# Responses-only fields that have no chat completions equivalent
_RESPONSES_ONLY_FIELDS = frozenset(
    {
        "input",
        "instructions",
        "max_output_tokens",
        "text",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "store",
        "truncation",
        "metadata",
        "previous_response_id",
        "reasoning",
        "include",
    }
)


def responses_to_chat_request(data: ResponsesRequest) -> ChatCompletionRequest:
    """Translate a Responses API request into a chat completion request."""

    # Sampler params are shared with chat completions via BaseSamplerRequest;
    # carry over everything the client set instead of an allowlist
    sampler_payload = data.model_dump(
        mode="json", exclude_none=True, exclude=_RESPONSES_ONLY_FIELDS
    )
    chat_data = ChatCompletionRequest(messages=data.to_chat_messages(), **sampler_payload)

    if data.max_output_tokens is not None:
        chat_data.max_tokens = data.max_output_tokens

    # Structured output: reuse the guided decoding machinery that chat
    # completions already wires up through json_schema
    text_format = data.text.format if data.text else None
    if text_format is not None and text_format.type in ("json_object", "json_schema"):
        schema = text_format.resolved_schema()
        chat_data.json_schema = schema if schema is not None else {"type": "object"}

    return chat_data


def _compose_response(
    request_id: str,
    generations: List[dict],
    model_name: Optional[str],
) -> ResponsesResponse:
    """Compose a Responses API response from collected generations."""

    output = []
    usage_stats = []
    for generation in generations:
        output.append(
            ResponseOutputMessage(
                content=[ResponseOutputText(text=generation.get("content") or "")]
            )
        )
        stats = get_usage_stats(generation)
        if stats is not None:
            usage_stats.append(stats)

    usage = None
    if usage_stats:
        aggregated = aggregate_usage_stats(usage_stats)
        usage = ResponsesUsage(
            input_tokens=aggregated.prompt_tokens,
            output_tokens=aggregated.completion_tokens,
            total_tokens=aggregated.total_tokens,
        )

    return ResponsesResponse(id=f"resp_{request_id}", model=model_name, output=output, usage=usage)


async def generate_response(
    prompt: str,
    mm_embeddings: Optional[MultimodalEmbeddingWrapper],
    chat_data: ChatCompletionRequest,
    request: Request,
    model_path: pathlib.Path,
    disconnect_handler: DisconnectHandler,
) -> ResponsesResponse:
    """Generate a non-streaming Responses API result via the chat pipeline."""

    gen_tasks: List[asyncio.Task] = []

    try:
        xlogger.debug(
            f"{request_tag(request)} responses payload, ID {request.state.id}",
            {
                "request_id": request.state.id,
                "prompt": prompt,
                "data": chat_data.model_dump(mode="json"),
                "model_path": str(model_path),
            },
        )

        start_in_reasoning_mode = _resolve_start_in_reasoning(prompt, chat_data)

        gen_task = asyncio.create_task(
            _chat_stream_collector(
                0,
                None,
                request.state.id,
                prompt,
                chat_data,
                start_in_reasoning_mode,
                mm_embeddings=mm_embeddings,
                streaming_mode=False,
                disconnect_handler=disconnect_handler,
                label=_gen_label(request, "responses", 1, 0, False),
            )
        )
        gen_tasks.append(gen_task)

        await asyncio.wait([*gen_tasks])

        generations = []
        for task in gen_tasks:
            result = task.result()
            if isinstance(result, Exception):
                raise result
            generations.append(result)

        response = _compose_response(request.state.id, generations, model_path.name)

        xlogger.debug(f"{request_tag(request)} responses finished", {"response": response})
        return response

    except CancelledError:
        raise

    except ContextLengthExceededError as exc:
        error_message = handle_request_error(str(exc), exc_info=False).error.message
        raise ContextLengthHTTPException(error_message) from exc

    except Exception as exc:
        error_message = handle_request_error(
            f"{request_tag(request)} responses generation aborted. Maybe the model was "
            "unloaded? Please check the server console."
        ).error.message

        raise HTTPException(503, error_message) from exc

    finally:
        await disconnect_handler.cleanup()


async def stream_generate_response(
    prompt: str,
    mm_embeddings: Optional[MultimodalEmbeddingWrapper],
    chat_data: ChatCompletionRequest,
    request: Request,
    model_path: pathlib.Path,
    disconnect_handler: DisconnectHandler,
):
    """Stream a Responses API generation as spec-compliant SSE events.

    Runs the same collector as non-streaming mode in streaming mode, then
    reframes its deltas into Responses API events. Reasoning deltas are
    folded into the message; only final content is emitted as output_text.
    """

    try:
        response_id = f"resp_{request.state.id}"
        model_name = model_path.name
        created_at = int(time())
        item_id = f"msg_{uuid4().hex}"

        def response_summary(status: str) -> dict:
            return {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "status": status,
                "model": model_name,
                "output": [],
            }

        yield _sse_event(
            "response.created",
            {"type": "response.created", "response": response_summary("in_progress")},
        )

        yield _sse_event(
            "response.in_progress",
            {"type": "response.in_progress", "response": response_summary("in_progress")},
        )

        yield _sse_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": item_id,
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            },
        )

        yield _sse_event(
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        )

        full_text = ""
        usage_stats_list: List[UsageStats] = []
        start_in_reasoning_mode = _resolve_start_in_reasoning(prompt, chat_data)

        gen_queue: asyncio.Queue = asyncio.Queue()
        gen_task = asyncio.create_task(
            _chat_stream_collector(
                0,
                gen_queue,
                request.state.id,
                prompt,
                chat_data,
                start_in_reasoning_mode,
                mm_embeddings=mm_embeddings,
                streaming_mode=True,
                disconnect_handler=disconnect_handler,
                label=_gen_label(request, "responses", 1, 0, True),
            )
        )

        while True:
            generation = await gen_queue.get()

            if isinstance(generation, Exception):
                raise generation

            delta_content = generation.get("delta_content") or ""
            if delta_content:
                full_text += delta_content
                yield _sse_event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": delta_content,
                    },
                )

            stats = get_usage_stats(generation)
            if stats is not None:
                usage_stats_list.append(stats)

            # The collector sends its finish chunk last; also stop if the
            # task exited without one (error paths push an exception instead)
            if gen_task.done() and gen_queue.empty():
                break

        await asyncio.wait([gen_task])
        if gen_task.exception():
            raise gen_task.exception()

        yield _sse_event(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": full_text,
            },
        )

        final_item = ResponseOutputMessage(id=item_id, content=[ResponseOutputText(text=full_text)])
        yield _sse_event(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": final_item.content[0].model_dump(mode="json"),
            },
        )

        yield _sse_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": final_item.model_dump(mode="json"),
            },
        )

        usage = None
        if usage_stats_list:
            usage = _usage_from_stream(usage_stats_list)

        completed_response = ResponsesResponse(
            id=response_id,
            model=model_name,
            created_at=created_at,
            status="completed",
            output=[final_item],
            usage=usage,
        )

        yield _sse_event(
            "response.completed",
            {
                "type": "response.completed",
                "response": completed_response.model_dump(mode="json"),
            },
        )

    except CancelledError:
        raise

    except ContextLengthExceededError as exc:
        yield get_context_length_generator_error(str(exc))

    except Exception as exc:
        xlogger.error("Error during responses streaming", str(exc), details=f"\n{str(exc)}")
        yield get_generator_error("Responses stream aborted. Please check the server console.")

    finally:
        await disconnect_handler.cleanup()


def _usage_from_stream(usage_stats: List[UsageStats]) -> Optional[ResponsesUsage]:
    """Aggregate collected usage stats into a Responses usage object."""

    if not usage_stats:
        return None

    aggregated = aggregate_usage_stats(usage_stats)
    return ResponsesUsage(
        input_tokens=aggregated.prompt_tokens,
        output_tokens=aggregated.completion_tokens,
        total_tokens=aggregated.total_tokens,
    )


def _sse_event(event_type: str, data: dict) -> dict:
    """Frame an SSE event for sse_starlette's EventSourceResponse."""
    return {"event": event_type, "data": json.dumps(data, ensure_ascii=False)}
