"""Tests for the OpenAI Responses API endpoint (/v1/responses)."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI

from common import model  # noqa: F401 - initialize backend imports before the OAI router
from common.networking import get_global_depends
from endpoints.OAI.types.responses import ResponsesRequest
from endpoints.OAI.utils import responses as responses_utils


def make_container():
    """A minimal model container stub for the chat template path."""

    async def render(vars):
        return " ".join(
            f"<{m['role']}>{m['content']}" for m in vars["messages"] if m.get("content")
        )

    template = SimpleNamespace(render=render)
    return SimpleNamespace(
        loaded=True,
        model_dir=Path("models/Test-Model"),
        prompt_template=template,
        reasoning=False,
        harmony=False,
        muse_glimmer=False,
        start_in_reasoning="never",
        use_vision=False,
        template_vars_default={},
        template_vars_force={},
        get_special_tokens=lambda: {},
        validate_context_length=lambda *args, **kwargs: None,
    )


def make_app():
    # Match endpoints/server.py setup_app: global deps provide request.state.id
    app = FastAPI(dependencies=get_global_depends())
    from endpoints.OAI import router as oai_router

    app.include_router(oai_router.setup())
    return app


class DummyDisconnectHandler:
    async def poll(self):
        pass

    async def cleanup(self):
        pass


def finish_generation(content, prompt_tokens=12, gen_tokens=7):
    """The generation dict _chat_stream_collector returns in non-streaming mode."""

    return {
        "index": 0,
        "content": content,
        "reasoning_content": "",
        "tool_calls": [],
        "finish_reason": "stop",
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "prompt_time": 0.01,
        "gen_time": 0.02,
        "total_time": 0.03,
    }


class ResponsesTranslationTests(unittest.TestCase):
    def test_string_input_becomes_user_message(self):
        data = ResponsesRequest.model_validate({"input": "hello there"})
        chat_data = responses_utils.responses_to_chat_request(data)
        self.assertEqual(
            [(m.role, m.content) for m in chat_data.messages], [("user", "hello there")]
        )

    def test_system_user_array_with_input_text_parts(self):
        data = ResponsesRequest.model_validate(
            {
                "input": [
                    {"role": "system", "content": "You are helpful."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Summarize: "},
                            {"type": "input_text", "text": "the page"},
                        ],
                    },
                ]
            }
        )
        chat_data = responses_utils.responses_to_chat_request(data)
        self.assertEqual(
            [(m.role, m.content) for m in chat_data.messages],
            [("system", "You are helpful."), ("user", "Summarize: the page")],
        )

    def test_instructions_prepended_as_system_message(self):
        data = ResponsesRequest.model_validate(
            {"instructions": "Be terse.", "input": [{"role": "user", "content": "hi"}]}
        )
        chat_data = responses_utils.responses_to_chat_request(data)
        self.assertEqual(
            [(m.role, m.content) for m in chat_data.messages],
            [("system", "Be terse."), ("user", "hi")],
        )

    def test_assistant_output_text_parts_accepted(self):
        data = ResponsesRequest.model_validate(
            {
                "input": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": [{"type": "output_text", "text": "a"}]},
                ]
            }
        )
        chat_data = responses_utils.responses_to_chat_request(data)
        self.assertEqual(
            [(m.role, m.content) for m in chat_data.messages],
            [("user", "q"), ("assistant", "a")],
        )

    def test_unknown_roles_map_to_user(self):
        data = ResponsesRequest.model_validate(
            {"input": [{"role": "developer", "content": "instructions"}]}
        )
        chat_data = responses_utils.responses_to_chat_request(data)
        self.assertEqual(chat_data.messages[-1].role, "user")

    def test_max_output_tokens_maps_to_max_tokens(self):
        data = ResponsesRequest.model_validate(
            {"input": "hi", "max_output_tokens": 123, "temperature": 0.25, "top_p": 0.9}
        )
        chat_data = responses_utils.responses_to_chat_request(data)
        self.assertEqual(chat_data.max_tokens, 123)
        self.assertEqual(chat_data.temperature, 0.25)
        self.assertEqual(chat_data.top_p, 0.9)

    def test_json_schema_format_uses_guided_decoding(self):
        schema = {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        }
        data = ResponsesRequest.model_validate(
            {
                "input": "summarize this",
                "text": {
                    "format": {
                        "type": "json_schema",
                        "strict": True,
                        "name": "response",
                        "schema": schema,
                    }
                },
            }
        )
        chat_data = responses_utils.responses_to_chat_request(data)
        self.assertEqual(chat_data.json_schema, schema)

    def test_json_object_format_defaults_to_object_schema(self):
        data = ResponsesRequest.model_validate(
            {"input": "hi", "text": {"format": {"type": "json_object"}}}
        )
        chat_data = responses_utils.responses_to_chat_request(data)
        self.assertEqual(chat_data.json_schema, {"type": "object"})

    def test_extra_fields_ignored(self):
        data = ResponsesRequest.model_validate(
            {
                "input": "hi",
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "store": False,
                "truncation": "auto",
                "metadata": {"x": 1},
                "reasoning": {"effort": "low"},
                "include": ["reasoning.encrypted_content"],
                "some_future_field": {"nested": True},
            }
        )
        chat_data = responses_utils.responses_to_chat_request(data)
        self.assertEqual(len(chat_data.messages), 1)


class ResponsesEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = make_app()
        self.transport = httpx.ASGITransport(app=self.app)
        self.client = httpx.AsyncClient(transport=self.transport, base_url="http://test")
        self._auth_patch = patch("common.auth.DISABLE_AUTH", True)
        self._auth_patch.start()

    async def asyncTearDown(self):
        self._auth_patch.stop()
        await self.client.aclose()

    async def post(self, payload):
        return await self.client.post("/v1/responses", json=payload)

    async def test_requires_api_key_when_auth_enabled(self):
        import common.auth as auth

        container = make_container()

        async def collector(*args, **kwargs):
            return finish_generation("ok")

        original = model.container
        model.container = container
        original_keys = auth.AUTH_KEYS
        auth.AUTH_KEYS = auth.AuthKeys(api_key="secret-key", admin_key="admin-key")
        self._auth_patch.stop()
        try:
            with patch.object(responses_utils, "_chat_stream_collector", collector):
                no_key = await self.client.post("/v1/responses", json={"input": "hi"})
                bad_key = await self.client.post(
                    "/v1/responses",
                    json={"input": "hi"},
                    headers={"Authorization": "Bearer wrong"},
                )
                good_key = await self.client.post(
                    "/v1/responses",
                    json={"input": "hi"},
                    headers={"Authorization": "Bearer secret-key"},
                )
        finally:
            auth.AUTH_KEYS = original_keys
            self._auth_patch.start()
            model.container = original

        self.assertEqual(no_key.status_code, 401)
        self.assertEqual(bad_key.status_code, 401)
        self.assertEqual(good_key.status_code, 200, good_key.text)

    async def test_non_streaming_response_structure(self):
        container = make_container()

        async def collector(*args, **kwargs):
            return finish_generation('{"summary": "short"}', prompt_tokens=10, gen_tokens=4)

        original = model.container
        model.container = container
        try:
            with patch.object(responses_utils, "_chat_stream_collector", collector):
                response = await self.post(
                    {
                        "model": "Test-Model",
                        "input": [
                            {"role": "system", "content": "You are a summarizer."},
                            {
                                "role": "user",
                                "content": [{"type": "input_text", "text": "Summarize: page"}],
                            },
                        ],
                        "text": {
                            "format": {
                                "type": "json_schema",
                                "strict": True,
                                "name": "response",
                                "schema": {
                                    "type": "object",
                                    "properties": {"summary": {"type": "string"}},
                                    "required": ["summary"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                    }
                )
        finally:
            model.container = original

        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()

        self.assertTrue(data["id"].startswith("resp_"))
        self.assertEqual(data["object"], "response")
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["model"], "Test-Model")
        self.assertIsInstance(data["created_at"], int)

        output = data["output"]
        self.assertEqual(len(output), 1)
        message = output[0]
        self.assertEqual(message["type"], "message")
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["status"], "completed")
        self.assertTrue(message["id"].startswith("msg_"))

        content = message["content"]
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "output_text")
        self.assertEqual(content[0]["text"], '{"summary": "short"}')
        self.assertEqual(content[0]["annotations"], [])

        usage = data["usage"]
        self.assertEqual(usage["input_tokens"], 10)
        self.assertEqual(usage["output_tokens"], 4)
        self.assertEqual(usage["total_tokens"], 14)
        self.assertEqual(usage["input_tokens_details"], {"cached_tokens": 0})
        self.assertEqual(usage["output_tokens_details"], {"reasoning_tokens": 0})

    async def test_string_input_and_instructions(self):
        container = make_container()

        async def collector(*args, **kwargs):
            return finish_generation("ok")

        original = model.container
        model.container = container
        try:
            with patch.object(responses_utils, "_chat_stream_collector", collector):
                response = await self.post(
                    {"input": "hello", "instructions": "Be terse.", "max_output_tokens": 16}
                )
        finally:
            model.container = original

        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["output"][0]["content"][0]["text"], "ok")

    async def test_alias_route_without_v1_prefix(self):
        container = make_container()

        async def collector(*args, **kwargs):
            return finish_generation("alias works")

        original = model.container
        model.container = container
        try:
            with patch.object(responses_utils, "_chat_stream_collector", collector):
                response = await self.client.post("/responses", json={"input": "hi"})
        finally:
            model.container = original

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["output"][0]["content"][0]["text"], "alias works")

    async def test_generation_error_returns_503(self):
        container = make_container()

        async def collector(*args, **kwargs):
            return ValueError("backend failure")

        original = model.container
        model.container = container
        try:
            with patch.object(responses_utils, "_chat_stream_collector", collector):
                response = await self.post({"input": "hi"})
        finally:
            model.container = original

        self.assertEqual(response.status_code, 503, response.text)


class ResponsesStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = make_app()
        self.transport = httpx.ASGITransport(app=self.app)
        self.client = httpx.AsyncClient(transport=self.transport, base_url="http://test")
        self._auth_patch = patch("common.auth.DISABLE_AUTH", True)
        self._auth_patch.start()

    async def asyncTearDown(self):
        self._auth_patch.stop()
        await self.client.aclose()

    async def test_stream_emits_responses_api_events(self):
        container = make_container()

        async def collector(
            task_idx,
            gen_queue,
            request_id,
            prompt,
            params,
            start_in_reasoning_mode,
            mm_embeddings=None,
            streaming_mode=True,
            disconnect_handler=None,
            label=None,
        ):
            assert streaming_mode is True
            chunks = [
                {"index": 0, "delta_content": '{"summary"', "delta_reasoning_content": ""},
                {"index": 0, "delta_content": ': "hi"}', "delta_reasoning_content": ""},
                {
                    "index": 0,
                    "delta_content": "",
                    "delta_reasoning_content": "",
                    "finish_reason": "stop",
                    "prompt_tokens": 9,
                    "gen_tokens": 3,
                },
            ]
            if gen_queue is None:
                return chunks[-1]
            for chunk in chunks:
                await gen_queue.put(chunk)

        original = model.container
        model.container = container
        try:
            with patch.object(responses_utils, "_chat_stream_collector", collector):
                response = await self.client.post(
                    "/v1/responses", json={"input": "hi", "stream": True}
                )
        finally:
            model.container = original

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))

        events = []
        for block in response.text.replace("\r\n", "\n").split("\n\n"):
            block = block.strip()
            if not block:
                continue
            event_type = None
            data = None
            for line in block.split("\n"):
                if line.startswith("event:"):
                    event_type = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[len("data:") :].strip())
            events.append((event_type, data))

        types = [t for t, _ in events]
        self.assertIn("response.created", types)
        self.assertIn("response.in_progress", types)
        self.assertIn("response.output_item.added", types)
        self.assertIn("response.content_part.added", types)
        self.assertIn("response.output_text.delta", types)
        self.assertIn("response.output_text.done", types)
        self.assertIn("response.content_part.done", types)
        self.assertIn("response.output_item.done", types)
        self.assertIn("response.completed", types)

        deltas = [d for t, d in events if t == "response.output_text.delta"]
        self.assertEqual("".join(d["delta"] for d in deltas), '{"summary": "hi"}')

        added = next(d for t, d in events if t == "response.output_item.added")
        self.assertEqual(added["item"]["type"], "message")
        self.assertEqual(added["item"]["role"], "assistant")

        completed = next(d for t, d in events if t == "response.completed")
        self.assertEqual(completed["response"]["status"], "completed")
        self.assertEqual(completed["response"]["id"], events[0][1]["response"]["id"])
        self.assertEqual(
            completed["response"]["output"][0]["content"][0]["text"], '{"summary": "hi"}'
        )
        self.assertEqual(completed["response"]["usage"]["input_tokens"], 9)
        self.assertEqual(completed["response"]["usage"]["output_tokens"], 3)
        self.assertEqual(completed["response"]["usage"]["total_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
