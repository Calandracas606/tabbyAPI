"""Types for the OpenAI Responses API (/v1/responses)."""

from pydantic import AliasChoices, BaseModel, Field
from time import time
from typing import Any, List, Literal, Optional, Union
from uuid import uuid4

from endpoints.OAI.types.chat_completion import ChatCompletionMessage
from endpoints.OAI.types.common import BaseSamplerRequest


class ResponseInputTextPart(BaseModel):
    """A text content part of a Responses API input item."""

    type: str = "input_text"
    text: str = ""


class ResponseInputMessage(BaseModel):
    """One input item of a Responses API request."""

    role: str = "user"
    content: Optional[Union[str, List[ResponseInputTextPart]]] = None
    type: Optional[str] = None


class ResponseTextFormat(BaseModel):
    """
    The text.format field of a Responses API request.

    Mirrors the chat completions response_format: json_object/json_schema
    requests structured output, text is a plain completion. The schema
    itself may sit in `schema` (Responses style) or `json_schema`
    (chat completions style); `schema` shadows a pydantic attribute so it
    is read into schema_ via an alias.
    """

    model_config = {"populate_by_name": True}

    type: Optional[str] = "text"
    json_schema: Optional[object] = None
    schema_: Optional[object] = Field(default=None, validation_alias=AliasChoices("schema"))
    name: Optional[str] = None
    strict: Optional[bool] = None
    description: Optional[str] = None

    def resolved_schema(self) -> Optional[dict]:
        """The JSON schema for structured output, from any accepted shape."""

        for candidate in (self.json_schema, self.schema_):
            if isinstance(candidate, dict):
                if candidate.get("type"):
                    return candidate
                # Chat completions style: json_schema wraps the schema
                nested = candidate.get("schema")
                if isinstance(nested, dict) and nested.get("type"):
                    return nested
        return None


class ResponseTextConfig(BaseModel):
    format: Optional[ResponseTextFormat] = Field(default_factory=ResponseTextFormat)


class ResponsesRequest(BaseSamplerRequest):
    """A request to POST /v1/responses."""

    # Model information. This parameter is not used for routing, the loaded
    # model is used instead (same as chat completions)
    model: Optional[str] = None

    # Input is normalized to chat messages by the endpoint
    input: Optional[Union[str, ResponseInputMessage, List[ResponseInputMessage]]] = None
    instructions: Optional[str] = None
    stream: Optional[bool] = False
    max_output_tokens: Optional[int] = Field(
        default=None,
        description="Alias for max_tokens",
    )
    text: Optional[ResponseTextConfig] = None

    # Structured decoding config, filled in from text.format
    json_schema: Optional[object] = None

    # Common Responses API fields that TabbyAPI does not implement. Accepted
    # for compatibility so SDKs don't get validation errors, then ignored.
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    store: Optional[bool] = None
    truncation: Optional[str] = None
    metadata: Optional[dict] = None
    previous_response_id: Optional[str] = None
    reasoning: Optional[Any] = None
    include: Optional[List[Any]] = None

    model_config = {"extra": "ignore"}

    def to_chat_messages(self) -> List[ChatCompletionMessage]:
        """Normalize the Responses input into chat completion messages."""

        messages: List[ChatCompletionMessage] = []

        if self.instructions:
            messages.append(ChatCompletionMessage(role="system", content=self.instructions))

        items: List[ResponseInputMessage] = []
        if isinstance(self.input, str):
            items = [ResponseInputMessage(role="user", content=self.input)]
        elif isinstance(self.input, ResponseInputMessage):
            items = [self.input]
        elif self.input is not None:
            items = self.input

        for item in items:
            role = item.role if item.role in ("system", "user", "assistant") else "user"
            content = _content_to_text(item.content)
            messages.append(ChatCompletionMessage(role=role, content=content))

        return messages


def _content_to_text(content) -> Optional[str]:
    """Flatten Responses item content into plain text."""

    if content is None or isinstance(content, str):
        return content

    text = ""
    for part in content:
        part_text = getattr(part, "text", None)
        if part_text:
            text += part_text
    return text or None


class ResponseOutputTextAnnotation(BaseModel):
    type: str = ""
    text: Optional[str] = None


class ResponseOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str = ""
    annotations: List[ResponseOutputTextAnnotation] = Field(default_factory=list)


class ResponseOutputMessage(BaseModel):
    type: Literal["message"] = "message"
    id: str = Field(default_factory=lambda: f"msg_{uuid4().hex}")
    status: str = "completed"
    role: str = "assistant"
    content: List[ResponseOutputText] = Field(default_factory=list)


class ResponsesUsage(BaseModel):
    input_tokens: int
    input_tokens_details: Optional[dict] = Field(default_factory=lambda: {"cached_tokens": 0})
    output_tokens: int
    output_tokens_details: Optional[dict] = Field(default_factory=lambda: {"reasoning_tokens": 0})
    total_tokens: int


class ResponsesResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"resp_{uuid4().hex}")
    object: str = "response"
    created_at: int = Field(default_factory=lambda: int(time()))
    status: str = "completed"
    model: str
    output: List[ResponseOutputMessage] = Field(default_factory=list)
    usage: Optional[ResponsesUsage] = None
