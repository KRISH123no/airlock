"""The shapes that move through the gateway.

Deliberately provider-neutral: a request here is not an OpenAI request or an
Anthropic request, it is what both have in common. Adapters translate at the
edge, which is what lets one request fall back from one vendor to another
mid-flight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True)
class Usage:
    """Token counts. `cached_input_tokens` is a subset of `input_tokens`."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cached_input_tokens + other.cached_input_tokens,
        )


@dataclass(frozen=True)
class CompletionRequest:
    model: str
    messages: tuple[Message, ...]
    max_output_tokens: int = 1024
    temperature: float = 0.0
    #: A JSON Schema the response must satisfy. Enables the repair path.
    response_schema: dict[str, Any] | None = None
    #: Identifies the caller for budgeting and attribution.
    api_key_id: str = "default"
    #: Free-form labels so cost can be attributed to a feature, not just a key.
    tags: tuple[str, ...] = ()

    def prompt_text(self) -> str:
        return "\n".join(f"{m.role.value}: {m.content}" for m in self.messages)


@dataclass
class CompletionResponse:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    provider: str = ""
    #: True when this response came from the cache rather than a provider.
    cached: bool = False
    finish_reason: str = "stop"
    #: Populated by the gateway, not the provider.
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    #: The parsed object, when the request asked for a schema.
    parsed: Any = None


class FailureKind(StrEnum):
    """Why a provider call failed, and therefore whether retrying can help."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    CONNECTION = "connection"
    BAD_REQUEST = "bad_request"
    AUTHENTICATION = "authentication"
    CONTENT_FILTER = "content_filter"
    CONTEXT_LENGTH = "context_length"
    #: The response was not valid against the requested schema. Retryable
    #: because a model that mangled its JSON once often gets it right again.
    SCHEMA_INVALID = "schema_invalid"
    UNKNOWN = "unknown"

    @property
    def retryable(self) -> bool:
        """A malformed request will be just as malformed the second time."""
        return self in {
            FailureKind.TIMEOUT,
            FailureKind.RATE_LIMIT,
            FailureKind.SERVER_ERROR,
            FailureKind.CONNECTION,
            FailureKind.SCHEMA_INVALID,
        }
