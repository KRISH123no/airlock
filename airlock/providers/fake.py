"""A provider you can order to misbehave.

This is the most important file in the test suite. Real providers fail rarely,
unpredictably, and expensively — which means the failure paths of a gateway
are exactly the paths that never get tested. Here, "return a 500 twice then
succeed" or "take nine seconds" or "return JSON with a trailing comma" is a
one-line script, and the resulting test is deterministic and instant.

    provider = FakeProvider("flaky", script=[
        ProviderError("down", FailureKind.SERVER_ERROR),
        ProviderError("down", FailureKind.SERVER_ERROR),
        "finally, an answer",
    ])
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from ..core.cost import estimate_tokens
from ..errors import ProviderError
from ..types import CompletionRequest, CompletionResponse, FailureKind, Usage

#: One step of a script: text to return, an error to raise, a full response,
#: or a callable that decides based on the request.
Behaviour = str | ProviderError | CompletionResponse | Callable[[CompletionRequest], Any]


class FakeProvider:
    """A scriptable stand-in for a real model provider."""

    def __init__(
        self,
        name: str = "fake",
        *,
        script: Sequence[Behaviour] | None = None,
        default_text: str = "ok",
        default_behaviour: Behaviour | None = None,
        latency_ms: float = 0.0,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.name = name
        self.script = list(script or [])
        self.default_text = default_text
        #: Used once the script runs out. Lets a provider's health be driven by
        #: state rather than by call count, which keeps tests robust when the
        #: number of calls depends on retry and breaker behaviour.
        self.default_behaviour = default_behaviour
        self.latency_ms = latency_ms
        self._sleep = sleep or asyncio.sleep
        #: Every request this provider was asked to serve, in order.
        self.calls: list[CompletionRequest] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        index = len(self.calls)
        self.calls.append(request)

        if self.latency_ms:
            await self._sleep(self.latency_ms / 1000.0)

        if index < len(self.script):
            behaviour: Behaviour = self.script[index]
        elif self.default_behaviour is not None:
            behaviour = self.default_behaviour
        else:
            behaviour = self.default_text
        if callable(behaviour) and not isinstance(behaviour, (str, ProviderError)):
            behaviour = behaviour(request)

        if isinstance(behaviour, ProviderError):
            # Re-raise a copy so a scripted error can be reused across tests
            # without carrying state from an earlier raise.
            raise ProviderError(
                # args[0] rather than str(): __str__ already appends the kind,
                # and re-wrapping would stutter it into the message twice.
                behaviour.args[0] if behaviour.args else "provider error",
                behaviour.kind,
                provider=self.name,
                retry_after_s=behaviour.retry_after_s,
            )
        if isinstance(behaviour, CompletionResponse):
            behaviour.provider = self.name
            return behaviour
        return self._respond(request, str(behaviour))

    def _respond(self, request: CompletionRequest, text: str) -> CompletionResponse:
        return CompletionResponse(
            text=text,
            model=request.model,
            provider=self.name,
            usage=Usage(
                input_tokens=estimate_tokens(request.prompt_text()),
                output_tokens=estimate_tokens(text),
            ),
        )


class FailingProvider:
    """A provider that fails every call. For fallback and breaker tests."""

    def __init__(self, name: str, kind: FailureKind, message: str = "provider is down") -> None:
        self.name = name
        self.kind = kind
        self.message = message
        self.calls: list[CompletionRequest] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        raise ProviderError(self.message, self.kind, provider=self.name)


class SlowProvider:
    """A provider that takes a fixed time, for timeout and latency tests."""

    def __init__(
        self,
        name: str,
        seconds: float,
        *,
        text: str = "eventually",
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.name = name
        self.seconds = seconds
        self.text = text
        self._sleep = sleep or asyncio.sleep
        self.calls: list[CompletionRequest] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        await self._sleep(self.seconds)
        return CompletionResponse(
            text=self.text,
            model=request.model,
            provider=self.name,
            usage=Usage(
                input_tokens=estimate_tokens(request.prompt_text()),
                output_tokens=estimate_tokens(self.text),
            ),
        )
