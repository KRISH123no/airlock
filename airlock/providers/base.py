"""What the gateway needs from a model provider.

Kept to one method on purpose. Every vendor difference — auth, URL shapes,
field names, error formats — is absorbed by the adapter, so the gateway never
learns which vendor it is talking to. That is what makes a mid-flight fallback
from one vendor to another possible.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import CompletionRequest, CompletionResponse


@runtime_checkable
class Provider(Protocol):
    """A thing that can turn a request into a response, or raise ProviderError."""

    name: str

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Run the completion.

        Must raise `ProviderError` with an accurate `FailureKind` on failure —
        the kind is what the retry policy reads to decide whether trying again
        could possibly help.
        """
        ...


class Route:
    """One option in a fallback chain: a provider plus the model to ask it for.

    A chain is ordered by preference. The gateway walks it until something
    answers, skipping any provider whose circuit is open.
    """

    def __init__(self, provider: Provider, model: str, *, weight: float = 1.0) -> None:
        self.provider = provider
        self.model = model
        self.weight = weight

    @property
    def name(self) -> str:
        return f"{self.provider.name}/{self.model}"

    def __repr__(self) -> str:
        return f"Route({self.name})"
