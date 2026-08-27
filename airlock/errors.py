"""Exceptions. Everything the gateway raises is an AirlockError."""

from __future__ import annotations

from .types import FailureKind


class AirlockError(Exception):
    """Base class for every failure the gateway reports."""


class ProviderError(AirlockError):
    """A provider call failed. `kind` decides whether a retry is worth trying."""

    def __init__(
        self,
        message: str,
        kind: FailureKind = FailureKind.UNKNOWN,
        *,
        provider: str = "",
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.provider = provider
        #: Honoured over the computed backoff when a provider tells us to wait.
        self.retry_after_s = retry_after_s

    @property
    def retryable(self) -> bool:
        return self.kind.retryable

    def __str__(self) -> str:
        where = f" [{self.provider}]" if self.provider else ""
        return f"{super().__str__()}{where} ({self.kind.value})"


class BudgetExceeded(AirlockError):
    """The caller has spent its allowance for the current window."""

    def __init__(self, key_id: str, spent_usd: float, limit_usd: float) -> None:
        super().__init__(
            f"{key_id} has spent ${spent_usd:.4f} of its ${limit_usd:.4f} limit"
        )
        self.key_id = key_id
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd


class CircuitOpen(AirlockError):
    """The breaker for this provider is open, so the call was not attempted."""

    def __init__(self, provider: str, retry_in_s: float) -> None:
        super().__init__(f"circuit for {provider} is open, retrying in {retry_in_s:.1f}s")
        self.provider = provider
        self.retry_in_s = retry_in_s


class AllProvidersFailed(AirlockError):
    """Every provider in the fallback chain was tried and none succeeded."""

    def __init__(self, attempts: list[tuple[str, str]]) -> None:
        detail = "; ".join(f"{provider}: {reason}" for provider, reason in attempts)
        super().__init__(f"no provider could serve the request — {detail}")
        self.attempts = attempts


class GuardrailTripped(AirlockError):
    """A request or response was blocked by a guardrail."""

    def __init__(self, guardrail: str, detail: str) -> None:
        super().__init__(f"{guardrail}: {detail}")
        self.guardrail = guardrail
        self.detail = detail
