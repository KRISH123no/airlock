"""Retry policy.

Two rules do most of the work:

**Only retry what can succeed on a second try.** A 500 or a timeout might.
A malformed request will be malformed again, and retrying it just multiplies
the failure. `FailureKind.retryable` encodes the distinction.

**Use full jitter.** Plain exponential backoff synchronises clients: everyone
who failed at the same moment retries at the same moment, and the recovering
provider is knocked over again. Full jitter — sleeping a uniform random
duration between zero and the computed backoff — spreads them out. This is
the AWS Architecture Blog's recommendation and it is not folklore; it
measurably reduces contention.

A provider that sends `Retry-After` overrides all of this. It knows better.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..errors import ProviderError


@dataclass(frozen=True)
class RetryPolicy:
    #: Attempts after the first. 2 means up to 3 calls in total.
    max_retries: int = 2
    base_delay_s: float = 0.5
    max_delay_s: float = 20.0
    multiplier: float = 2.0
    #: Cap on how long a provider's Retry-After will be honoured.
    max_retry_after_s: float = 60.0

    def should_retry(self, error: ProviderError, attempt: int) -> bool:
        """`attempt` is zero-based: 0 is the first call."""
        return error.retryable and attempt < self.max_retries

    def delay_for(
        self,
        attempt: int,
        error: ProviderError | None = None,
        rng: random.Random | None = None,
    ) -> float:
        """Seconds to wait before attempt `attempt + 1`."""
        if error is not None and error.retry_after_s is not None:
            return min(max(0.0, error.retry_after_s), self.max_retry_after_s)
        ceiling = min(self.base_delay_s * (self.multiplier**attempt), self.max_delay_s)
        return (rng or random).uniform(0.0, ceiling)

    def describe(self) -> str:
        return (
            f"up to {self.max_retries} retries, "
            f"{self.base_delay_s}s base, x{self.multiplier}, "
            f"{self.max_delay_s}s cap, full jitter"
        )
