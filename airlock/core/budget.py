"""Spending limits that hold under concurrency.

The naive design — add up what has been spent, compare against the limit,
then make the call — is wrong the moment two requests are in flight. Both read
"plenty left", both proceed, and together they overshoot. With slow LLM calls
and high concurrency the overshoot is not marginal.

So this reserves the worst-case cost *before* the call and commits the actual
cost after. A reservation is money that is already gone as far as every other
request is concerned, until it is committed or released.

Windows roll: a $10/hour limit means the last 3600 seconds, not a bucket that
resets on the hour and lets you spend $20 across the boundary.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from ..errors import BudgetExceeded


@dataclass(frozen=True)
class BudgetLimit:
    limit_usd: float
    #: Length of the rolling window in seconds. 3600 = "per hour".
    window_s: float = 3600.0


@dataclass
class Reservation:
    id: int
    key_id: str
    amount_usd: float
    created_at: float
    settled: bool = False


@dataclass
class _KeyState:
    spend: deque[tuple[float, float]] = field(default_factory=deque)  # (when, usd)
    reserved: dict[int, float] = field(default_factory=dict)


class BudgetLedger:
    """Tracks spending per API key against a rolling window."""

    def __init__(self, now: Callable[[], float] | None = None) -> None:
        self._now = now or time.monotonic
        self._limits: dict[str, BudgetLimit] = {}
        self._state: dict[str, _KeyState] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ config

    def set_limit(self, key_id: str, limit_usd: float, window_s: float = 3600.0) -> None:
        with self._lock:
            self._limits[key_id] = BudgetLimit(limit_usd, window_s)

    def limit_for(self, key_id: str) -> BudgetLimit | None:
        return self._limits.get(key_id)

    # ------------------------------------------------------------------- query

    def spent(self, key_id: str) -> float:
        """Committed spend inside the current window."""
        with self._lock:
            return self._committed(key_id)

    def committed_and_reserved(self, key_id: str) -> float:
        with self._lock:
            state = self._state.get(key_id)
            reserved = sum(state.reserved.values()) if state else 0.0
            return self._committed(key_id) + reserved

    def remaining(self, key_id: str) -> float:
        """How much a new request may cost. Unlimited keys report infinity."""
        limit = self._limits.get(key_id)
        if limit is None:
            return float("inf")
        return max(0.0, limit.limit_usd - self.committed_and_reserved(key_id))

    # ----------------------------------------------------------- reserve/settle

    def reserve(self, key_id: str, amount_usd: float) -> Reservation:
        """Hold `amount_usd` against the key, or raise BudgetExceeded."""
        with self._lock:
            limit = self._limits.get(key_id)
            state = self._state.setdefault(key_id, _KeyState())
            if limit is not None:
                self._prune(key_id, limit.window_s)
                in_use = self._committed(key_id) + sum(state.reserved.values())
                if in_use + amount_usd > limit.limit_usd:
                    raise BudgetExceeded(key_id, in_use, limit.limit_usd)
            reservation = Reservation(
                id=next(self._ids),
                key_id=key_id,
                amount_usd=amount_usd,
                created_at=self._now(),
            )
            state.reserved[reservation.id] = amount_usd
            return reservation

    def commit(self, reservation: Reservation, actual_usd: float) -> None:
        """Replace the reserved ceiling with what was actually spent.

        Idempotent: settling a reservation twice must not bill twice. A retry
        loop or an over-eager `finally` would otherwise double-charge, and
        over-billing a customer is a worse bug than letting one call through.
        """
        with self._lock:
            if reservation.settled:
                return
            self._settle(reservation)
            state = self._state.setdefault(reservation.key_id, _KeyState())
            state.spend.append((self._now(), actual_usd))

    def release(self, reservation: Reservation) -> None:
        """Give the reservation back — the call failed and cost nothing."""
        with self._lock:
            self._settle(reservation)

    def _settle(self, reservation: Reservation) -> None:
        if reservation.settled:
            return
        state = self._state.get(reservation.key_id)
        if state is not None:
            state.reserved.pop(reservation.id, None)
        reservation.settled = True

    # ---------------------------------------------------------------- internals

    def _committed(self, key_id: str) -> float:
        state = self._state.get(key_id)
        if state is None:
            return 0.0
        limit = self._limits.get(key_id)
        if limit is not None:
            self._prune(key_id, limit.window_s)
        return sum(amount for _, amount in state.spend)

    def _prune(self, key_id: str, window_s: float) -> None:
        state = self._state.get(key_id)
        if state is None:
            return
        cutoff = self._now() - window_s
        while state.spend and state.spend[0][0] < cutoff:
            state.spend.popleft()

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Per-key spending, for the dashboard and the CLI."""
        out: dict[str, dict[str, float]] = {}
        for key_id in set(self._limits) | set(self._state):
            limit = self._limits.get(key_id)
            out[key_id] = {
                "spent_usd": self.spent(key_id),
                "reserved_usd": self.committed_and_reserved(key_id) - self.spent(key_id),
                "limit_usd": limit.limit_usd if limit else float("inf"),
                "remaining_usd": self.remaining(key_id),
            }
        return out
