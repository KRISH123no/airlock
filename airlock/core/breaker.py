"""A circuit breaker per provider.

When a provider is failing, continuing to send it traffic wastes the caller's
time — every request pays the full timeout before failing anyway — and adds
load to something already struggling. The breaker notices and stops trying.

Three states:

    CLOSED    normal. Failures are counted in a rolling window.
    OPEN      failing. Calls are rejected immediately, without a request.
    HALF_OPEN cooldown elapsed. A few probes are allowed through; if they
              succeed the breaker closes, if one fails it opens again.

The half-open state is the part that is easy to get wrong. Going straight from
open to closed sends the full traffic at a provider that may still be down.
Letting a limited number of probes through tests it with one request instead
of thousands.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from enum import StrEnum


class State(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_rate_threshold: float = 0.5,
        minimum_volume: int = 5,
        window_s: float = 60.0,
        cooldown_s: float = 30.0,
        half_open_probes: int = 2,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.name = name
        #: Trip when this fraction of recent calls failed...
        self.failure_rate_threshold = failure_rate_threshold
        #: ...but never on fewer than this many calls. Two failures out of two
        #: is a 100% failure rate and means nothing.
        self.minimum_volume = minimum_volume
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self.half_open_probes = half_open_probes
        self._now = now or time.monotonic

        self._state = State.CLOSED
        self._events: deque[tuple[float, bool]] = deque()  # (when, ok)
        self._opened_at = 0.0
        self._probes_started = 0
        self._probes_succeeded = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------- state

    @property
    def state(self) -> State:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def retry_in_s(self) -> float:
        with self._lock:
            if self._state is not State.OPEN:
                return 0.0
            return max(0.0, self.cooldown_s - (self._now() - self._opened_at))

    def allows_request(self) -> bool:
        """True if a call may proceed. Reserves a probe slot when half-open."""
        with self._lock:
            self._maybe_half_open()
            if self._state is State.CLOSED:
                return True
            if self._state is State.OPEN:
                return False
            if self._probes_started < self.half_open_probes:
                self._probes_started += 1
                return True
            return False

    # ------------------------------------------------------------------ record

    def record_success(self) -> None:
        with self._lock:
            self._events.append((self._now(), True))
            if self._state is State.HALF_OPEN:
                self._probes_succeeded += 1
                if self._probes_succeeded >= self.half_open_probes:
                    self._close()

    def record_failure(self) -> None:
        with self._lock:
            self._events.append((self._now(), False))
            if self._state is State.HALF_OPEN:
                # One failed probe is enough: the provider is still unwell.
                self._open()
                return
            if self._state is State.CLOSED and self._should_trip():
                self._open()

    # ---------------------------------------------------------------- internals

    def _should_trip(self) -> bool:
        self._prune()
        total = len(self._events)
        if total < self.minimum_volume:
            return False
        failures = sum(1 for _, ok in self._events if not ok)
        return failures / total >= self.failure_rate_threshold

    def _open(self) -> None:
        self._state = State.OPEN
        self._opened_at = self._now()
        self._probes_started = 0
        self._probes_succeeded = 0

    def _close(self) -> None:
        self._state = State.CLOSED
        self._events.clear()
        self._probes_started = 0
        self._probes_succeeded = 0

    def _maybe_half_open(self) -> None:
        if self._state is State.OPEN and self._now() - self._opened_at >= self.cooldown_s:
            self._state = State.HALF_OPEN
            self._probes_started = 0
            self._probes_succeeded = 0

    def _prune(self) -> None:
        cutoff = self._now() - self.window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def stats(self) -> dict[str, object]:
        with self._lock:
            self._prune()
            total = len(self._events)
            failures = sum(1 for _, ok in self._events if not ok)
            return {
                "provider": self.name,
                "state": self.state.value,
                "calls_in_window": total,
                "failures_in_window": failures,
                "failure_rate": (failures / total) if total else 0.0,
                "retry_in_s": self.retry_in_s(),
            }
