"""Traces: the record of what actually happened to one request.

When someone asks "why was that answer slow / wrong / expensive?", the answer
has to come from somewhere. A trace is that somewhere. Each request produces
one trace containing every attempt made on its behalf — including the ones
that failed, which are usually the interesting ones.

The shape follows OpenTelemetry's span model closely enough to export later,
without taking a dependency on it now.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

_ids = itertools.count(1)


@dataclass
class Attempt:
    """One call to one provider."""

    provider: str
    model: str
    started_at: float
    duration_ms: float = 0.0
    ok: bool = False
    error: str | None = None
    error_kind: str | None = None
    #: Which retry this was within the current route. 0 is the first try.
    retry: int = 0
    #: Seconds slept before this attempt.
    backoff_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "duration_ms": round(self.duration_ms, 2),
            "ok": self.ok,
            "error": self.error,
            "error_kind": self.error_kind,
            "retry": self.retry,
            "backoff_s": round(self.backoff_s, 3),
        }


@dataclass
class Trace:
    id: int
    key_id: str
    requested_model: str
    started_at: float
    tags: tuple[str, ...] = ()
    attempts: list[Attempt] = field(default_factory=list)
    #: Providers skipped because their circuit was open.
    skipped: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    ok: bool = False
    cache_hit: bool = False
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    served_by: str | None = None
    outcome: str = "pending"
    guardrails: list[str] = field(default_factory=list)

    @property
    def retries(self) -> int:
        return max(0, len(self.attempts) - 1)

    @property
    def provider_time_ms(self) -> float:
        """Time spent waiting on providers, failed attempts included."""
        return sum(a.duration_ms for a in self.attempts)

    @property
    def overhead_ms(self) -> float:
        """What the gateway itself cost. Should stay small."""
        return max(0.0, self.duration_ms - self.provider_time_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "key_id": self.key_id,
            "requested_model": self.requested_model,
            "tags": list(self.tags),
            "ok": self.ok,
            "outcome": self.outcome,
            "served_by": self.served_by,
            "cache_hit": self.cache_hit,
            "duration_ms": round(self.duration_ms, 2),
            "overhead_ms": round(self.overhead_ms, 2),
            "cost_usd": round(self.cost_usd, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "retries": self.retries,
            "skipped": list(self.skipped),
            "guardrails": list(self.guardrails),
            "attempts": [a.to_dict() for a in self.attempts],
        }

    def explain(self) -> str:
        """A human-readable account of the request, for the CLI."""
        header = (
            f"trace {self.id}  {self.outcome}  {self.duration_ms:.0f}ms  "
            f"${self.cost_usd:.6f}  key={self.key_id}"
        )
        lines = [header]
        if self.cache_hit:
            lines.append("  cache hit — no provider was called")
        for name in self.skipped:
            lines.append(f"  skipped {name} (circuit open)")
        for attempt in self.attempts:
            status = "ok" if attempt.ok else f"failed: {attempt.error}"
            backoff = f" after {attempt.backoff_s:.2f}s backoff" if attempt.backoff_s else ""
            lines.append(
                f"  attempt {attempt.retry}: {attempt.provider}/{attempt.model} "
                f"{attempt.duration_ms:.0f}ms{backoff} — {status}"
            )
        for note in self.guardrails:
            lines.append(f"  guardrail: {note}")
        if self.served_by:
            lines.append(f"  served by {self.served_by}")
        return "\n".join(lines)


class TraceStore:
    """A bounded ring of recent traces, plus running totals."""

    def __init__(self, capacity: int = 1000, now: Callable[[], float] | None = None) -> None:
        self.capacity = capacity
        self._now = now or time.monotonic
        self._traces: deque[Trace] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.total_requests = 0
        self.total_cost_usd = 0.0
        self.total_cache_hits = 0
        self.total_failures = 0

    def start(self, key_id: str, model: str, tags: tuple[str, ...] = ()) -> Trace:
        return Trace(
            id=next(_ids),
            key_id=key_id,
            requested_model=model,
            started_at=self._now(),
            tags=tags,
        )

    def finish(self, trace: Trace) -> Trace:
        with self._lock:
            self._traces.append(trace)
            self.total_requests += 1
            self.total_cost_usd += trace.cost_usd
            if trace.cache_hit:
                self.total_cache_hits += 1
            if not trace.ok:
                self.total_failures += 1
        return trace

    def recent(self, limit: int = 20) -> list[Trace]:
        with self._lock:
            return list(self._traces)[-limit:][::-1]

    def get(self, trace_id: int) -> Trace | None:
        with self._lock:
            for trace in reversed(self._traces):
                if trace.id == trace_id:
                    return trace
        return None

    def __len__(self) -> int:
        return len(self._traces)

    def __iter__(self) -> Iterator[Trace]:
        with self._lock:
            return iter(list(self._traces))

    def summary(self) -> dict[str, Any]:
        with self._lock:
            requests = self.total_requests
            return {
                "requests": requests,
                "failures": self.total_failures,
                "success_rate": (requests - self.total_failures) / requests if requests else 0.0,
                "cache_hits": self.total_cache_hits,
                "cache_hit_rate": self.total_cache_hits / requests if requests else 0.0,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "traces_retained": len(self._traces),
            }
