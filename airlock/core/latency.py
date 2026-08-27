"""Latency statistics.

Averages hide the problem. If 99 requests take 200ms and one takes 30s, the
mean is 500ms and looks fine, while one user in a hundred is staring at a
frozen screen. Percentiles are what you page on, so those are what this keeps.

The window is bounded and exact: the last N samples, sorted on read. That is
O(N log N) per read rather than the O(1) of an HDR histogram, but N is a few
thousand and the numbers are exact rather than bucketed — a fair trade at this
scale, and stated here so nobody mistakes it for a streaming quantile
estimator.
"""

from __future__ import annotations

import threading
from collections import deque


class LatencyWindow:
    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = capacity
        self._samples: deque[float] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record(self, milliseconds: float) -> None:
        with self._lock:
            self._samples.append(milliseconds)

    def __len__(self) -> int:
        return len(self._samples)

    def percentile(self, p: float) -> float:
        """`p` in 0..100. Nearest-rank, so the value is one that really occurred."""
        if not 0 <= p <= 100:
            raise ValueError("percentile must be between 0 and 100")
        with self._lock:
            if not self._samples:
                return 0.0
            ordered = sorted(self._samples)
        if p == 0:
            return ordered[0]
        rank = max(1, min(len(ordered), -(-len(ordered) * int(p * 100) // 10000)))
        return ordered[rank - 1]

    def mean(self) -> float:
        with self._lock:
            return sum(self._samples) / len(self._samples) if self._samples else 0.0

    def stats(self) -> dict[str, float]:
        with self._lock:
            ordered = sorted(self._samples)
        if not ordered:
            return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
        return {
            "count": len(ordered),
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "max": ordered[-1],
            "mean": sum(ordered) / len(ordered),
        }
