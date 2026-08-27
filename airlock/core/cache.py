"""Response caching.

Two layers, because they catch different things:

**Exact** — the same prompt, verbatim. Cheap to check and always correct.
Catches retries, refreshes, and the same question asked twice.

**Semantic** — a prompt that *means* the same thing. "How do I reset my
password?" and "how can i reset password" should hit the same answer. Catches
far more, at the cost of being a judgement call rather than a fact.

Caching a model response is only sound when the request is deterministic —
temperature 0. Above that the caller asked for variety, and handing back a
stored answer silently ignores that, so this refuses to cache it.

The embedder here is a local character-trigram hasher, not a learned model.
That keeps the whole thing dependency-free and deterministic for tests. It is
weaker than real embeddings at paraphrase and is meant to be swapped for one
in production — the interface is a single `embed(text) -> list[float]`.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ..types import CompletionRequest, CompletionResponse


def cache_key(request: CompletionRequest) -> str:
    """Everything that could change the answer, hashed."""
    parts = [
        request.model,
        f"{request.temperature:.4f}",
        str(request.max_output_tokens),
        repr(request.response_schema),
        *(f"{m.role.value}:{m.content}" for m in request.messages),
    ]
    return hashlib.sha256("␟".join(parts).encode("utf-8")).hexdigest()


def is_cacheable(request: CompletionRequest) -> bool:
    """Only deterministic requests. Above temperature 0 the caller wants variety."""
    return request.temperature == 0.0


@dataclass
class _Entry:
    response: CompletionResponse
    stored_at: float
    vector: list[float] | None = None
    prompt: str = ""


class ExactCache:
    """LRU with a TTL, keyed on the exact request."""

    def __init__(
        self,
        max_entries: int = 1024,
        ttl_s: float = 3600.0,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._now = now or time.monotonic
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, request: CompletionRequest) -> CompletionResponse | None:
        if not is_cacheable(request):
            return None
        key = cache_key(request)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            if self._now() - entry.stored_at > self.ttl_s:
                del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return _copy(entry.response)

    def put(self, request: CompletionRequest, response: CompletionResponse) -> None:
        if not is_cacheable(request):
            return
        with self._lock:
            self._entries[cache_key(request)] = _Entry(_copy(response), self._now())
            self._entries.move_to_end(cache_key(request))
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...  # pragma: no cover


class TrigramEmbedder:
    """Hashes character trigrams into a fixed-width unit vector.

    Deliberately simple and deterministic: no model, no network, no state. It
    captures spelling and word overlap, so it matches "reset my password" to
    "reset password". It does not capture meaning, so it will miss "I am
    locked out of my account" — a real embedding model would catch that.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        normalised = " ".join(text.lower().split())
        if len(normalised) < 3:
            normalised = normalised.ljust(3)
        for i in range(len(normalised) - 2):
            trigram = normalised[i : i + 3]
            digest = hashlib.blake2b(trigram.encode("utf-8"), digest_size=4).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            vector[index] += 1.0
        return _normalise(vector)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _normalise(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(v * v for v in vector))
    if magnitude == 0:
        return vector
    return [v / magnitude for v in vector]


class SemanticCache:
    """Exact cache first, then nearest-neighbour over embedded prompts.

    The scan is linear over retained entries. At a few thousand entries that
    is microseconds and needs no index; past that it wants a real vector store,
    and the `Embedder` seam is where that swap happens.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.92,
        max_entries: int = 512,
        ttl_s: float = 3600.0,
        embedder: Embedder | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self.embedder = embedder or TrigramEmbedder()
        self._now = now or time.monotonic
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.near_hits = 0
        self.misses = 0

    def get(self, request: CompletionRequest) -> CompletionResponse | None:
        if not is_cacheable(request):
            return None
        key = cache_key(request)
        prompt = request.prompt_text()
        with self._lock:
            self._expire()
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return _copy(entry.response)

            vector = self.embedder.embed(prompt)
            best: _Entry | None = None
            best_score = 0.0
            for candidate in self._entries.values():
                # Only a response for the same model is interchangeable.
                if candidate.response.model != request.model or candidate.vector is None:
                    continue
                score = cosine(vector, candidate.vector)
                if score > best_score:
                    best, best_score = candidate, score
            if best is not None and best_score >= self.threshold:
                self.near_hits += 1
                return _copy(best.response)
            self.misses += 1
            return None

    def put(self, request: CompletionRequest, response: CompletionResponse) -> None:
        if not is_cacheable(request):
            return
        prompt = request.prompt_text()
        with self._lock:
            self._entries[cache_key(request)] = _Entry(
                _copy(response), self._now(), self.embedder.embed(prompt), prompt
            )
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def _expire(self) -> None:
        cutoff = self._now() - self.ttl_s
        stale = [k for k, e in self._entries.items() if e.stored_at < cutoff]
        for key in stale:
            del self._entries[key]

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, float]:
        total = self.hits + self.near_hits + self.misses
        return {
            "entries": len(self._entries),
            "exact_hits": self.hits,
            "semantic_hits": self.near_hits,
            "misses": self.misses,
            "hit_rate": (self.hits + self.near_hits) / total if total else 0.0,
            "threshold": self.threshold,
        }


def _copy(response: CompletionResponse) -> CompletionResponse:
    """Hand out a copy so a caller cannot mutate what is stored."""
    return CompletionResponse(
        text=response.text,
        model=response.model,
        usage=response.usage,
        provider=response.provider,
        cached=response.cached,
        finish_reason=response.finish_reason,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
        parsed=response.parsed,
    )
