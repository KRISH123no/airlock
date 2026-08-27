"""The gateway: one request in, one response out, and everything in between.

The order of operations is the design, so it is worth stating plainly:

    1. Cache        — an answer we already have costs nothing and takes no time
    2. Budget       — reserve the worst case before spending anything
    3. Route        — walk the fallback chain, skipping providers whose circuit is open
    4. Retry        — within a route, retry only failures that could succeed
    5. Settle       — commit what was actually spent, or give the reservation back
    6. Trace        — record every attempt, including the failed ones

Cache before budget is deliberate: a cached answer should not consume an
allowance. Budget before routing is also deliberate: refusing early is cheaper
than discovering the limit after three providers have been paid.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from .core.breaker import CircuitBreaker
from .core.budget import BudgetLedger
from .core.cost import PriceBook, estimate_tokens
from .core.guardrails import InjectionScreen, Redaction, Redactor, SchemaError, parse_structured
from .core.latency import LatencyWindow
from .core.retry import RetryPolicy
from .errors import AllProvidersFailed, BudgetExceeded, GuardrailTripped, ProviderError
from .providers.base import Route
from .trace import Attempt, Trace, TraceStore
from .types import CompletionRequest, CompletionResponse, FailureKind


class ResponseCache:
    """Structural type for a cache. See `core.cache` for implementations."""

    def get(self, request: CompletionRequest) -> CompletionResponse | None: ...  # pragma: no cover
    def put(
        self, request: CompletionRequest, response: CompletionResponse
    ) -> None: ...  # pragma: no cover


class Gateway:
    def __init__(
        self,
        routes: Mapping[str, Sequence[Route]],
        *,
        price_book: PriceBook | None = None,
        budget: BudgetLedger | None = None,
        retry_policy: RetryPolicy | None = None,
        cache: ResponseCache | None = None,
        redactor: Redactor | None = None,
        injection_screen: InjectionScreen | None = None,
        request_timeout_s: float | None = 30.0,
        breaker_options: dict[str, Any] | None = None,
        traces: TraceStore | None = None,
        now: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if not routes:
            raise ValueError("a gateway needs at least one route")
        self.routes = {name: list(chain) for name, chain in routes.items()}
        self.price_book = price_book or PriceBook()
        self.budget = budget or BudgetLedger(now=now)
        self.retry_policy = retry_policy or RetryPolicy()
        self.cache = cache
        #: Strips personal data on the way out and restores it on the way back.
        self.redactor = redactor
        #: Screens inbound prompts for injection attempts.
        self.injection_screen = injection_screen
        self.request_timeout_s = request_timeout_s
        self.traces = traces or TraceStore(now=now)
        self._now = now or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._rng = rng or random.Random()
        self._breaker_options = breaker_options or {}
        self._breakers: dict[str, CircuitBreaker] = {}
        self.latency = LatencyWindow()
        #: Latency per route, so a slow provider is visible on its own.
        self.route_latency: dict[str, LatencyWindow] = {}

    # ----------------------------------------------------------------- helpers

    def breaker_for(self, route: Route) -> CircuitBreaker:
        if route.provider.name not in self._breakers:
            self._breakers[route.provider.name] = CircuitBreaker(
                route.provider.name, now=self._now, **self._breaker_options
            )
        return self._breakers[route.provider.name]

    def chain_for(self, model: str) -> list[Route]:
        if model not in self.routes:
            raise ProviderError(f"no route configured for model {model!r}", FailureKind.BAD_REQUEST)
        return self.routes[model]

    def _reservation_amount(self, request: CompletionRequest, chain: Sequence[Route]) -> float:
        """Reserve against the most expensive model the request could reach.

        A fallback may be pricier than the primary. Reserving only the
        primary's cost would let a request slip past the budget by failing
        over to a more expensive model.
        """
        input_tokens = estimate_tokens(request.prompt_text())
        return max(
            self.price_book.worst_case_cost(route.model, input_tokens, request.max_output_tokens)
            for route in chain
        )

    # ----------------------------------------------------------------- request

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        trace = self.traces.start(request.api_key_id, request.model, request.tags)
        started = self._now()
        chain = self.chain_for(request.model)

        # 0. Screen the prompt before anything else happens to it.
        if self.injection_screen is not None:
            verdict = self.injection_screen.inspect(_user_text(request))
            if verdict.score >= self.injection_screen.threshold:
                trace.outcome = "blocked"
                trace.guardrails.append(f"injection blocked — {verdict.explain()}")
                trace.duration_ms = (self._now() - started) * 1000
                self.traces.finish(trace)
                raise GuardrailTripped("injection_screen", verdict.explain())
            if verdict.suspicious:
                trace.guardrails.append(f"injection score {verdict.score} — {verdict.explain()}")

        # 1. Cache — free and instant, and must not touch the budget.
        if self.cache is not None:
            hit = self.cache.get(request)
            if hit is not None:
                trace.cache_hit = True
                trace.ok = True
                trace.outcome = "cache_hit"
                trace.served_by = "cache"
                trace.duration_ms = (self._now() - started) * 1000
                self.traces.finish(trace)
                hit.cached = True
                hit.latency_ms = trace.duration_ms
                return hit

        # 2. Budget — refuse before spending, not after.
        try:
            reservation = self.budget.reserve(
                request.api_key_id, self._reservation_amount(request, chain)
            )
        except BudgetExceeded:
            trace.outcome = "budget_exceeded"
            trace.duration_ms = (self._now() - started) * 1000
            self.traces.finish(trace)
            raise

        # 2b. Redact personal data. The provider sees placeholders; the caller
        #     gets the real values back, so neither side notices the swap.
        outbound = request
        redaction: Redaction | None = None
        if self.redactor is not None:
            outbound, redaction = _redact_request(self.redactor, request)
            if redaction.replacements:
                trace.guardrails.append(
                    f"redacted {len(redaction.replacements)} value(s): {', '.join(redaction.found)}"
                )

        settled = False
        failures: list[tuple[str, str]] = []
        try:
            for route in chain:
                breaker = self.breaker_for(route)
                if not breaker.allows_request():
                    trace.skipped.append(route.name)
                    wait = breaker.retry_in_s()
                    failures.append((route.name, f"circuit open, retry in {wait:.0f}s"))
                    continue

                response = await self._try_route(route, outbound, trace, failures, breaker)
                if response is None:
                    continue

                if redaction is not None and redaction.replacements:
                    response.text = redaction.restore(response.text)

                cost = self.price_book.cost(response.model, response.usage)
                self.budget.commit(reservation, cost)
                settled = True

                response.cost_usd = cost
                response.latency_ms = (self._now() - started) * 1000
                if self.cache is not None:
                    self.cache.put(request, response)

                trace.ok = True
                trace.outcome = "ok"
                trace.served_by = route.name
                trace.cost_usd = cost
                trace.input_tokens = response.usage.input_tokens
                trace.output_tokens = response.usage.output_tokens
                trace.duration_ms = response.latency_ms
                self.latency.record(trace.duration_ms)
                self._route_window(route).record(trace.duration_ms)
                self.traces.finish(trace)
                return response

            trace.outcome = "all_providers_failed"
            trace.duration_ms = (self._now() - started) * 1000
            self.traces.finish(trace)
            raise AllProvidersFailed(failures)
        finally:
            if not settled:
                # Nothing was delivered, so nothing should be charged.
                self.budget.release(reservation)

    async def _try_route(
        self,
        route: Route,
        request: CompletionRequest,
        trace: Trace,
        failures: list[tuple[str, str]],
        breaker: CircuitBreaker,
    ) -> CompletionResponse | None:
        """Attempt one route, with retries. None means move to the next route."""
        attempt_request = (
            request if request.model == route.model else _with_model(request, route.model)
        )
        last_error: ProviderError | None = None

        for attempt in range(self.retry_policy.max_retries + 1):
            backoff = 0.0
            if attempt > 0:
                backoff = self.retry_policy.delay_for(attempt - 1, last_error, self._rng)
                if backoff > 0:
                    await self._sleep(backoff)

            record = Attempt(
                provider=route.provider.name,
                model=route.model,
                started_at=self._now(),
                retry=attempt,
                backoff_s=backoff,
            )
            call_started = self._now()
            try:
                response = await self._call(route, attempt_request)
            except ProviderError as error:
                record.duration_ms = (self._now() - call_started) * 1000
                record.error = str(error)
                record.error_kind = error.kind.value
                trace.attempts.append(record)
                breaker.record_failure()
                last_error = error
                if not self.retry_policy.should_retry(error, attempt):
                    failures.append((route.name, str(error)))
                    return None
                continue

            record.duration_ms = (self._now() - call_started) * 1000

            # A response that fails its schema is a failed response. Treating
            # it as success would hand the caller something it cannot parse.
            if attempt_request.response_schema is not None:
                try:
                    response.parsed = parse_structured(
                        response.text, attempt_request.response_schema
                    )
                except SchemaError as error:
                    record.error = f"schema: {error}"
                    record.error_kind = FailureKind.SCHEMA_INVALID.value
                    trace.attempts.append(record)
                    # The provider answered, so it is healthy — this is a
                    # formatting problem and must not trip its breaker.
                    last_error = ProviderError(
                        str(error), FailureKind.SCHEMA_INVALID, provider=route.provider.name
                    )
                    if not self.retry_policy.should_retry(last_error, attempt):
                        failures.append((route.name, f"schema: {error}"))
                        return None
                    continue

            record.ok = True
            trace.attempts.append(record)
            breaker.record_success()
            return response

        failures.append((route.name, str(last_error) if last_error else "retries exhausted"))
        return None

    async def _call(self, route: Route, request: CompletionRequest) -> CompletionResponse:
        """Invoke the provider, turning a timeout into a typed failure."""
        if self.request_timeout_s is None:
            return await route.provider.complete(request)
        try:
            return await asyncio.wait_for(
                route.provider.complete(request), timeout=self.request_timeout_s
            )
        except TimeoutError as exc:
            raise ProviderError(
                f"no response within {self.request_timeout_s}s",
                FailureKind.TIMEOUT,
                provider=route.provider.name,
            ) from exc

    def _route_window(self, route: Route) -> LatencyWindow:
        if route.name not in self.route_latency:
            self.route_latency[route.name] = LatencyWindow()
        return self.route_latency[route.name]

    # ------------------------------------------------------------------- views

    def stats(self) -> dict[str, Any]:
        return {
            "traffic": self.traces.summary(),
            "latency_ms": self.latency.stats(),
            "latency_by_route": {
                name: window.stats() for name, window in self.route_latency.items()
            },
            "breakers": [b.stats() for b in self._breakers.values()],
            "budgets": self.budget.snapshot(),
        }


def _with_model(request: CompletionRequest, model: str) -> CompletionRequest:
    """A copy of the request aimed at a different model, for fallback."""
    return CompletionRequest(
        model=model,
        messages=request.messages,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        response_schema=request.response_schema,
        api_key_id=request.api_key_id,
        tags=request.tags,
    )


def _user_text(request: CompletionRequest) -> str:
    """Only user and tool content is screened; the system prompt is ours."""
    from .types import Role

    return "\n".join(
        m.content for m in request.messages if m.role in (Role.USER, Role.TOOL)
    )


def _redact_request(
    redactor: Redactor, request: CompletionRequest
) -> tuple[CompletionRequest, Redaction]:
    """Redact every message against one shared placeholder map."""
    from .types import Message

    texts, redaction = redactor.redact_many([m.content for m in request.messages])
    messages = tuple(
        Message(original.role, text) for original, text in zip(request.messages, texts, strict=True)
    )
    return (
        CompletionRequest(
            model=request.model,
            messages=messages,
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
            response_schema=request.response_schema,
            api_key_id=request.api_key_id,
            tags=request.tags,
        ),
        redaction,
    )
