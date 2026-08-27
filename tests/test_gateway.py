"""The gateway: routing, retries, fallback, breakers, budget and traces."""

import asyncio

import pytest

from airlock import (
    BudgetExceeded,
    CompletionRequest,
    ExactCache,
    FailingProvider,
    FakeProvider,
    Gateway,
    Message,
    ProviderError,
    RetryPolicy,
    Role,
    Route,
    SlowProvider,
)
from airlock.errors import AllProvidersFailed
from airlock.types import FailureKind


class Clock:
    """Advances only when the code under test sleeps."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    async def sleep(self, seconds):
        self.t += seconds
        await asyncio.sleep(0)

    def advance(self, seconds):
        self.t += seconds


def ask(text="what is a b+ tree?", **kwargs):
    return CompletionRequest(model="medium", messages=(Message(Role.USER, text),), **kwargs)


def build(chain, **kwargs):
    kwargs.setdefault("retry_policy", RetryPolicy(max_retries=2, base_delay_s=0.0))
    kwargs.setdefault("sleep", lambda s: asyncio.sleep(0))
    return Gateway({"medium": chain}, **kwargs)


# ------------------------------------------------------------------ happy path


async def test_a_successful_call_returns_the_answer_and_prices_it():
    provider = FakeProvider("primary", default_text="a sorted tree on disk")
    gateway = build([Route(provider, "medium")])
    response = await gateway.complete(ask())

    assert response.text == "a sorted tree on disk"
    assert response.provider == "primary"
    assert response.cost_usd > 0
    assert response.usage.input_tokens > 0
    assert provider.call_count == 1


async def test_an_unknown_model_is_rejected_before_anything_happens():
    gateway = build([Route(FakeProvider(), "medium")])
    with pytest.raises(ProviderError) as caught:
        await gateway.complete(CompletionRequest(model="nope", messages=()))
    assert caught.value.kind is FailureKind.BAD_REQUEST


async def test_a_gateway_needs_routes():
    with pytest.raises(ValueError):
        Gateway({})


# ---------------------------------------------------------------------- retry


async def test_a_transient_failure_is_retried_on_the_same_provider():
    provider = FakeProvider(
        "flaky",
        script=[
            ProviderError("503", FailureKind.SERVER_ERROR),
            ProviderError("503", FailureKind.SERVER_ERROR),
            "third time lucky",
        ],
    )
    gateway = build([Route(provider, "medium")])
    response = await gateway.complete(ask())

    assert response.text == "third time lucky"
    assert provider.call_count == 3
    assert gateway.traces.recent(1)[0].retries == 2


async def test_a_permanent_failure_is_not_retried():
    provider = FakeProvider("strict", script=[ProviderError("bad json", FailureKind.BAD_REQUEST)])
    gateway = build([Route(provider, "medium")])
    with pytest.raises(AllProvidersFailed):
        await gateway.complete(ask())
    assert provider.call_count == 1, "retrying a malformed request cannot help"


async def test_retries_are_bounded():
    provider = FailingProvider("down", FailureKind.SERVER_ERROR)
    policy = RetryPolicy(max_retries=1, base_delay_s=0.0)
    gateway = build([Route(provider, "medium")], retry_policy=policy)
    with pytest.raises(AllProvidersFailed):
        await gateway.complete(ask())
    assert provider.call_count == 2


async def test_backoff_is_actually_waited_and_recorded():
    clock = Clock()
    provider = FakeProvider(
        "flaky", script=[ProviderError("503", FailureKind.SERVER_ERROR), "ok"]
    )
    gateway = Gateway(
        {"medium": [Route(provider, "medium")]},
        retry_policy=RetryPolicy(max_retries=2, base_delay_s=4.0),
        now=clock,
        sleep=clock.sleep,
    )
    await gateway.complete(ask())
    trace = gateway.traces.recent(1)[0]
    assert trace.attempts[1].backoff_s > 0
    assert clock.t > 0, "the gateway really waited"


# ------------------------------------------------------------------- fallback


async def test_it_falls_back_to_the_next_provider():
    primary = FailingProvider("primary", FailureKind.SERVER_ERROR)
    backup = FakeProvider("backup", default_text="served by the spare")
    gateway = build([Route(primary, "medium"), Route(backup, "small")])

    response = await gateway.complete(ask())
    assert response.text == "served by the spare"
    assert response.provider == "backup"


async def test_a_fallback_asks_for_its_own_model_not_the_original():
    primary = FailingProvider("primary", FailureKind.SERVER_ERROR)
    backup = FakeProvider("backup")
    gateway = build([Route(primary, "medium"), Route(backup, "small")])

    await gateway.complete(ask())
    assert backup.calls[0].model == "small"


async def test_a_permanent_failure_still_falls_through_to_the_backup():
    """No point retrying a bad request here, but another vendor may accept it."""
    primary = FailingProvider("primary", FailureKind.CONTENT_FILTER)
    backup = FakeProvider("backup", default_text="allowed")
    gateway = build([Route(primary, "medium"), Route(backup, "small")])

    assert (await gateway.complete(ask())).text == "allowed"
    assert primary.call_count == 1


async def test_when_everything_fails_the_error_names_every_attempt():
    a = FailingProvider("alpha", FailureKind.SERVER_ERROR, "alpha exploded")
    b = FailingProvider("beta", FailureKind.TIMEOUT, "beta timed out")
    gateway = build([Route(a, "medium"), Route(b, "small")])

    with pytest.raises(AllProvidersFailed) as caught:
        await gateway.complete(ask())
    message = str(caught.value)
    assert "alpha exploded" in message and "beta timed out" in message
    assert len(caught.value.attempts) == 2


# -------------------------------------------------------------------- breaker


async def test_a_failing_provider_is_eventually_skipped_without_being_called():
    clock = Clock()
    broken = FailingProvider("broken", FailureKind.SERVER_ERROR)
    backup = FakeProvider("backup", default_text="fine")
    gateway = Gateway(
        {"medium": [Route(broken, "medium"), Route(backup, "small")]},
        retry_policy=RetryPolicy(max_retries=1, base_delay_s=0.0),
        breaker_options={"minimum_volume": 4, "failure_rate_threshold": 0.5, "cooldown_s": 60.0},
        now=clock,
        sleep=clock.sleep,
    )

    for _ in range(3):
        await gateway.complete(ask())
    calls_before = broken.call_count
    await gateway.complete(ask())
    assert broken.call_count == calls_before, "the circuit is open; do not call it"
    assert "broken/medium" in gateway.traces.recent(1)[0].skipped


async def test_the_breaker_lets_traffic_back_after_the_cooldown():
    clock = Clock()
    healthy = False

    def behaviour(_request):
        if not healthy:
            raise ProviderError("503", FailureKind.SERVER_ERROR)
        return "better now"

    provider = FakeProvider("recovering", default_behaviour=behaviour)
    gateway = Gateway(
        {"medium": [Route(provider, "medium")]},
        retry_policy=RetryPolicy(max_retries=1, base_delay_s=0.0),
        breaker_options={"minimum_volume": 4, "cooldown_s": 30.0, "half_open_probes": 1},
        now=clock,
        sleep=clock.sleep,
    )
    for _ in range(3):
        with pytest.raises(AllProvidersFailed):
            await gateway.complete(ask())
    assert gateway.stats()["breakers"][0]["state"] == "open"

    healthy = True
    clock.advance(31)
    response = await gateway.complete(ask())
    assert response.text == "better now"
    assert gateway.stats()["breakers"][0]["state"] == "closed"


# --------------------------------------------------------------------- budget


async def test_a_call_over_budget_never_reaches_the_provider():
    provider = FakeProvider("primary")
    gateway = build([Route(provider, "medium")])
    gateway.budget.set_limit("broke", 0.0000001)

    with pytest.raises(BudgetExceeded):
        await gateway.complete(ask(api_key_id="broke"))
    assert provider.call_count == 0, "refusing early is the point"
    assert gateway.traces.recent(1)[0].outcome == "budget_exceeded"


async def test_a_failed_call_costs_nothing():
    gateway = build([Route(FailingProvider("down", FailureKind.SERVER_ERROR), "medium")])
    gateway.budget.set_limit("team", 10.0)

    with pytest.raises(AllProvidersFailed):
        await gateway.complete(ask(api_key_id="team"))
    assert gateway.budget.spent("team") == 0.0
    assert gateway.budget.remaining("team") == pytest.approx(10.0), "the reservation was released"


async def test_the_budget_is_charged_the_actual_cost_not_the_reservation():
    gateway = build([Route(FakeProvider("p", default_text="short"), "medium")])
    gateway.budget.set_limit("team", 10.0)

    response = await gateway.complete(ask(api_key_id="team", max_output_tokens=4000))
    spent = gateway.budget.spent("team")
    assert spent == pytest.approx(response.cost_usd)
    assert spent < 0.01, "billed for the tokens used, not the 4000 reserved"


async def test_the_reservation_covers_the_priciest_model_in_the_chain():
    """A fallback to a dearer model must not slip past the limit."""
    cheap = FailingProvider("cheap", FailureKind.SERVER_ERROR)
    dear = FakeProvider("dear")
    gateway = build([Route(cheap, "small"), Route(dear, "large")])
    # Enough for the small model's worst case, not for the large one's.
    small_worst = gateway.price_book.worst_case_cost("small", 20, 1024)
    gateway.budget.set_limit("team", small_worst * 2)

    with pytest.raises(BudgetExceeded):
        await gateway.complete(ask(api_key_id="team"))
    assert dear.call_count == 0


async def test_spending_accumulates_across_requests():
    gateway = build([Route(FakeProvider("p"), "medium")])
    gateway.budget.set_limit("team", 10.0)
    for _ in range(3):
        await gateway.complete(ask(f"question {_}", api_key_id="team"))
    assert gateway.budget.spent("team") > 0
    assert len(gateway.traces) == 3


# ---------------------------------------------------------------------- cache


async def test_a_cache_hit_skips_the_provider_and_the_budget():
    provider = FakeProvider("primary", default_text="cached answer")
    gateway = build([Route(provider, "medium")], cache=ExactCache())
    gateway.budget.set_limit("team", 10.0)

    first = await gateway.complete(ask(api_key_id="team"))
    spent_after_first = gateway.budget.spent("team")
    second = await gateway.complete(ask(api_key_id="team"))

    assert second.text == first.text
    assert second.cached is True
    assert provider.call_count == 1
    assert gateway.budget.spent("team") == spent_after_first, "a cache hit is free"
    assert gateway.traces.recent(1)[0].outcome == "cache_hit"


async def test_different_prompts_do_not_share_a_cache_entry():
    provider = FakeProvider("primary")
    gateway = build([Route(provider, "medium")], cache=ExactCache())
    await gateway.complete(ask("first question"))
    await gateway.complete(ask("second question"))
    assert provider.call_count == 2


# -------------------------------------------------------------------- timeout


async def test_a_slow_provider_times_out_and_the_backup_answers():
    """Real sleeps here, deliberately.

    The timeout is enforced by asyncio.wait_for, which reads the event loop's
    own clock — a fake clock cannot drive it. So this uses real durations,
    kept to milliseconds.
    """
    slow = SlowProvider("slow", seconds=0.5)
    backup = FakeProvider("backup", default_text="quick")
    gateway = Gateway(
        {"medium": [Route(slow, "medium"), Route(backup, "small")]},
        retry_policy=RetryPolicy(max_retries=0),
        request_timeout_s=0.01,
    )
    response = await gateway.complete(ask())
    assert response.text == "quick"
    trace = gateway.traces.recent(1)[0]
    assert trace.attempts[0].error_kind == "timeout"


async def test_a_timeout_can_be_disabled():
    slow = SlowProvider("slow", seconds=0.02)
    gateway = Gateway({"medium": [Route(slow, "medium")]}, request_timeout_s=None)
    assert (await gateway.complete(ask())).text == "eventually"


# --------------------------------------------------------------------- traces


async def test_the_trace_records_every_attempt_including_failures():
    provider = FakeProvider(
        "flaky", script=[ProviderError("503", FailureKind.SERVER_ERROR), "ok"]
    )
    gateway = build([Route(provider, "medium")])
    await gateway.complete(ask(tags=("checkout",)))

    trace = gateway.traces.recent(1)[0]
    assert trace.ok is True
    assert trace.tags == ("checkout",)
    assert len(trace.attempts) == 2
    assert trace.attempts[0].ok is False
    assert trace.attempts[1].ok is True
    assert "503" in trace.explain()
    assert trace.to_dict()["retries"] == 1


async def test_traces_are_retrievable_by_id_and_bounded():
    gateway = build([Route(FakeProvider("p"), "medium")])
    await gateway.complete(ask())
    trace = gateway.traces.recent(1)[0]
    assert gateway.traces.get(trace.id) is trace
    assert gateway.traces.get(999_999) is None


async def test_stats_report_traffic_latency_breakers_and_budgets():
    gateway = build([Route(FakeProvider("p"), "medium")])
    gateway.budget.set_limit("team", 5.0)
    await gateway.complete(ask(api_key_id="team"))

    stats = gateway.stats()
    assert stats["traffic"]["requests"] == 1
    assert stats["traffic"]["success_rate"] == 1.0
    assert stats["latency_ms"]["count"] == 1
    assert "p/medium" in stats["latency_by_route"]
    assert stats["budgets"]["team"]["limit_usd"] == 5.0
    assert stats["breakers"][0]["state"] == "closed"


async def test_the_summary_counts_failures_and_cache_hits():
    gateway = build([Route(FailingProvider("down", FailureKind.SERVER_ERROR), "medium")])
    with pytest.raises(AllProvidersFailed):
        await gateway.complete(ask())
    summary = gateway.traces.summary()
    assert summary["failures"] == 1
    assert summary["success_rate"] == 0.0
