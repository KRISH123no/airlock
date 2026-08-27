"""Retry policy and latency percentiles."""

import random

import pytest

from airlock.core.latency import LatencyWindow
from airlock.core.retry import RetryPolicy
from airlock.errors import ProviderError
from airlock.types import FailureKind


def error(kind, retry_after=None):
    return ProviderError("boom", kind, retry_after_s=retry_after)


@pytest.mark.parametrize(
    "kind,retryable",
    [
        (FailureKind.TIMEOUT, True),
        (FailureKind.RATE_LIMIT, True),
        (FailureKind.SERVER_ERROR, True),
        (FailureKind.CONNECTION, True),
        (FailureKind.BAD_REQUEST, False),
        (FailureKind.AUTHENTICATION, False),
        (FailureKind.CONTENT_FILTER, False),
        (FailureKind.CONTEXT_LENGTH, False),
    ],
)
def test_only_transient_failures_are_retried(kind, retryable):
    policy = RetryPolicy(max_retries=3)
    assert policy.should_retry(error(kind), attempt=0) is retryable


def test_retries_stop_at_the_limit():
    policy = RetryPolicy(max_retries=2)
    assert policy.should_retry(error(FailureKind.TIMEOUT), attempt=1) is True
    assert policy.should_retry(error(FailureKind.TIMEOUT), attempt=2) is False


def test_backoff_ceiling_grows_exponentially_and_is_capped():
    policy = RetryPolicy(base_delay_s=1.0, multiplier=2.0, max_delay_s=5.0)
    rng = random.Random(0)
    ceilings = []
    for attempt in range(5):
        samples = [policy.delay_for(attempt, rng=rng) for _ in range(200)]
        ceilings.append(max(samples))
    assert ceilings[0] <= 1.0
    assert ceilings[1] <= 2.0
    assert ceilings[2] <= 4.0
    assert all(c <= 5.0 for c in ceilings), "the cap holds"
    assert ceilings[1] > ceilings[0], "and it does grow"


def test_full_jitter_spreads_retries_out():
    """Fixed backoff would synchronise every client onto the same instant."""
    policy = RetryPolicy(base_delay_s=1.0)
    rng = random.Random(7)
    delays = [policy.delay_for(2, rng=rng) for _ in range(500)]
    assert min(delays) < 0.5 < max(delays)
    assert len(set(delays)) > 400


def test_a_provider_retry_after_overrides_the_computed_backoff():
    policy = RetryPolicy(base_delay_s=1.0)
    assert policy.delay_for(0, error(FailureKind.RATE_LIMIT, retry_after=7.5)) == 7.5


def test_an_absurd_retry_after_is_capped():
    policy = RetryPolicy(max_retry_after_s=60.0)
    assert policy.delay_for(0, error(FailureKind.RATE_LIMIT, retry_after=99999)) == 60.0


def test_a_negative_retry_after_does_not_go_backwards():
    policy = RetryPolicy()
    assert policy.delay_for(0, error(FailureKind.RATE_LIMIT, retry_after=-5)) == 0.0


def test_percentiles_return_values_that_actually_occurred():
    window = LatencyWindow()
    for value in range(1, 101):
        window.record(float(value))
    assert window.percentile(50) == 50
    assert window.percentile(95) == 95
    assert window.percentile(99) == 99
    assert window.percentile(100) == 100


def test_percentiles_are_not_fooled_by_one_slow_request():
    """The reason we track p99 instead of the mean."""
    window = LatencyWindow()
    for _ in range(99):
        window.record(200.0)
    window.record(30_000.0)
    assert window.percentile(50) == 200.0
    assert window.percentile(99) == 200.0
    assert window.stats()["max"] == 30_000.0
    assert window.mean() > 400.0, "the mean is dragged up, which is the point"


def test_an_empty_window_reports_zeros():
    window = LatencyWindow()
    assert window.percentile(95) == 0.0
    assert window.stats() == {
        "count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0,
    }


def test_the_window_is_bounded():
    window = LatencyWindow(capacity=10)
    for value in range(100):
        window.record(float(value))
    assert len(window) == 10
    assert window.percentile(50) >= 90, "only the most recent samples remain"


def test_percentile_bounds_are_validated():
    with pytest.raises(ValueError):
        LatencyWindow().percentile(101)
