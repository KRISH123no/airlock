"""Circuit breaker state machine."""

import pytest

from airlock.core.breaker import CircuitBreaker, State


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def clock():
    return Clock()


def breaker(clock, **kwargs):
    defaults = dict(minimum_volume=4, failure_rate_threshold=0.5, cooldown_s=30.0)
    defaults.update(kwargs)
    return CircuitBreaker("provider", now=clock, **defaults)


def test_starts_closed_and_allows_traffic(clock):
    cb = breaker(clock)
    assert cb.state is State.CLOSED
    assert cb.allows_request() is True


def test_a_few_failures_do_not_trip_it(clock):
    """Two failures out of two is a 100% failure rate and means nothing."""
    cb = breaker(clock)
    cb.record_failure()
    cb.record_failure()
    assert cb.state is State.CLOSED


def test_trips_once_the_failure_rate_holds_over_enough_calls(clock):
    cb = breaker(clock)
    for _ in range(4):
        cb.record_failure()
    assert cb.state is State.OPEN
    assert cb.allows_request() is False


def test_a_healthy_mix_stays_closed(clock):
    cb = breaker(clock, failure_rate_threshold=0.5, minimum_volume=4)
    for _ in range(8):
        cb.record_success()
    for _ in range(3):
        cb.record_failure()
    assert cb.state is State.CLOSED


def test_old_failures_age_out_of_the_window(clock):
    cb = breaker(clock, window_s=60.0)
    for _ in range(3):
        cb.record_failure()
    clock.advance(61)
    cb.record_failure()  # alone in the window now
    assert cb.state is State.CLOSED


def test_goes_half_open_after_the_cooldown(clock):
    cb = breaker(clock)
    for _ in range(4):
        cb.record_failure()
    assert cb.state is State.OPEN
    assert cb.retry_in_s() == pytest.approx(30.0)

    clock.advance(30)
    assert cb.state is State.HALF_OPEN
    assert cb.retry_in_s() == 0.0


def test_half_open_admits_only_a_few_probes(clock):
    cb = breaker(clock, half_open_probes=2)
    for _ in range(4):
        cb.record_failure()
    clock.advance(31)
    assert cb.allows_request() is True
    assert cb.allows_request() is True
    assert cb.allows_request() is False, "a third probe would defeat the point"


def test_successful_probes_close_the_circuit(clock):
    cb = breaker(clock, half_open_probes=2)
    for _ in range(4):
        cb.record_failure()
    clock.advance(31)
    cb.allows_request()
    cb.record_success()
    cb.allows_request()
    cb.record_success()
    assert cb.state is State.CLOSED
    assert cb.allows_request() is True


def test_one_failed_probe_reopens_the_circuit(clock):
    cb = breaker(clock, half_open_probes=2)
    for _ in range(4):
        cb.record_failure()
    clock.advance(31)
    cb.allows_request()
    cb.record_failure()
    assert cb.state is State.OPEN
    assert cb.retry_in_s() == pytest.approx(30.0), "the cooldown restarts"


def test_stats_describe_the_window(clock):
    cb = breaker(clock)
    cb.record_success()
    cb.record_failure()
    stats = cb.stats()
    assert stats["provider"] == "provider"
    assert stats["calls_in_window"] == 2
    assert stats["failure_rate"] == pytest.approx(0.5)
