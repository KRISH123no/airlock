"""Spending limits, including the concurrency case that motivates reservations."""

import pytest

from airlock.core.budget import BudgetLedger
from airlock.errors import BudgetExceeded


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def test_a_key_with_no_limit_is_unlimited():
    ledger = BudgetLedger()
    assert ledger.remaining("anyone") == float("inf")
    ledger.commit(ledger.reserve("anyone", 999.0), 999.0)
    assert ledger.spent("anyone") == 999.0


def test_commit_records_the_actual_not_the_reserved_amount():
    ledger = BudgetLedger()
    ledger.set_limit("k", 10.0)
    reservation = ledger.reserve("k", 5.0)
    ledger.commit(reservation, 0.25)
    assert ledger.spent("k") == pytest.approx(0.25)
    assert ledger.remaining("k") == pytest.approx(9.75)


def test_release_gives_the_money_back():
    ledger = BudgetLedger()
    ledger.set_limit("k", 10.0)
    reservation = ledger.reserve("k", 4.0)
    assert ledger.remaining("k") == pytest.approx(6.0)
    ledger.release(reservation)
    assert ledger.remaining("k") == pytest.approx(10.0)
    assert ledger.spent("k") == 0.0


def test_reservations_stop_concurrent_requests_from_overspending():
    """The whole point: two in-flight requests must not both see full budget."""
    ledger = BudgetLedger()
    ledger.set_limit("k", 10.0)
    first = ledger.reserve("k", 6.0)
    with pytest.raises(BudgetExceeded):
        ledger.reserve("k", 6.0)
    ledger.commit(first, 1.0)
    # Once the real cost is known, the headroom comes back.
    ledger.reserve("k", 6.0)


def test_exceeding_the_limit_names_the_key_and_the_numbers():
    ledger = BudgetLedger()
    ledger.set_limit("team-a", 1.0)
    ledger.commit(ledger.reserve("team-a", 1.0), 1.0)
    with pytest.raises(BudgetExceeded) as caught:
        ledger.reserve("team-a", 0.01)
    assert caught.value.key_id == "team-a"
    assert caught.value.limit_usd == 1.0
    assert "team-a" in str(caught.value)


def test_the_window_rolls_rather_than_resetting():
    clock = Clock()
    ledger = BudgetLedger(now=clock)
    ledger.set_limit("k", 10.0, window_s=100.0)
    ledger.commit(ledger.reserve("k", 9.0), 9.0)
    assert ledger.remaining("k") == pytest.approx(1.0)

    clock.advance(50)
    assert ledger.remaining("k") == pytest.approx(1.0), "still inside the window"

    clock.advance(51)
    assert ledger.remaining("k") == pytest.approx(10.0), "the old spend aged out"


def test_settling_twice_is_harmless():
    ledger = BudgetLedger()
    ledger.set_limit("k", 5.0)
    reservation = ledger.reserve("k", 1.0)
    ledger.commit(reservation, 1.0)
    ledger.release(reservation)
    ledger.commit(reservation, 99.0)
    assert ledger.spent("k") == pytest.approx(1.0)


def test_budgets_are_per_key():
    ledger = BudgetLedger()
    ledger.set_limit("a", 1.0)
    ledger.set_limit("b", 1.0)
    ledger.commit(ledger.reserve("a", 1.0), 1.0)
    assert ledger.remaining("a") == 0.0
    assert ledger.remaining("b") == pytest.approx(1.0)


def test_snapshot_reports_every_known_key():
    ledger = BudgetLedger()
    ledger.set_limit("a", 2.0)
    ledger.commit(ledger.reserve("a", 0.5), 0.5)
    ledger.reserve("a", 0.25)
    snapshot = ledger.snapshot()["a"]
    assert snapshot["spent_usd"] == pytest.approx(0.5)
    assert snapshot["reserved_usd"] == pytest.approx(0.25)
    assert snapshot["remaining_usd"] == pytest.approx(1.25)
