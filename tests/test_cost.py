"""Pricing and token accounting."""

import pytest

from airlock.core.cost import DEFAULT_PRICES, ModelPrice, PriceBook, estimate_tokens
from airlock.types import Usage


def test_output_tokens_cost_more_than_input():
    price = ModelPrice(input_per_m=1.0, output_per_m=10.0)
    input_only = price.cost(Usage(input_tokens=1_000_000))
    output_only = price.cost(Usage(output_tokens=1_000_000))
    assert input_only == pytest.approx(1.0)
    assert output_only == pytest.approx(10.0)


def test_cached_input_is_billed_at_the_cheaper_rate():
    price = ModelPrice(input_per_m=10.0, output_per_m=10.0, cached_input_per_m=1.0)
    usage = Usage(input_tokens=1_000_000, cached_input_tokens=900_000)
    # 100k fresh at $10/M plus 900k cached at $1/M
    assert price.cost(usage) == pytest.approx(1.0 + 0.9)


def test_cached_tokens_cannot_exceed_input_tokens():
    price = ModelPrice(input_per_m=10.0, output_per_m=0.0, cached_input_per_m=0.0)
    usage = Usage(input_tokens=100, cached_input_tokens=999_999)
    assert price.cost(usage) == pytest.approx(0.0)


def test_a_model_with_no_price_is_not_free():
    book = PriceBook(prices={})
    assert book.cost("mystery-model", Usage(input_tokens=1_000_000)) > 0


def test_registering_a_price_overrides_the_default():
    book = PriceBook(prices={})
    book.register("mine", ModelPrice(input_per_m=2.0, output_per_m=2.0))
    assert book.cost("mine", Usage(input_tokens=1_000_000)) == pytest.approx(2.0)


def test_worst_case_assumes_every_output_token_is_used():
    book = PriceBook()
    worst = book.worst_case_cost("medium", input_tokens=1000, max_output_tokens=4000)
    actual = book.cost("medium", Usage(input_tokens=1000, output_tokens=10))
    assert worst > actual


def test_usage_adds():
    total = Usage(10, 20, 5) + Usage(1, 2, 1)
    assert (total.input_tokens, total.output_tokens, total.cached_input_tokens) == (11, 22, 6)
    assert total.total_tokens == 33


@pytest.mark.parametrize("text,expected", [("", 0), ("a", 1), ("abcd", 1), ("abcde", 2)])
def test_token_estimate(text, expected):
    assert estimate_tokens(text) == expected


def test_default_prices_are_ordered_by_tier():
    assert DEFAULT_PRICES["small"].output_per_m < DEFAULT_PRICES["medium"].output_per_m
    assert DEFAULT_PRICES["medium"].output_per_m < DEFAULT_PRICES["large"].output_per_m
