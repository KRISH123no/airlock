"""What a request costs.

Prices are per million tokens and differ for input and output, which is the
detail most naive cost tracking gets wrong: output tokens are typically three
to five times more expensive, so a request's cost is not proportional to its
length. Cached input is cheaper again.

Prices here are illustrative defaults. They are configuration, not truth —
override them from your own price list, because vendor pricing changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..types import Usage


@dataclass(frozen=True)
class ModelPrice:
    """US dollars per million tokens."""

    input_per_m: float
    output_per_m: float
    #: Most vendors bill a cache hit at a fraction of the input price.
    cached_input_per_m: float | None = None

    def cost(self, usage: Usage) -> float:
        cached = min(usage.cached_input_tokens, usage.input_tokens)
        fresh = usage.input_tokens - cached
        cached_rate = (
            self.cached_input_per_m
            if self.cached_input_per_m is not None
            else self.input_per_m
        )
        return (
            fresh * self.input_per_m
            + cached * cached_rate
            + usage.output_tokens * self.output_per_m
        ) / 1_000_000


DEFAULT_PRICES: dict[str, ModelPrice] = {
    "small": ModelPrice(input_per_m=0.25, output_per_m=1.25, cached_input_per_m=0.03),
    "medium": ModelPrice(input_per_m=3.00, output_per_m=15.00, cached_input_per_m=0.30),
    "large": ModelPrice(input_per_m=15.00, output_per_m=75.00, cached_input_per_m=1.50),
}


class PriceBook:
    """Model name to price, with a fallback so an unknown model is not free."""

    def __init__(
        self,
        prices: dict[str, ModelPrice] | None = None,
        *,
        default: ModelPrice | None = None,
    ) -> None:
        self.prices = dict(prices if prices is not None else DEFAULT_PRICES)
        # An unpriced model must not silently cost nothing — that is how a
        # budget gets blown by the one model nobody added to the table.
        self.default = default or ModelPrice(input_per_m=15.0, output_per_m=75.0)

    def price_for(self, model: str) -> ModelPrice:
        return self.prices.get(model, self.default)

    def cost(self, model: str, usage: Usage) -> float:
        return self.price_for(model).cost(usage)

    def worst_case_cost(self, model: str, input_tokens: int, max_output_tokens: int) -> float:
        """The most a request could cost, used to reserve budget before calling.

        Reserving the ceiling is what stops many concurrent requests from each
        seeing "plenty of budget left" and collectively overspending.
        """
        return self.cost(model, Usage(input_tokens=input_tokens, output_tokens=max_output_tokens))

    def register(self, model: str, price: ModelPrice) -> None:
        self.prices[model] = price


def estimate_tokens(text: str) -> int:
    """A cheap token estimate for budgeting before the real count is known.

    Roughly four characters per token for English, with a floor of one token
    for any non-empty string. This is for reserving budget, never for billing:
    the provider's reported usage is what gets committed.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
