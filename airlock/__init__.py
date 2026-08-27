"""airlock — a gateway that makes LLM calls reliable, affordable and observable.

    >>> from airlock import Gateway, Route, FakeProvider, CompletionRequest, Message, Role
    >>> gateway = Gateway({"medium": [Route(FakeProvider(script=["hello"]), "medium")]})
    >>> import asyncio
    >>> response = asyncio.run(gateway.complete(
    ...     CompletionRequest(model="medium", messages=(Message(Role.USER, "hi"),))))
    >>> response.text
    'hello'
"""

from .core.breaker import CircuitBreaker, State
from .core.budget import BudgetLedger
from .core.cache import ExactCache, SemanticCache, TrigramEmbedder
from .core.cost import ModelPrice, PriceBook
from .core.latency import LatencyWindow
from .core.retry import RetryPolicy
from .errors import (
    AirlockError,
    AllProvidersFailed,
    BudgetExceeded,
    CircuitOpen,
    GuardrailTripped,
    ProviderError,
)
from .gateway import Gateway
from .providers.base import Provider, Route
from .providers.fake import FailingProvider, FakeProvider, SlowProvider
from .trace import Attempt, Trace, TraceStore
from .types import (
    CompletionRequest,
    CompletionResponse,
    FailureKind,
    Message,
    Role,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "AirlockError",
    "AllProvidersFailed",
    "Attempt",
    "BudgetExceeded",
    "BudgetLedger",
    "CircuitBreaker",
    "CircuitOpen",
    "CompletionRequest",
    "CompletionResponse",
    "ExactCache",
    "FailingProvider",
    "FailureKind",
    "FakeProvider",
    "Gateway",
    "GuardrailTripped",
    "LatencyWindow",
    "Message",
    "ModelPrice",
    "PriceBook",
    "Provider",
    "ProviderError",
    "RetryPolicy",
    "Role",
    "Route",
    "SemanticCache",
    "SlowProvider",
    "State",
    "Trace",
    "TraceStore",
    "TrigramEmbedder",
    "Usage",
]
