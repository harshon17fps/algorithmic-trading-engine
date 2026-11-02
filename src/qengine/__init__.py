"""qengine — an event-driven, multi-asset trading engine and backtester."""

from qengine.engine import BacktestResult, EventLoop
from qengine.events import (
    Direction,
    EventType,
    FillEvent,
    FinancingEvent,
    MarketEvent,
    OrderEvent,
    Side,
    SignalEvent,
    Universe,
)

__all__ = [
    "BacktestResult",
    "Direction",
    "EventLoop",
    "EventType",
    "FillEvent",
    "FinancingEvent",
    "MarketEvent",
    "OrderEvent",
    "Side",
    "SignalEvent",
    "Universe",
]
__version__ = "0.1.0"
