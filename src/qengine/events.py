"""Immutable event types that flow through the engine's queue.

Every component communicates exclusively through these messages, which is what
lets the same Strategy / Portfolio / ExecutionHandler code run against a
historical replay or a live feed without modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from types import MappingProxyType
from typing import ClassVar, Mapping, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
IntArray: TypeAlias = npt.NDArray[np.int64]


class EventType(Enum):
    MARKET = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()
    FINANCING = auto()


class Direction(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


class Side(IntEnum):
    BUY = 1
    SELL = -1


class OrderType(Enum):
    MARKET = auto()


@dataclass(frozen=True, slots=True)
class Universe:
    """Fixed symbol and field ordering shared by every component.

    Indices into ``MarketEvent.values`` are resolved once here so that hot-path
    lookups are O(1) dictionary hits rather than pandas label searches.
    """

    symbols: tuple[str, ...]
    fields: tuple[str, ...]
    _symbol_index: Mapping[str, int] = field(init=False, repr=False, compare=False)
    _field_index: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Universe symbols must be unique")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("Universe fields must be unique")
        object.__setattr__(
            self, "_symbol_index", MappingProxyType({s: i for i, s in enumerate(self.symbols)})
        )
        object.__setattr__(
            self, "_field_index", MappingProxyType({f: i for i, f in enumerate(self.fields)})
        )

    @property
    def n_assets(self) -> int:
        return len(self.symbols)

    def symbol_index(self, symbol: str) -> int:
        try:
            return self._symbol_index[symbol]
        except KeyError as exc:
            raise KeyError(f"Unknown symbol {symbol!r}") from exc

    def field_index(self, name: str) -> int:
        try:
            return self._field_index[name]
        except KeyError as exc:
            raise KeyError(f"Unknown field {name!r}; available: {self.fields}") from exc

    def has_field(self, name: str) -> bool:
        return name in self._field_index


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """A cross-sectional snapshot of one bar for every asset in the universe.

    ``values`` is a read-only ``(n_assets, n_fields)`` view into the
    pre-computed feature cube, so building an event is zero-copy and consumers
    can operate on whole columns (``event.field("close")``) with NumPy.
    """

    type: ClassVar[EventType] = EventType.MARKET
    timestamp: pd.Timestamp
    universe: Universe
    values: FloatArray
    available: BoolArray

    def field(self, name: str) -> FloatArray:
        return self.values[:, self.universe.field_index(name)]

    def value(self, symbol: str, name: str) -> float:
        return float(
            self.values[self.universe.symbol_index(symbol), self.universe.field_index(name)]
        )

    @property
    def open(self) -> FloatArray:
        return self.field("open")

    @property
    def close(self) -> FloatArray:
        return self.field("close")


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """A strategy's directional view on one asset, with conviction in [0, 1]."""

    type: ClassVar[EventType] = EventType.SIGNAL
    timestamp: pd.Timestamp
    symbol: str
    direction: Direction
    strength: float = 1.0
    strategy_id: str = "default"

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Signal strength must be in [0, 1], got {self.strength}")

    @property
    def signed_strength(self) -> float:
        return float(self.direction) * self.strength


@dataclass(frozen=True, slots=True)
class OrderEvent:
    type: ClassVar[EventType] = EventType.ORDER
    timestamp: pd.Timestamp
    symbol: str
    side: Side
    quantity: float
    order_id: int
    order_type: OrderType = OrderType.MARKET
    reason: str = "rebalance"

    def __post_init__(self) -> None:
        if not self.quantity > 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")

    @property
    def signed_quantity(self) -> float:
        return float(self.side) * self.quantity


@dataclass(frozen=True, slots=True)
class FillEvent:
    """Broker confirmation. ``fill_price`` already includes slippage."""

    type: ClassVar[EventType] = EventType.FILL
    timestamp: pd.Timestamp
    symbol: str
    side: Side
    quantity: float
    fill_price: float
    commission: float
    slippage_cost: float
    order_id: int
    venue: str = "SIM"

    @property
    def signed_quantity(self) -> float:
        return float(self.side) * self.quantity

    @property
    def notional(self) -> float:
        return self.quantity * self.fill_price


@dataclass(frozen=True, slots=True)
class FinancingEvent:
    """A cash charge posted by the broker, e.g. stock-borrow fees on shorts."""

    type: ClassVar[EventType] = EventType.FINANCING
    timestamp: pd.Timestamp
    symbol: str
    amount: float
    kind: str = "borrow_fee"


Event: TypeAlias = MarketEvent | SignalEvent | OrderEvent | FillEvent | FinancingEvent
