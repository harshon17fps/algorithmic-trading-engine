"""ExecutionHandler: the boundary between the engine and a (simulated) broker."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from qengine.events import (
    Event,
    FillEvent,
    FinancingEvent,
    FloatArray,
    MarketEvent,
    OrderEvent,
    Universe,
)
from qengine.execution.costs import CommissionModel, PerShareCommission, SlippageModel, SquareRootImpact


class ExecutionHandler(ABC):
    @abstractmethod
    def on_order(self, order: OrderEvent) -> Sequence[Event]:
        """Accept an order. Simulated brokers queue it; live adapters route it."""

    @abstractmethod
    def on_market(self, event: MarketEvent) -> Sequence[Event]:
        """Advance broker state to a new bar; return any fills/financing posted."""


class SimulatedBroker(ExecutionHandler):
    """Fills market orders at the *next* bar's open, plus costs.

    Timeline for bar ``t`` (prevents look-ahead by construction):

    1. Overnight stock-borrow fees are posted for shorts held since bar
       ``t-1``'s close (Actual/360, the stock-loan market convention).
    2. Orders submitted after bar ``t-1`` closed are filled at bar ``t``'s
       open, adjusted adversely by the slippage model; commission is added.
       Orders for symbols with no print this bar remain resting.

    The broker keeps its own position ledger — exactly as a real prime broker
    does — so borrow fees do not depend on the client's (Portfolio's) books.
    """

    def __init__(
        self,
        universe: Universe,
        *,
        commission: CommissionModel = PerShareCommission(),
        slippage: SlippageModel = SquareRootImpact(),
        borrow_rate: float | Mapping[str, float] = 0.005,
        day_count: float = 360.0,
    ) -> None:
        self.universe = universe
        self.commission = commission
        self.slippage = slippage
        n = universe.n_assets
        if isinstance(borrow_rate, Mapping):
            self._borrow: FloatArray = np.array([borrow_rate.get(s, 0.005) for s in universe.symbols])
        else:
            self._borrow = np.full(n, float(borrow_rate))
        self.day_count = day_count
        self._resting: list[OrderEvent] = []
        self._ledger: FloatArray = np.zeros(n)
        self._last_close: FloatArray = np.full(n, np.nan)
        self._last_ts: pd.Timestamp | None = None
        self._has_vol = universe.has_field("vol")

    @property
    def resting_orders(self) -> tuple[OrderEvent, ...]:
        return tuple(self._resting)

    @property
    def ledger(self) -> FloatArray:
        """Broker-side positions built from its own fills (for reconciliation)."""
        return self._ledger.copy()

    def on_order(self, order: OrderEvent) -> list[Event]:
        self.universe.symbol_index(order.symbol)  # validate early
        self._resting.append(order)
        return []

    def on_market(self, event: MarketEvent) -> list[Event]:
        out: list[Event] = self._accrue_borrow(event.timestamp)
        out.extend(self._match(event))
        close = event.close
        self._last_close = np.where(event.available & np.isfinite(close), close, self._last_close)
        self._last_ts = event.timestamp
        return out

    def _accrue_borrow(self, now: pd.Timestamp) -> list[Event]:
        if self._last_ts is None:
            return []
        days = (now - self._last_ts) / pd.Timedelta(days=1)
        short_value = np.where(self._ledger < 0, -self._ledger * np.nan_to_num(self._last_close), 0.0)
        fees = short_value * self._borrow * days / self.day_count
        return [
            FinancingEvent(timestamp=now, symbol=self.universe.symbols[i], amount=float(fees[i]))
            for i in np.flatnonzero(fees > 0)
        ]

    def _match(self, event: MarketEvent) -> list[Event]:
        if not self._resting:
            return []
        opens = event.open
        volume = event.field("volume")
        vol = event.field("vol") if self._has_vol else np.full(self.universe.n_assets, np.nan)
        fills: list[Event] = []
        still_resting: list[OrderEvent] = []
        for order in self._resting:
            i = self.universe.symbol_index(order.symbol)
            ref = float(opens[i])
            if not (event.available[i] and np.isfinite(ref) and ref > 0):
                still_resting.append(order)
                continue
            impact = self.slippage.impact(order.quantity, ref, float(volume[i]), float(vol[i]))
            price = ref * (1.0 + float(order.side) * impact)
            fills.append(
                FillEvent(
                    timestamp=event.timestamp,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    fill_price=price,
                    commission=self.commission(order.quantity, price),
                    slippage_cost=order.quantity * abs(price - ref),
                    order_id=order.order_id,
                )
            )
            self._ledger[i] += order.signed_quantity
        self._resting = still_resting
        return fills
