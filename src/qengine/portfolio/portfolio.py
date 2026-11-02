"""Portfolio: the system's book of record for cash, positions and risk.

Responsibilities, in event order within one bar:

* ``on_fill`` / ``on_financing`` — apply broker confirmations to cash and
  positions (average-cost accounting) and maintain round-trip trade records.
* ``on_market`` — mark positions to the bar's close and feed the sizer's
  return estimators.
* ``on_signal`` — record the strategy's latest view in the *intent book*.
  Signals never trade directly; this lets multi-asset sizers (risk parity,
  Kelly) allocate across the whole book instead of one symbol at a time.
* ``generate_orders`` — size the intent book, project it onto the risk limits,
  diff against current + in-flight positions, and emit OrderEvents.

State is held in NumPy arrays indexed by the universe so mark-to-market,
exposure and margin are single vectorized expressions.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from qengine.events import (
    Direction,
    FillEvent,
    FinancingEvent,
    FloatArray,
    MarketEvent,
    OrderEvent,
    Side,
    SignalEvent,
    Universe,
)
from qengine.portfolio.accounting import QTY_EPS, OpenTrade, TradeRecord, apply_fill
from qengine.portfolio.risk import RiskLimits
from qengine.portfolio.sizing import PositionSizer


@dataclass(frozen=True, slots=True)
class PositionView:
    symbol: str
    quantity: float
    avg_price: float
    last_price: float
    realized_pnl: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price if self.quantity else 0.0

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.last_price - self.avg_price) if self.quantity else 0.0


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    timestamp: pd.Timestamp
    equity: float
    cash: float
    gross_exposure: float
    net_exposure: float
    margin_requirement: float
    n_positions: int

    @property
    def gross_leverage(self) -> float:
        return self.gross_exposure / self.equity if self.equity > 0 else np.inf


class Portfolio:
    def __init__(
        self,
        universe: Universe,
        initial_capital: float,
        sizer: PositionSizer,
        limits: RiskLimits = RiskLimits(),
        *,
        lot_size: float = 1.0,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if lot_size <= 0:
            raise ValueError("lot_size must be positive")
        n = universe.n_assets
        self.universe = universe
        self.initial_capital = float(initial_capital)
        self.sizer = sizer
        self.limits = limits
        self.lot_size = lot_size

        self.cash = float(initial_capital)
        self._qty: FloatArray = np.zeros(n)
        self._avg: FloatArray = np.zeros(n)
        self._realized: FloatArray = np.zeros(n)
        self._last: FloatArray = np.full(n, np.nan)
        self._pending: FloatArray = np.zeros(n)
        self._intent: FloatArray = np.zeros(n)
        self._open_orders: dict[int, float] = {}  # order_id -> signed unfilled quantity

        self._open_trades: dict[str, OpenTrade] = {}
        self.closed_trades: list[TradeRecord] = []
        self._order_ids: Iterator[int] = itertools.count(1)
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_financing = 0.0
        self.halted = False

    # ------------------------------------------------------------------ views
    @property
    def quantities(self) -> FloatArray:
        return self._qty.copy()

    @property
    def pending(self) -> FloatArray:
        return self._pending.copy()

    @property
    def intent(self) -> FloatArray:
        return self._intent.copy()

    @property
    def market_values(self) -> FloatArray:
        return np.where(self._qty != 0.0, self._qty * np.nan_to_num(self._last), 0.0)

    @property
    def equity(self) -> float:
        return self.cash + float(self.market_values.sum())

    @property
    def gross_exposure(self) -> float:
        return float(np.abs(self.market_values).sum())

    @property
    def net_exposure(self) -> float:
        return float(self.market_values.sum())

    @property
    def weights(self) -> FloatArray:
        eq = self.equity
        return self.market_values / eq if eq > 0 else np.zeros_like(self._qty)

    def margin_requirement(self, *, maintenance: bool = False) -> float:
        return self.limits.margin.requirement(self.market_values, maintenance=maintenance)

    @property
    def excess_liquidity(self) -> float:
        """Equity above the initial requirement — i.e. capacity to add risk."""
        return self.equity - self.margin_requirement()

    @property
    def in_margin_call(self) -> bool:
        return self.equity < self.margin_requirement(maintenance=True)

    @property
    def unrealized_pnl(self) -> float:
        return float(np.sum(np.where(self._qty != 0.0, self._qty * (np.nan_to_num(self._last) - self._avg), 0.0)))

    @property
    def realized_pnl(self) -> float:
        return float(self._realized.sum())

    def position(self, symbol: str) -> PositionView:
        i = self.universe.symbol_index(symbol)
        return PositionView(
            symbol=symbol,
            quantity=float(self._qty[i]),
            avg_price=float(self._avg[i]),
            last_price=float(self._last[i]),
            realized_pnl=float(self._realized[i]),
        )

    def snapshot(self, timestamp: pd.Timestamp) -> PortfolioSnapshot:
        mv = self.market_values
        return PortfolioSnapshot(
            timestamp=timestamp,
            equity=self.cash + float(mv.sum()),
            cash=self.cash,
            gross_exposure=float(np.abs(mv).sum()),
            net_exposure=float(mv.sum()),
            margin_requirement=self.limits.margin.requirement(mv),
            n_positions=int(np.count_nonzero(self._qty)),
        )

    def trades(self) -> list[TradeRecord]:
        """Closed round trips followed by still-open trades marked to market."""
        open_records = [
            t.to_record(None, mark_price=float(self._last[self.universe.symbol_index(s)]))
            for s, t in self._open_trades.items()
        ]
        return [*self.closed_trades, *open_records]

    # --------------------------------------------------------------- handlers
    def on_market(self, event: MarketEvent) -> None:
        close = event.close
        self._last = np.where(event.available & np.isfinite(close), close, self._last)
        self.sizer.update(event)

    def on_signal(self, signal: SignalEvent) -> list[OrderEvent]:
        i = self.universe.symbol_index(signal.symbol)
        view = signal.signed_strength
        if view < 0 and not self.limits.allow_short:
            view = 0.0
        self._intent[i] = view
        return []

    def on_fill(self, fill: FillEvent) -> None:
        i = self.universe.symbol_index(fill.symbol)
        q0 = float(self._qty[i])
        dq = fill.signed_quantity
        q1, avg1, realized = apply_fill(q0, float(self._avg[i]), dq, fill.fill_price)

        self.cash -= dq * fill.fill_price + fill.commission
        # Only fills against our own working orders reduce "in flight" quantity;
        # external fills (manual trades, assignments) change the position only.
        remaining = self._open_orders.pop(fill.order_id, None)
        if remaining is not None:
            self._pending[i] -= dq
            if abs(self._pending[i]) < QTY_EPS:
                self._pending[i] = 0.0
            if abs(remaining - dq) > QTY_EPS:
                self._open_orders[fill.order_id] = remaining - dq
        self.total_commission += fill.commission
        self.total_slippage += fill.slippage_cost

        reducing = q0 != 0.0 and np.sign(dq) != np.sign(q0)
        closing_qty = min(abs(dq), abs(q0)) if reducing else 0.0
        opening_qty = abs(dq) - closing_qty
        close_cost = fill.commission * closing_qty / abs(dq)

        if closing_qty > 0.0:
            trade = self._open_trades[fill.symbol]
            trade.add_exit(closing_qty, fill.fill_price, close_cost, realized)
            if q1 == 0.0 or np.sign(q1) != np.sign(q0):
                self.closed_trades.append(trade.to_record(fill.timestamp))
                del self._open_trades[fill.symbol]
        if opening_qty > QTY_EPS:
            trade = self._open_trades.get(fill.symbol) or OpenTrade(
                symbol=fill.symbol, direction=Direction(int(np.sign(q1))), entry_time=fill.timestamp
            )
            trade.add_entry(opening_qty, fill.fill_price, fill.commission - close_cost, abs(q1))
            self._open_trades[fill.symbol] = trade

        self._qty[i] = q1
        self._avg[i] = avg1
        self._realized[i] += realized
        if not np.isfinite(self._last[i]):
            self._last[i] = fill.fill_price

    def on_financing(self, event: FinancingEvent) -> None:
        self.cash -= event.amount
        self.total_financing += event.amount
        trade = self._open_trades.get(event.symbol)
        if trade is not None:
            trade.costs += event.amount

    # ----------------------------------------------------------------- orders
    def generate_orders(self, timestamp: pd.Timestamp) -> list[OrderEvent]:
        """Convert the intent book into orders, subject to risk and margin."""
        if self.halted:
            return []
        equity = self.equity
        if equity <= 0.0:
            self.halted = True
            self._intent[:] = 0.0
            return self._orders_to_target(timestamp, np.zeros_like(self._qty), "liquidation", use_band=False)

        if self.in_margin_call:
            # Deleverage proportionally until back inside *initial* margin with buffer.
            capacity = equity * (1.0 - self.limits.margin_buffer)
            scale = min(1.0, capacity / self.margin_requirement())
            target = self._round_lots(self._qty * scale)
            return self._orders_to_target(timestamp, target, "margin_call", use_band=False)

        price = self._last
        tradable = np.isfinite(price) & (price > 0.0)
        raw = self.sizer.target_weights(self._intent, tradable)
        weights = self.limits.apply(np.where(tradable, raw, 0.0))
        safe_price = np.where(tradable, price, 1.0)
        target = np.where(tradable, self._round_lots(weights * equity / safe_price), self._qty + self._pending)
        return self._orders_to_target(timestamp, target, "rebalance", use_band=True)

    def _round_lots(self, qty: FloatArray) -> FloatArray:
        # Truncate toward zero so rounding can only shrink risk, never breach limits.
        return np.trunc(qty / self.lot_size + 1e-9 * np.sign(qty)) * self.lot_size

    def _orders_to_target(
        self, timestamp: pd.Timestamp, target: FloatArray, reason: str, *, use_band: bool
    ) -> list[OrderEvent]:
        effective = self._qty + self._pending
        delta = target - effective
        trade = np.abs(delta) > QTY_EPS
        if use_band:
            notional = np.abs(delta) * np.nan_to_num(self._last)
            exiting = (target == 0.0) & (effective != 0.0)
            flipping = np.sign(target) * np.sign(effective) < 0
            trade &= (notional >= self.limits.rebalance_band * self.equity) | exiting | flipping

        idx = np.flatnonzero(trade)
        # Risk-reducing orders first so a live broker frees margin before we add risk.
        reduces = np.abs(target[idx]) < np.abs(effective[idx])
        idx = idx[np.argsort(~reduces, kind="stable")]

        orders: list[OrderEvent] = []
        for i in idx:
            d = float(delta[i])
            order = OrderEvent(
                timestamp=timestamp,
                symbol=self.universe.symbols[i],
                side=Side.BUY if d > 0 else Side.SELL,
                quantity=abs(d),
                order_id=next(self._order_ids),
                reason=reason,
            )
            orders.append(order)
            self._open_orders[order.order_id] = d
            self._pending[i] += d
        return orders
