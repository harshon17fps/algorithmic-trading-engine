"""Pure position arithmetic and round-trip trade bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qengine.events import Direction

QTY_EPS = 1e-9


def apply_fill(
    quantity: float, avg_price: float, fill_qty: float, fill_price: float
) -> tuple[float, float, float]:
    """Apply a signed fill to a position using average-cost accounting.

    Returns ``(new_quantity, new_avg_price, realized_pnl)``. Realized P&L is
    gross of costs and only arises on the portion of the fill that reduces
    the existing position; a fill that crosses zero closes the old position
    in full and opens the remainder at ``fill_price``.
    """
    new_qty = quantity + fill_qty
    if abs(new_qty) < QTY_EPS:
        new_qty = 0.0

    if quantity == 0.0 or np.sign(quantity) == np.sign(fill_qty):
        new_avg = (avg_price * abs(quantity) + fill_price * abs(fill_qty)) / abs(new_qty)
        return new_qty, new_avg, 0.0

    closed = min(abs(fill_qty), abs(quantity))
    realized = closed * (fill_price - avg_price) * np.sign(quantity)
    if new_qty == 0.0:
        new_avg = 0.0
    elif np.sign(new_qty) != np.sign(quantity):
        new_avg = fill_price
    else:
        new_avg = avg_price
    return new_qty, new_avg, float(realized)


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """One round trip: from flat (or a flip) until flat (or the next flip)."""

    symbol: str
    direction: Direction
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp | None
    entry_price: float
    exit_price: float
    max_quantity: float
    gross_pnl: float
    costs: float
    is_open: bool = False

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs

    @property
    def holding_period(self) -> pd.Timedelta | None:
        return None if self.exit_time is None else self.exit_time - self.entry_time


@dataclass(slots=True)
class OpenTrade:
    symbol: str
    direction: Direction
    entry_time: pd.Timestamp
    entry_notional: float = 0.0
    entry_qty: float = 0.0
    exit_notional: float = 0.0
    exit_qty: float = 0.0
    max_quantity: float = 0.0
    gross_pnl: float = 0.0
    costs: float = 0.0

    def add_entry(self, qty: float, price: float, cost: float, position_size: float) -> None:
        self.entry_notional += qty * price
        self.entry_qty += qty
        self.costs += cost
        self.max_quantity = max(self.max_quantity, position_size)

    def add_exit(self, qty: float, price: float, cost: float, realized: float) -> None:
        self.exit_notional += qty * price
        self.exit_qty += qty
        self.costs += cost
        self.gross_pnl += realized

    def to_record(self, exit_time: pd.Timestamp | None, *, mark_price: float = np.nan) -> TradeRecord:
        is_open = exit_time is None
        if is_open:
            remaining = self.entry_qty - self.exit_qty
            avg_entry = self.entry_notional / self.entry_qty if self.entry_qty else np.nan
            unrealized = remaining * (mark_price - avg_entry) * float(self.direction)
            gross = self.gross_pnl + (unrealized if np.isfinite(unrealized) else 0.0)
            exit_price = mark_price
        else:
            gross = self.gross_pnl
            exit_price = self.exit_notional / self.exit_qty if self.exit_qty else np.nan
        return TradeRecord(
            symbol=self.symbol,
            direction=self.direction,
            entry_time=self.entry_time,
            exit_time=exit_time,
            entry_price=self.entry_notional / self.entry_qty if self.entry_qty else np.nan,
            exit_price=exit_price,
            max_quantity=self.max_quantity,
            gross_pnl=gross,
            costs=self.costs,
            is_open=is_open,
        )
