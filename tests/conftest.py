from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

from qengine.events import FillEvent, MarketEvent, Side, Universe

SYMBOLS: tuple[str, ...] = ("AAA", "BBB", "CCC")
FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "ret", "vol", "risk_free", "sentiment", "trend")
T0 = pd.Timestamp("2024-01-02")


@pytest.fixture
def universe() -> Universe:
    return Universe(symbols=SYMBOLS, fields=FIELDS)


def make_market(
    universe: Universe,
    closes: Sequence[float],
    *,
    ts: pd.Timestamp = T0,
    opens: Sequence[float] | None = None,
    extra: Mapping[str, Sequence[float]] | None = None,
) -> MarketEvent:
    n, f = universe.n_assets, len(universe.fields)
    values = np.full((n, f), np.nan)
    close = np.asarray(closes, dtype=np.float64)
    values[:, universe.field_index("close")] = close
    values[:, universe.field_index("open")] = close if opens is None else np.asarray(opens, dtype=np.float64)
    values[:, universe.field_index("volume")] = 1e6
    for name, col in (extra or {}).items():
        values[:, universe.field_index(name)] = np.asarray(col, dtype=np.float64)
    values.setflags(write=False)
    return MarketEvent(timestamp=ts, universe=universe, values=values, available=np.isfinite(close))


def make_fill(
    symbol: str,
    side: Side,
    quantity: float,
    price: float,
    *,
    commission: float = 0.0,
    ts: pd.Timestamp = T0,
    order_id: int = 0,
) -> FillEvent:
    return FillEvent(
        timestamp=ts,
        symbol=symbol,
        side=side,
        quantity=quantity,
        fill_price=price,
        commission=commission,
        slippage_cost=0.0,
        order_id=order_id,
    )
