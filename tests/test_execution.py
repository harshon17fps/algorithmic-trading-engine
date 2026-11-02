from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import T0, make_market
from qengine.events import FillEvent, FinancingEvent, OrderEvent, Side, Universe
from qengine.execution import BpsCommission, FixedBpsSlippage, PerShareCommission, SimulatedBroker, SquareRootImpact


def order(symbol: str, side: Side, qty: float, oid: int = 1) -> OrderEvent:
    return OrderEvent(timestamp=T0, symbol=symbol, side=side, quantity=qty, order_id=oid)


def broker(universe: Universe, **kw: object) -> SimulatedBroker:
    defaults: dict[str, object] = {"commission": BpsCommission(0.0), "slippage": FixedBpsSlippage(10.0)}
    defaults.update(kw)
    return SimulatedBroker(universe, **defaults)  # type: ignore[arg-type]


def test_fills_at_next_open_with_adverse_slippage(universe: Universe) -> None:
    b = broker(universe)
    b.on_market(make_market(universe, [50.0, 20.0, 10.0]))
    assert b.on_order(order("AAA", Side.BUY, 100, 1)) == []
    assert b.on_order(order("BBB", Side.SELL, 100, 2)) == []
    events = b.on_market(make_market(universe, [52.0, 21.0, 10.0], opens=[51.0, 19.0, 10.0], ts=T0 + pd.Timedelta(days=1)))
    fills = {f.symbol: f for f in events if isinstance(f, FillEvent)}
    assert fills["AAA"].fill_price == pytest.approx(51.0 * 1.001)
    assert fills["BBB"].fill_price == pytest.approx(19.0 * 0.999)
    assert fills["AAA"].slippage_cost == pytest.approx(100 * 51.0 * 0.001)
    assert fills["AAA"].timestamp == T0 + pd.Timedelta(days=1)
    assert b.resting_orders == ()


def test_order_rests_until_symbol_trades(universe: Universe) -> None:
    b = broker(universe)
    b.on_order(order("AAA", Side.BUY, 10))
    assert b.on_market(make_market(universe, [np.nan, 20.0, 10.0])) == []
    assert len(b.resting_orders) == 1
    fills = b.on_market(make_market(universe, [50.0, 20.0, 10.0]))
    assert len(fills) == 1 and isinstance(fills[0], FillEvent)


def test_borrow_fee_accrues_actual_360_on_prior_close(universe: Universe) -> None:
    b = broker(universe, slippage=FixedBpsSlippage(0.0), borrow_rate={"BBB": 0.03})
    friday = pd.Timestamp("2024-01-05")
    b.on_order(order("BBB", Side.SELL, 100))
    b.on_market(make_market(universe, [50.0, 40.0, 10.0], ts=friday))  # short filled Friday
    events = b.on_market(make_market(universe, [50.0, 41.0, 10.0], ts=friday + pd.Timedelta(days=3)))
    (fee,) = [e for e in events if isinstance(e, FinancingEvent)]
    # 100 shares * $40 Friday close * 3% * 3 days / 360
    assert fee.symbol == "BBB"
    assert fee.amount == pytest.approx(100 * 40.0 * 0.03 * 3 / 360)


def test_long_positions_pay_no_borrow(universe: Universe) -> None:
    b = broker(universe)
    b.on_order(order("AAA", Side.BUY, 100))
    b.on_market(make_market(universe, [50.0, 40.0, 10.0]))
    events = b.on_market(make_market(universe, [50.0, 40.0, 10.0], ts=T0 + pd.Timedelta(days=1)))
    assert not any(isinstance(e, FinancingEvent) for e in events)


def test_per_share_commission_min_and_cap() -> None:
    c = PerShareCommission(per_share=0.005, minimum=1.0, max_pct_of_notional=0.01)
    assert c(100, 50.0) == pytest.approx(1.0)        # 0.50 -> minimum 1.00
    assert c(10_000, 50.0) == pytest.approx(50.0)    # 10,000 * 0.005
    assert c(1_000, 0.05) == pytest.approx(0.5)      # capped at 1% of $50 notional


def test_square_root_impact() -> None:
    m = SquareRootImpact(spread_bps=4.0, impact_coef=0.5, periods_per_year=252)
    # 2bp half-spread + 0.5 * (0.32 / sqrt(252)) * sqrt(10,000 / 1,000,000)
    expected = 2e-4 + 0.5 * (0.32 / np.sqrt(252)) * 0.1
    assert m.impact(10_000, 50.0, 1_000_000, 0.32) == pytest.approx(expected)
    assert m.impact(10_000, 50.0, np.nan, 0.32) == pytest.approx(2e-4)
