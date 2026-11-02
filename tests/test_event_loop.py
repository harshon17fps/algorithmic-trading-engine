from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qengine import Direction, EventLoop, EventType, MarketEvent, SignalEvent
from qengine.data import FeatureConfig, HistoricalDataHandler, SyntheticMarketConfig, generate_synthetic_market
from qengine.execution import BpsCommission, FixedBpsSlippage, SimulatedBroker
from qengine.portfolio import FixedFractionSizer, Portfolio, RiskLimits, RiskParitySizer
from qengine.strategy import Strategy, TrendSentimentStrategy

SMALL_FEATURES = FeatureConfig(ema_fast=2, ema_slow=3, vol_window=2, mu_halflife=2, rsi_window=2, atr_window=2, adv_window=2)


class BuyOnceStrategy(Strategy):
    """Goes long the first symbol on the first bar and records what it saw."""

    def __init__(self, universe) -> None:  # type: ignore[no-untyped-def]
        super().__init__(universe)
        self.seen: list[pd.Timestamp] = []

    def on_market(self, event: MarketEvent) -> list[SignalEvent]:
        self.seen.append(event.timestamp)
        if len(self.seen) == 1:
            return [SignalEvent(event.timestamp, self.universe.symbols[0], Direction.LONG, 1.0)]
        return []


def toy_handler() -> HistoricalDataHandler:
    idx = pd.bdate_range("2024-01-01", periods=5)
    bars = {
        "AAA": pd.DataFrame(
            {"open": [10, 11, 12, 13, 14], "high": [11, 12, 13, 14, 15], "low": [9, 10, 11, 12, 13],
             "close": [10.5, 11.5, 12.5, 13.5, 14.5], "volume": [1e6] * 5},
            index=idx, dtype=float,
        ),
        "BBB": pd.DataFrame(
            {"open": [20] * 5, "high": [21] * 5, "low": [19] * 5, "close": [20] * 5, "volume": [1e6] * 5},
            index=idx, dtype=float,
        ),
    }
    return HistoricalDataHandler(bars, features=SMALL_FEATURES)


def test_signal_on_bar_t_fills_at_open_of_bar_t_plus_1() -> None:
    data = toy_handler()
    u = data.universe
    strategy = BuyOnceStrategy(u)
    # A wide band isolates the entry; otherwise target-weight rebalancing trims as price rises.
    portfolio = Portfolio(u, 10_000.0, FixedFractionSizer(0.5), RiskLimits(max_position_weight=1.0, rebalance_band=0.3))
    broker = SimulatedBroker(u, commission=BpsCommission(0.0), slippage=FixedBpsSlippage(0.0))
    result = EventLoop(data, strategy, portfolio, broker).run()

    fills = result.report.fills
    assert len(fills) == 1
    assert fills.loc[0, "timestamp"] == data.index[1]
    assert fills.loc[0, "fill_price"] == 11.0          # bar-1 open, not bar-0 close
    assert fills.loc[0, "quantity"] == 476             # trunc(0.5 * 10,000 / 10.5)

    curve = result.report.equity_curve["equity"]
    assert curve.iloc[0] == 10_000.0                   # nothing traded yet on bar 0
    assert curve.iloc[1] == pytest.approx(10_000 + 476 * (11.5 - 11.0))  # filled at open, marked at close
    assert strategy.seen == list(data.index)
    assert result.event_counts[EventType.MARKET] == 5


@pytest.fixture(scope="module")
def synthetic_result():  # type: ignore[no-untyped-def]
    ds = generate_synthetic_market(SyntheticMarketConfig(n_bars=500, seed=11))
    data = HistoricalDataHandler(ds.bars, sentiment=ds.sentiment, benchmark=ds.benchmark, risk_free=ds.risk_free)
    u = data.universe
    portfolio = Portfolio(u, 1_000_000.0, RiskParitySizer(u.n_assets, target_vol=0.12), RiskLimits())
    broker = SimulatedBroker(u, borrow_rate=0.02)
    loop = EventLoop(data, TrendSentimentStrategy(u), portfolio, broker)
    return data, broker, loop.run()


def test_end_to_end_runs_on_multi_asset_universe(synthetic_result) -> None:  # type: ignore[no-untyped-def]
    data, broker, result = synthetic_result
    assert data.universe.n_assets >= 5
    assert len(result.report.equity_curve) == len(data)
    counts = result.event_counts
    # Orders submitted on the final bar have no next open to fill against.
    assert counts[EventType.ORDER] == counts[EventType.FILL] + len(broker.resting_orders)
    assert counts[EventType.FILL] > 0
    assert counts[EventType.FINANCING] > 0  # shorts were held and charged borrow


def test_report_contains_required_analytics(synthetic_result) -> None:  # type: ignore[no-untyped-def]
    _, _, result = synthetic_result
    flat = result.report.flat()
    for key in ("sharpe_ratio", "sortino_ratio", "max_drawdown", "max_drawdown_duration_bars",
                "max_drawdown_duration_days", "profit_factor", "win_loss_ratio", "beta"):
        value = flat[key]
        assert isinstance(value, (int, float)) and math.isfinite(value), key


def test_books_reconcile_with_broker_and_identity(synthetic_result) -> None:  # type: ignore[no-untyped-def]
    _, broker, result = synthetic_result
    pf = result.portfolio
    np.testing.assert_allclose(pf.quantities, broker.ledger)
    resting = np.array(
        [sum(o.signed_quantity for o in broker.resting_orders if o.symbol == s) for s in pf.universe.symbols]
    )
    np.testing.assert_allclose(pf.pending, resting)
    expected = pf.initial_capital + pf.realized_pnl + pf.unrealized_pnl - pf.total_commission - pf.total_financing
    assert pf.equity == pytest.approx(expected, rel=1e-10)
    assert result.report.equity_curve["equity"].iloc[-1] == pytest.approx(pf.equity)


def test_report_export(synthetic_result, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    _, _, result = synthetic_result
    paths = result.report.export(tmp_path)
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    assert "Sharpe" in paths["markdown"].read_text().title()
