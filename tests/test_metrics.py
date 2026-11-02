from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qengine.performance import (
    benchmark_stats,
    drawdown_stats,
    sharpe_ratio,
    sortino_ratio,
    to_period_rate,
    trade_stats,
)

IDX = pd.bdate_range("2024-01-01", periods=6)


def test_period_rate_compounds_back_to_annual() -> None:
    r = to_period_rate(0.05, 252)
    assert (1 + r) ** 252 == pytest.approx(1.05)


def test_sharpe_matches_definition_with_dynamic_rf() -> None:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=500)
    rets = pd.Series(rng.normal(0.0008, 0.01, 500), index=idx)
    rf = pd.Series(np.linspace(0.0, 0.0002, 500), index=idx)
    ex = rets - rf
    assert sharpe_ratio(rets, rf, 252) == pytest.approx(ex.mean() / ex.std(ddof=1) * np.sqrt(252))
    # A rising rf must lower Sharpe relative to rf = 0.
    assert sharpe_ratio(rets, rf, 252) < sharpe_ratio(rets, 0.0, 252)


def test_sortino_uses_downside_deviation() -> None:
    rets = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01], index=IDX[:5])
    downside = np.sqrt(np.mean(np.minimum(rets.to_numpy(), 0) ** 2))
    assert sortino_ratio(rets, 0.0, 252) == pytest.approx(rets.mean() / downside * np.sqrt(252))


def test_drawdown_depth_and_duration() -> None:
    equity = pd.Series([100, 110, 99, 105, 111, 108], index=IDX, dtype=float)
    dd = drawdown_stats(equity)
    assert dd.max_drawdown == pytest.approx(99 / 110 - 1)
    assert (dd.peak, dd.trough, dd.recovery) == (IDX[1], IDX[2], IDX[4])
    assert dd.max_duration_bars == 3
    assert dd.max_duration_days == (IDX[4] - IDX[1]).days
    assert dd.current_drawdown == pytest.approx(108 / 111 - 1)


def test_unrecovered_drawdown_runs_to_end() -> None:
    equity = pd.Series([100, 120, 90, 95, 100, 110], index=IDX, dtype=float)
    dd = drawdown_stats(equity)
    assert dd.recovery is None
    assert dd.max_duration_bars == 4
    assert (dd.longest_start, dd.longest_end) == (IDX[1], IDX[5])


def test_monotonic_equity_has_no_drawdown() -> None:
    dd = drawdown_stats(pd.Series([1.0, 2, 3, 4, 5, 6], index=IDX))
    assert dd.max_drawdown == 0.0 and dd.max_duration_bars == 0


def test_trade_stats() -> None:
    ts = trade_stats([300.0, -100.0, 200.0, -200.0, 100.0])
    assert ts.n_trades == 5
    assert ts.win_rate == pytest.approx(0.6)
    assert ts.profit_factor == pytest.approx(600 / 300)
    assert ts.avg_win == pytest.approx(200.0)
    assert ts.avg_loss == pytest.approx(150.0)
    assert ts.win_loss_ratio == pytest.approx(200 / 150)
    assert ts.expectancy == pytest.approx(60.0)


def test_trade_stats_without_losses() -> None:
    assert trade_stats([10.0, 20.0]).profit_factor == float("inf")
    assert np.isnan(trade_stats([]).profit_factor)


def test_beta_and_alpha_of_levered_benchmark() -> None:
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2024-01-01", periods=750)
    bench = pd.Series(rng.normal(0.0004, 0.01, 750), index=idx)
    bs = benchmark_stats(2.0 * bench, bench, 0.0, 252)
    assert bs.beta == pytest.approx(2.0)
    assert bs.alpha == pytest.approx(0.0, abs=1e-12)
    assert bs.correlation == pytest.approx(1.0)


def test_beta_of_uncorrelated_strategy_is_near_zero() -> None:
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2024-01-01", periods=5_000)
    bs = benchmark_stats(
        pd.Series(rng.normal(0, 0.01, 5_000), index=idx), pd.Series(rng.normal(0, 0.01, 5_000), index=idx), 0.0, 252
    )
    assert abs(bs.beta) < 0.05
