"""Vectorized performance statistics. Pure functions over pandas/NumPy inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


def to_period_rate(annual_rate: pd.Series | float, periods_per_year: int) -> pd.Series | float:
    """Convert an annualized (compounded) rate to a per-period rate."""
    return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0


def simple_returns(equity: pd.Series) -> pd.Series:
    return (equity / equity.shift(1) - 1.0).iloc[1:]


def _excess(returns: pd.Series, rf_period: pd.Series | float) -> pd.Series:
    if isinstance(rf_period, pd.Series):
        rf = rf_period.reindex(returns.index).ffill().fillna(0.0)
        return returns - rf
    return returns - rf_period


def annualized_return(equity: pd.Series, periods_per_year: int) -> float:
    n = len(equity) - 1
    if n <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    growth = equity.iloc[-1] / equity.iloc[0]
    return float(growth ** (periods_per_year / n) - 1.0) if growth > 0 else -1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, rf_period: pd.Series | float, periods_per_year: int) -> float:
    """Annualized Sharpe on per-period excess returns over a (possibly time-varying) risk-free rate."""
    excess = _excess(returns, rf_period)
    sd = excess.std(ddof=1)
    if len(excess) < 2 or not sd > 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, rf_period: pd.Series | float, periods_per_year: int) -> float:
    """Annualized Sortino; downside deviation is the root-mean-square of negative
    excess returns taken over *all* periods (target = risk-free)."""
    excess = _excess(returns, rf_period)
    if len(excess) < 2:
        return float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2)))
    mean = float(excess.mean())
    if downside == 0.0:
        return float("inf") if mean > 0 else float("nan")
    return mean / downside * float(np.sqrt(periods_per_year))


@dataclass(frozen=True, slots=True)
class DrawdownStats:
    max_drawdown: float
    peak: pd.Timestamp
    trough: pd.Timestamp
    recovery: pd.Timestamp | None
    max_duration_bars: int
    max_duration_days: float
    longest_start: pd.Timestamp
    longest_end: pd.Timestamp
    current_drawdown: float


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def drawdown_stats(equity: pd.Series) -> DrawdownStats:
    """Depth and duration of drawdowns, fully vectorized.

    Duration is measured peak-to-recovery (or peak-to-end when the equity has
    not yet regained its high-water mark), in both bars and calendar days.
    """
    v = equity.to_numpy(dtype=np.float64)
    idx = pd.DatetimeIndex(equity.index)
    t = np.arange(v.size)
    hwm = np.maximum.accumulate(v)
    dd = v / hwm - 1.0

    trough = int(np.argmin(dd))
    peak = int(np.flatnonzero(v[: trough + 1] >= hwm[trough])[-1])
    after = np.flatnonzero(v[trough:] >= hwm[trough])
    recovery = int(trough + after[0]) if dd[trough] < 0 and after.size else None

    at_peak = v >= hwm
    last_peak = np.maximum.accumulate(np.where(at_peak, t, 0))
    prev_last_peak = np.concatenate(([0], last_peak[:-1]))
    was_underwater = np.concatenate(([False], ~at_peak[:-1]))
    start = np.where(~at_peak, last_peak, np.where(was_underwater, prev_last_peak, t))
    dur_bars = t - start
    dur_days = np.asarray((idx - idx[start]) / pd.Timedelta(days=1), dtype=np.float64)
    longest = int(np.argmax(dur_bars))

    return DrawdownStats(
        max_drawdown=float(dd[trough]),
        peak=idx[peak],
        trough=idx[trough],
        recovery=idx[recovery] if recovery is not None else None,
        max_duration_bars=int(dur_bars[longest]),
        max_duration_days=float(np.max(dur_days)),
        longest_start=idx[int(start[longest])],
        longest_end=idx[longest],
        current_drawdown=float(dd[-1]),
    )


@dataclass(frozen=True, slots=True)
class TradeStats:
    n_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    expectancy: float
    largest_win: float
    largest_loss: float


def trade_stats(pnls: npt.ArrayLike) -> TradeStats:
    """Profit factor = gross profit / gross loss; win/loss ratio = avg win / avg loss."""
    p = np.asarray(pnls, dtype=np.float64)
    n = int(p.size)
    if n == 0:
        nan = float("nan")
        return TradeStats(0, nan, nan, nan, nan, nan, nan, nan, nan)
    wins, losses = p[p > 0], p[p < 0]
    gross_profit, gross_loss = float(wins.sum()), float(-losses.sum())
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(-losses.mean()) if losses.size else 0.0

    def inf_or_nan(x: float) -> float:
        return float("inf") if x > 0 else float("nan")

    return TradeStats(
        n_trades=n,
        win_rate=wins.size / n,
        profit_factor=gross_profit / gross_loss if gross_loss > 0 else inf_or_nan(gross_profit),
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=avg_win / avg_loss if avg_loss > 0 else inf_or_nan(avg_win),
        expectancy=float(p.mean()),
        largest_win=float(np.max(p)),
        largest_loss=float(np.min(p)),
    )


@dataclass(frozen=True, slots=True)
class BenchmarkStats:
    beta: float
    alpha: float
    correlation: float
    tracking_error: float
    information_ratio: float


def benchmark_stats(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_period: pd.Series | float,
    periods_per_year: int,
) -> BenchmarkStats:
    """CAPM beta/alpha on excess returns, plus active-risk statistics."""
    joined = pd.concat({"p": returns, "b": benchmark_returns}, axis=1).dropna()
    nan = float("nan")
    if len(joined) < 3:
        return BenchmarkStats(nan, nan, nan, nan, nan)
    ep = _excess(joined["p"], rf_period).to_numpy()
    eb = _excess(joined["b"], rf_period).to_numpy()
    cov = np.cov(ep, eb, ddof=1)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else nan
    alpha = (ep.mean() - beta * eb.mean()) * periods_per_year
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) if cov[0, 0] > 0 and cov[1, 1] > 0 else nan
    active = (joined["p"] - joined["b"]).to_numpy()
    te = float(active.std(ddof=1) * np.sqrt(periods_per_year))
    ir = float(active.mean() * periods_per_year / te) if te > 0 else nan
    return BenchmarkStats(float(beta), float(alpha), float(corr), te, ir)
