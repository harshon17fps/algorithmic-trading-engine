"""Market-data adapters: CSV, yfinance, and a reproducible synthetic market.

Each loader returns ``dict[symbol, DataFrame]`` with lowercase OHLCV columns and
a sorted ``DatetimeIndex`` — the only contract :class:`HistoricalDataHandler`
relies on. Adding Alpaca/Binance is a matter of writing another such function.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from qengine.data.handler import OHLCV


# --------------------------------------------------------------------------- CSV
def load_csv_bars(
    directory: str | Path,
    symbols: Sequence[str] | None = None,
    date_column: str = "date",
) -> dict[str, pd.DataFrame]:
    """Load ``<SYMBOL>.csv`` files. An ``adj close`` column, if present, is used
    to back-adjust OHLC for splits/dividends."""
    root = Path(directory)
    files = {p.stem: p for p in sorted(root.glob("*.csv"))}
    wanted = list(symbols) if symbols is not None else list(files)
    out: dict[str, pd.DataFrame] = {}
    for sym in wanted:
        if sym not in files:
            raise FileNotFoundError(f"No CSV for {sym!r} in {root}")
        df = pd.read_csv(files[sym])
        df.columns = [str(c).strip().lower() for c in df.columns]
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column).sort_index()
        if "adj close" in df.columns:
            factor = df["adj close"] / df["close"]
            for col in ("open", "high", "low", "close"):
                df[col] = df[col] * factor
        out[sym] = df[list(OHLCV)].astype("float64")
    return out


# ---------------------------------------------------------------------- yfinance
def _require_yfinance() -> ModuleType:
    try:
        return importlib.import_module("yfinance")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("yfinance is optional: pip install 'qengine[yfinance]'") from exc


def load_yfinance_bars(
    symbols: Sequence[str], start: str, end: str | None = None, interval: str = "1d"
) -> dict[str, pd.DataFrame]:
    yf = _require_yfinance()
    raw = yf.download(
        list(symbols), start=start, end=end, interval=interval,
        auto_adjust=True, group_by="ticker", progress=False, threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = raw[sym].rename(columns=str.lower).dropna(how="all")
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        out[sym] = df[list(OHLCV)].astype("float64")
    return out


def load_yfinance_close(ticker: str, start: str, end: str | None = None) -> pd.Series:
    """Close series for a benchmark (e.g. ``^GSPC``) or rate index (``^IRX``)."""
    yf = _require_yfinance()
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"]
    series = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
    series.index = pd.DatetimeIndex(series.index).tz_localize(None)
    return series.astype("float64").rename(ticker)


def load_yfinance_risk_free(start: str, end: str | None = None, ticker: str = "^IRX") -> pd.Series:
    """13-week T-bill yield (quoted in percent) as an annualized decimal rate."""
    return (load_yfinance_close(ticker, start, end) / 100.0).rename("risk_free")


# --------------------------------------------------------------------- synthetic
@dataclass(frozen=True, slots=True)
class SyntheticMarketConfig:
    """Factor-model market with regime-switching drifts and noisy news sentiment.

    Assets load on a market factor and a sector factor, so names inside a sector
    are highly correlated while different sectors are only loosely related.
    Each asset's drift follows a hidden Markov regime in {-1, 0, +1}; the
    simulated NLP sentiment is a noisy, intermittent observation of that regime.
    """

    symbols: tuple[str, ...] = ("TECH1", "TECH2", "TECH3", "ENGY1", "ENGY2", "FIN1", "FIN2", "GOLD")
    sectors: tuple[int, ...] = (0, 0, 0, 1, 1, 2, 2, 3)
    n_bars: int = 1008
    start: str = "2021-01-04"
    seed: int = 42
    market_loading: float = 0.5
    sector_loading: float = 0.6
    vol_range: tuple[float, float] = (0.18, 0.45)
    market_vol: float = 0.16
    market_drift: float = 0.06
    regime_drift: float = 0.45
    regime_switch_prob: float = 0.012
    sentiment_signal: float = 0.9
    sentiment_noise: float = 0.8
    news_probability: float = 0.65
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if len(self.symbols) != len(self.sectors):
            raise ValueError("symbols and sectors must have equal length")
        if self.market_loading**2 + self.sector_loading**2 >= 1.0:
            raise ValueError("market_loading^2 + sector_loading^2 must be < 1")


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    bars: dict[str, pd.DataFrame]
    sentiment: pd.DataFrame
    benchmark: pd.Series
    risk_free: pd.Series
    regimes: pd.DataFrame = field(repr=False)


def _markov_regimes(
    n_bars: int, n_series: int, p_switch: float, rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    """Vectorized regime paths: segment ids from cumulative switch draws, then a
    lookup into a table of random states — no per-step loop."""
    switches = rng.random((n_bars, n_series)) < p_switch
    segment = np.cumsum(switches, axis=0)
    table = rng.choice(np.array([-1, 0, 1]), size=(n_bars + 1, n_series))
    return np.take_along_axis(table, segment, axis=0).astype(np.int64)


def generate_synthetic_market(cfg: SyntheticMarketConfig = SyntheticMarketConfig()) -> SyntheticDataset:
    rng = np.random.default_rng(cfg.seed)
    T, N, ppy = cfg.n_bars, len(cfg.symbols), cfg.periods_per_year
    index = pd.bdate_range(cfg.start, periods=T)
    sectors = np.asarray(cfg.sectors)
    n_sectors = int(sectors.max()) + 1

    a, b = cfg.market_loading, cfg.sector_loading
    m = rng.standard_normal(T)
    s = rng.standard_normal((T, n_sectors))
    e = rng.standard_normal((T, N))
    z = a * m[:, None] + b * s[:, sectors] + np.sqrt(1.0 - a * a - b * b) * e

    ann_vol = rng.uniform(*cfg.vol_range, N)
    daily_vol = ann_vol / np.sqrt(ppy)
    regimes = _markov_regimes(T, N, cfg.regime_switch_prob, rng)
    drift = cfg.market_drift + cfg.regime_drift * regimes
    log_ret = (drift / ppy - 0.5 * daily_vol**2) + daily_vol * z

    start_px = rng.uniform(20.0, 300.0, N)
    close = start_px * np.exp(np.cumsum(log_ret, axis=0))
    prev_close = np.vstack([start_px, close[:-1]])
    gap = 0.25 * daily_vol * rng.standard_normal((T, N))
    open_ = prev_close * np.exp(gap)
    wick = 0.5 * daily_vol
    high = np.maximum(open_, close) * np.exp(np.abs(rng.standard_normal((T, N))) * wick)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.standard_normal((T, N))) * wick)
    base_shares = rng.uniform(1e6, 2e7, N)
    volume = base_shares * np.exp(0.3 * rng.standard_normal((T, N))) * (1.0 + 2.0 * np.abs(z))

    bars = {
        sym: pd.DataFrame(
            {"open": open_[:, i], "high": high[:, i], "low": low[:, i],
             "close": close[:, i], "volume": volume[:, i]},
            index=index,
        )
        for i, sym in enumerate(cfg.symbols)
    }

    raw_sent = np.tanh(cfg.sentiment_signal * regimes + cfg.sentiment_noise * rng.standard_normal((T, N)))
    has_news = rng.random((T, N)) < cfg.news_probability
    sentiment = pd.DataFrame(np.where(has_news, raw_sent, np.nan), index=index, columns=list(cfg.symbols))

    mkt_daily = cfg.market_vol / np.sqrt(ppy)
    bench_log = (cfg.market_drift / ppy - 0.5 * mkt_daily**2) + mkt_daily * m
    benchmark = pd.Series(4000.0 * np.exp(np.cumsum(bench_log)), index=index, name="benchmark")

    # A hiking cycle: ~0.2% to ~5% with mild noise, so the risk-free rate is genuinely dynamic.
    t = np.arange(T)
    rf_path = 0.002 + 0.048 / (1.0 + np.exp(-(t - 0.35 * T) / (0.05 * T)))
    rf_path = np.clip(rf_path + np.cumsum(rng.normal(0.0, 2e-4, T)), 0.0, None)
    risk_free = pd.Series(rf_path, index=index, name="risk_free")

    return SyntheticDataset(
        bars=bars,
        sentiment=sentiment,
        benchmark=benchmark,
        risk_free=risk_free,
        regimes=pd.DataFrame(regimes, index=index, columns=list(cfg.symbols)),
    )
