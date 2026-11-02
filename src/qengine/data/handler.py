"""DataHandler: turns raw bars + alternative data into a stream of MarketEvents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Mapping

import numpy as np
import pandas as pd

from qengine.data.features import FEATURE_FIELDS, FeatureConfig, compute_features
from qengine.events import BoolArray, FloatArray, MarketEvent, Universe

OHLCV: tuple[str, ...] = ("open", "high", "low", "close", "volume")
ALT_FIELDS: tuple[str, ...] = ("sentiment", "risk_free")


class DataHandler(ABC):
    """Source of truth for what the engine is allowed to know, and when."""

    @property
    @abstractmethod
    def universe(self) -> Universe: ...

    @property
    @abstractmethod
    def periods_per_year(self) -> int: ...

    @abstractmethod
    def stream(self) -> Iterator[MarketEvent]:
        """Yield one cross-sectional MarketEvent per timestamp, in order."""

    def benchmark(self) -> pd.Series | None:
        """Benchmark *price* series aligned to the event index, if available."""
        return None

    def risk_free(self) -> pd.Series | None:
        """Annualized risk-free rate aligned to the event index, if available."""
        return None


def _align_series(series: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series | None:
    """As-of align an exogenous series onto the event index (forward-fill only)."""
    if series is None:
        return None
    s = series.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(index)).ffill().reindex(index)


class HistoricalDataHandler(DataHandler):
    """Replays a historical, multi-asset panel as if it were a live feed.

    Construction does all heavy lifting with vectorized pandas/NumPy:

    1. Bars for each symbol are pivoted into wide ``time x symbol`` frames.
    2. Sentiment is as-of joined onto the same grid (missing = no news that bar).
    3. Every indicator in :mod:`qengine.data.features` is computed column-wise.
    4. Everything is packed into a contiguous, read-only ``(T, N, F)`` float cube.

    ``stream()`` then yields zero-copy ``(N, F)`` slices of that cube.
    """

    def __init__(
        self,
        bars: Mapping[str, pd.DataFrame],
        *,
        sentiment: pd.DataFrame | None = None,
        benchmark: pd.Series | None = None,
        risk_free: pd.Series | None = None,
        features: FeatureConfig = FeatureConfig(),
        default_risk_free: float = 0.0,
    ) -> None:
        if not bars:
            raise ValueError("At least one symbol is required")
        symbols = tuple(bars)
        for sym, df in bars.items():
            missing = [c for c in OHLCV if c not in df.columns]
            if missing:
                raise ValueError(f"{sym}: missing OHLCV columns {missing}")
            if not df.index.is_monotonic_increasing:
                raise ValueError(f"{sym}: index must be sorted ascending")

        panel: dict[str, pd.DataFrame] = {
            f: pd.DataFrame({s: bars[s][f].astype("float64") for s in symbols}).sort_index()
            for f in OHLCV
        }
        index = pd.DatetimeIndex(panel["close"].index)
        self._index = index
        self._features_cfg = features

        if sentiment is not None:
            sent = sentiment.reindex(index=index, columns=list(symbols)).astype("float64")
        else:
            sent = pd.DataFrame(np.nan, index=index, columns=list(symbols))
        panel["sentiment"] = sent

        rf = _align_series(risk_free, index)
        rf_values = (
            rf.fillna(default_risk_free).to_numpy(dtype=np.float64)
            if rf is not None
            else np.full(len(index), default_risk_free)
        )
        self._risk_free = pd.Series(rf_values, index=index, name="risk_free")
        panel["risk_free"] = pd.DataFrame(
            np.repeat(rf_values[:, None], len(symbols), axis=1), index=index, columns=list(symbols)
        )
        self._benchmark = _align_series(benchmark, index)

        panel.update(compute_features(panel, features))

        fields = OHLCV + ALT_FIELDS + FEATURE_FIELDS
        self._universe = Universe(symbols=symbols, fields=fields)
        cube = np.stack([panel[f].to_numpy(dtype=np.float64) for f in fields], axis=-1)
        self._cube: FloatArray = np.ascontiguousarray(cube)
        self._cube.setflags(write=False)

        close = self._cube[:, :, fields.index("close")]
        open_ = self._cube[:, :, fields.index("open")]
        available: BoolArray = np.isfinite(close) & np.isfinite(open_) & (close > 0)
        available.setflags(write=False)
        self._available = available

    @property
    def universe(self) -> Universe:
        return self._universe

    @property
    def periods_per_year(self) -> int:
        return self._features_cfg.periods_per_year

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._index

    def __len__(self) -> int:
        return len(self._index)

    def stream(self) -> Iterator[MarketEvent]:
        universe, cube, available = self._universe, self._cube, self._available
        for t, ts in enumerate(self._index):
            yield MarketEvent(timestamp=ts, universe=universe, values=cube[t], available=available[t])

    def benchmark(self) -> pd.Series | None:
        return self._benchmark

    def risk_free(self) -> pd.Series | None:
        return self._risk_free

    def panel(self, field: str) -> pd.DataFrame:
        """Wide DataFrame of one field — for research/diagnostics, not the hot path."""
        j = self._universe.field_index(field)
        return pd.DataFrame(self._cube[:, :, j], index=self._index, columns=list(self._universe.symbols))
