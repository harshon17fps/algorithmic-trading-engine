"""Streaming return-moment estimators consumed by the position sizers."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from qengine.events import BoolArray, FloatArray, IntArray


class ReturnEstimator(Protocol):
    """Annualized first and second moments of asset returns, updated per bar."""

    def update(self, returns: FloatArray) -> None: ...

    def mean(self) -> FloatArray: ...

    def covariance(self) -> FloatArray: ...

    @property
    def ready(self) -> BoolArray: ...


class EWMACovariance:
    """Exponentially weighted mean/covariance with per-asset missing-data handling.

    Uses the incremental form ``S <- (1 - a) * (S + a * d d')`` so each update
    is a single O(N^2) outer product. Only pairs where both assets printed a
    return this bar are updated; others are left untouched.
    """

    __slots__ = ("_alpha", "_ppy", "_min_periods", "_mean", "_cov", "_count")

    def __init__(
        self, n_assets: int, *, halflife: float = 30.0, min_periods: int = 30, periods_per_year: int = 252
    ) -> None:
        self._alpha = 1.0 - np.exp(np.log(0.5) / halflife)
        self._ppy = periods_per_year
        self._min_periods = min_periods
        self._mean: FloatArray = np.zeros(n_assets)
        self._cov: FloatArray = np.zeros((n_assets, n_assets))
        self._count: IntArray = np.zeros(n_assets, dtype=np.int64)

    def update(self, returns: FloatArray) -> None:
        valid = np.isfinite(returns)
        if not valid.any():
            return
        a = self._alpha
        r = np.where(valid, returns, 0.0)
        seed = valid & (self._count == 0)
        self._mean[seed] = r[seed]
        d = np.where(valid & ~seed, r - self._mean, 0.0)
        self._mean += a * d
        both = np.outer(valid, valid)
        self._cov = np.where(both, (1.0 - a) * (self._cov + a * np.outer(d, d)), self._cov)
        self._count += valid

    def mean(self) -> FloatArray:
        return self._mean * self._ppy

    def covariance(self) -> FloatArray:
        return self._cov * self._ppy

    @property
    def ready(self) -> BoolArray:
        return self._count >= self._min_periods
