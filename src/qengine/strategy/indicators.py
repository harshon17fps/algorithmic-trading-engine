"""Online (streaming) indicators, vectorized across the asset dimension.

These complement the batch features in :mod:`qengine.data.features`: anything
that must react to state only known at run time (e.g. irregular news arrival)
is updated here one event at a time, in O(n_assets) per bar.
"""

from __future__ import annotations

import numpy as np

from qengine.events import BoolArray, FloatArray


class OnlineEWMA:
    """Exponentially weighted mean that ignores missing observations.

    An asset's value is seeded by its first valid observation and is left
    unchanged on bars where its input is NaN or the asset did not trade.
    """

    __slots__ = ("alpha", "_value")

    def __init__(self, n_assets: int, span: float) -> None:
        if span < 1:
            raise ValueError("span must be >= 1")
        self.alpha = 2.0 / (span + 1.0)
        self._value: FloatArray = np.full(n_assets, np.nan)

    def update(self, x: FloatArray, mask: BoolArray | None = None) -> FloatArray:
        valid = np.isfinite(x) if mask is None else (mask & np.isfinite(x))
        seed = valid & np.isnan(self._value)
        step = valid & ~seed
        self._value[seed] = x[seed]
        self._value[step] += self.alpha * (x[step] - self._value[step])
        return self._value.copy()

    @property
    def value(self) -> FloatArray:
        return self._value.copy()


class OnlineEWMVariance:
    """Exponentially weighted mean and variance (West's incremental form)."""

    __slots__ = ("alpha", "_mean", "_var")

    def __init__(self, n_assets: int, span: float) -> None:
        self.alpha = 2.0 / (span + 1.0)
        self._mean: FloatArray = np.full(n_assets, np.nan)
        self._var: FloatArray = np.zeros(n_assets)

    def update(self, x: FloatArray, mask: BoolArray | None = None) -> FloatArray:
        valid = np.isfinite(x) if mask is None else (mask & np.isfinite(x))
        seed = valid & np.isnan(self._mean)
        step = valid & ~seed
        self._mean[seed] = x[seed]
        diff = x[step] - self._mean[step]
        incr = self.alpha * diff
        self._mean[step] += incr
        self._var[step] = (1.0 - self.alpha) * (self._var[step] + diff * incr)
        return self.std

    @property
    def mean(self) -> FloatArray:
        return self._mean.copy()

    @property
    def std(self) -> FloatArray:
        return np.sqrt(self._var)
