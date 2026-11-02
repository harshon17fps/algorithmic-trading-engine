"""Dynamic position sizing: map signed conviction to target portfolio weights.

Sizers see the *intent* vector (signed signal strength per asset, in [-1, 1])
and return target weights as a fraction of equity. Hard risk limits are
applied afterwards by :class:`qengine.portfolio.risk.RiskLimits`, so sizers
can focus purely on the allocation model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from qengine.events import BoolArray, FloatArray, MarketEvent
from qengine.portfolio.estimators import EWMACovariance, ReturnEstimator


class PositionSizer(ABC):
    def update(self, event: MarketEvent) -> None:
        """Ingest the latest bar (called once per MarketEvent, before sizing)."""

    @abstractmethod
    def target_weights(self, intent: FloatArray, tradable: BoolArray) -> FloatArray: ...


class FixedFractionSizer(PositionSizer):
    """Baseline: ``weight = intent * fraction``. Useful for tests and A/B comparisons."""

    def __init__(self, fraction: float = 0.1) -> None:
        self.fraction = fraction

    def target_weights(self, intent: FloatArray, tradable: BoolArray) -> FloatArray:
        return np.where(tradable, intent * self.fraction, 0.0)


def risk_budget_weights(
    cov: FloatArray, budgets: FloatArray, *, tol: float = 1e-10, max_iter: int = 500
) -> FloatArray:
    """Long-only risk-budgeting portfolio via cyclical coordinate descent.

    Solves ``min 0.5 w'Σw - Σ b_i ln w_i`` (Griveau-Billion, Richard & Roncalli,
    2013), whose optimum satisfies ``w_i (Σw)_i ∝ b_i`` — i.e. each asset's risk
    contribution matches its budget. Returns weights normalised to sum to 1.
    """
    n = budgets.shape[0]
    cov = cov + np.eye(n) * 1e-12 * max(float(np.trace(cov)), 1.0)
    diag = np.diag(cov)
    w: FloatArray = 1.0 / np.sqrt(diag)
    w /= w.sum()
    for _ in range(max_iter):
        w_prev = w.copy()
        for i in range(n):
            b = cov[i] @ w - diag[i] * w[i]
            w[i] = (-b + np.sqrt(b * b + 4.0 * diag[i] * budgets[i])) / (2.0 * diag[i])
        if np.max(np.abs(w - w_prev)) < tol * np.max(np.abs(w)):
            break
    normalized: FloatArray = w / w.sum()
    return normalized


def _active_set(
    intent: FloatArray, tradable: BoolArray, estimator: ReturnEstimator
) -> tuple[np.ndarray, FloatArray]:
    active = (intent != 0) & tradable & estimator.ready
    idx = np.flatnonzero(active)
    return idx, np.sign(intent[idx])


class RiskParitySizer(PositionSizer):
    """Equal (or conviction-weighted) risk contribution, scaled to a vol target.

    Shorts are handled by solving on ``DΣD`` with ``D = diag(sign)``, which
    turns a long/short book into an equivalent long-only problem; the signs
    are re-applied afterwards. Risk budgets are proportional to |intent|.
    """

    def __init__(
        self,
        n_assets: int,
        *,
        target_vol: float = 0.10,
        estimator: ReturnEstimator | None = None,
        periods_per_year: int = 252,
    ) -> None:
        self.target_vol = target_vol
        self.estimator: ReturnEstimator = estimator or EWMACovariance(
            n_assets, halflife=40, min_periods=40, periods_per_year=periods_per_year
        )

    def update(self, event: MarketEvent) -> None:
        self.estimator.update(event.field("ret"))

    def target_weights(self, intent: FloatArray, tradable: BoolArray) -> FloatArray:
        weights = np.zeros_like(intent, dtype=np.float64)
        idx, sign = _active_set(intent, tradable, self.estimator)
        if idx.size == 0:
            return weights
        cov = self.estimator.covariance()[np.ix_(idx, idx)] * np.outer(sign, sign)
        budgets = np.abs(intent[idx])
        x = risk_budget_weights(cov, budgets / budgets.sum())
        port_vol = float(np.sqrt(x @ cov @ x))
        if port_vol <= 0 or not np.isfinite(port_vol):
            return weights
        weights[idx] = sign * x * (self.target_vol / port_vol)
        return weights


class KellySizer(PositionSizer):
    """Fractional multivariate Kelly: ``w = f · Σ⁻¹ (μ − r_f)`` on the active set.

    * Expected excess returns use the estimator's EWMA mean minus the *current*
      risk-free rate streamed in the MarketEvent (``risk_free`` field).
    * Only edge in the direction of the signal is used; assets whose estimated
      drift disagrees with the strategy get no allocation (never bet against
      the signal), and conviction scales each asset's edge.
    * The covariance is shrunk toward its diagonal for a stable inverse.
    """

    def __init__(
        self,
        n_assets: int,
        *,
        fraction: float = 0.5,
        shrinkage: float = 0.3,
        estimator: ReturnEstimator | None = None,
        periods_per_year: int = 252,
    ) -> None:
        if not 0 < fraction <= 1:
            raise ValueError("Kelly fraction must be in (0, 1]")
        self.fraction = fraction
        self.shrinkage = shrinkage
        self.estimator: ReturnEstimator = estimator or EWMACovariance(
            n_assets, halflife=60, min_periods=60, periods_per_year=periods_per_year
        )
        self._risk_free = 0.0

    def update(self, event: MarketEvent) -> None:
        self.estimator.update(event.field("ret"))
        if event.universe.has_field("risk_free"):
            rf = event.field("risk_free")
            finite = rf[np.isfinite(rf)]
            if finite.size:
                self._risk_free = float(finite[0])

    def set_risk_free(self, rate: float) -> None:
        self._risk_free = rate

    def target_weights(self, intent: FloatArray, tradable: BoolArray) -> FloatArray:
        weights = np.zeros_like(intent, dtype=np.float64)
        idx, sign = _active_set(intent, tradable, self.estimator)
        if idx.size == 0:
            return weights
        excess = self.estimator.mean()[idx] - self._risk_free
        edge = np.maximum(sign * excess, 0.0) * np.abs(intent[idx])
        if not np.any(edge > 0):
            return weights
        cov = self.estimator.covariance()[np.ix_(idx, idx)]
        shrunk = (1.0 - self.shrinkage) * cov + self.shrinkage * np.diag(np.diag(cov))
        raw = self.fraction * np.linalg.solve(shrunk, sign * edge)
        weights[idx] = np.where(sign * raw > 0, raw, 0.0)
        return weights
