"""Transaction-cost models: commissions and market impact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class CommissionModel(Protocol):
    def __call__(self, quantity: float, price: float) -> float: ...


class SlippageModel(Protocol):
    def impact(self, quantity: float, price: float, bar_volume: float, annual_vol: float) -> float:
        """Adverse price move as a *fraction* of price (always >= 0)."""
        ...


@dataclass(frozen=True, slots=True)
class PerShareCommission:
    """Interactive-Brokers-style tiered pricing: per share, with min and % cap."""

    per_share: float = 0.005
    minimum: float = 1.0
    max_pct_of_notional: float = 0.01

    def __call__(self, quantity: float, price: float) -> float:
        fee = max(abs(quantity) * self.per_share, self.minimum)
        return min(fee, self.max_pct_of_notional * abs(quantity) * price)


@dataclass(frozen=True, slots=True)
class BpsCommission:
    """Percentage-of-notional fee (crypto venues, most FX/futures brokers)."""

    bps: float = 1.0

    def __call__(self, quantity: float, price: float) -> float:
        return abs(quantity) * price * self.bps * 1e-4


@dataclass(frozen=True, slots=True)
class FixedBpsSlippage:
    bps: float = 5.0

    def impact(self, quantity: float, price: float, bar_volume: float, annual_vol: float) -> float:
        return self.bps * 1e-4


@dataclass(frozen=True, slots=True)
class SquareRootImpact:
    """Half-spread plus the square-root market-impact law.

    ``impact = spread/2 + k · σ_daily · sqrt(Q / V)`` — the empirical form used
    by most institutional TCA models (Almgren et al., 2005; Tóth et al., 2011).
    Falls back to the half-spread when volume or volatility is unknown.
    """

    spread_bps: float = 4.0
    impact_coef: float = 0.5
    periods_per_year: int = 252

    def impact(self, quantity: float, price: float, bar_volume: float, annual_vol: float) -> float:
        half_spread = 0.5 * self.spread_bps * 1e-4
        if not (np.isfinite(bar_volume) and bar_volume > 0 and np.isfinite(annual_vol)):
            return half_spread
        daily_vol = annual_vol / float(np.sqrt(self.periods_per_year))
        return half_spread + self.impact_coef * daily_vol * float(np.sqrt(abs(quantity) / bar_volume))
