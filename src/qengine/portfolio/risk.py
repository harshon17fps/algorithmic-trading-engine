"""Hard risk constraints and margin rules, independent of any sizing model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qengine.events import FloatArray


@dataclass(frozen=True, slots=True)
class MarginModel:
    """Margin as a fraction of position notional.

    Requirement = Σ long_mv · long_rate + Σ |short_mv| · short_rate, and the
    account is compliant while ``equity >= requirement``.

    * ``long_initial=1.0`` is a cash account (no leverage on longs);
      ``0.5`` is US Reg-T (2x buying power).
    * ``short_initial=0.5`` is Reg-T's 150% rule: proceeds stay in the account
      as cash and an extra 50% of the short value must be posted as equity.
    """

    long_initial: float = 1.0
    short_initial: float = 0.5
    long_maintenance: float = 0.25
    short_maintenance: float = 0.30

    def __post_init__(self) -> None:
        if self.long_maintenance > self.long_initial or self.short_maintenance > self.short_initial:
            raise ValueError("Maintenance margin cannot exceed initial margin")

    def requirement(self, market_values: FloatArray, *, maintenance: bool = False) -> float:
        long_rate = self.long_maintenance if maintenance else self.long_initial
        short_rate = self.short_maintenance if maintenance else self.short_initial
        mv = np.nan_to_num(market_values)
        return float(np.sum(np.where(mv > 0, mv * long_rate, -mv * short_rate)))


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_position_weight: float = 0.25
    max_gross_leverage: float = 1.5
    max_net_leverage: float = 1.0
    allow_short: bool = True
    rebalance_band: float = 0.01
    margin_buffer: float = 0.05
    margin: MarginModel = field(default_factory=MarginModel)

    def apply(self, weights: FloatArray) -> FloatArray:
        """Project target weights onto the feasible set.

        Each constraint scales the whole book proportionally (after per-name
        clipping), so relative risk allocations from the sizer are preserved.
        """
        w = np.nan_to_num(np.asarray(weights, dtype=np.float64))
        if not self.allow_short:
            w = np.maximum(w, 0.0)
        w = np.clip(w, -self.max_position_weight, self.max_position_weight)

        gross = float(np.abs(w).sum())
        if gross > self.max_gross_leverage:
            w *= self.max_gross_leverage / gross
        net = float(w.sum())
        if abs(net) > self.max_net_leverage:
            w *= self.max_net_leverage / abs(net)

        # With market value = w * equity, the requirement in units of equity is
        # just the requirement evaluated on the weights themselves.
        required = self.margin.requirement(w)
        capacity = 1.0 - self.margin_buffer
        if required > capacity:
            w *= capacity / required
        return w
