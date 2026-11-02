from __future__ import annotations

import numpy as np
import pytest

from qengine.events import BoolArray, FloatArray
from qengine.portfolio import EWMACovariance, KellySizer, RiskParitySizer, risk_budget_weights


class StubEstimator:
    """Fixed annualized moments, so sizer math can be checked in closed form."""

    def __init__(self, mean: FloatArray, cov: FloatArray) -> None:
        self._mean, self._cov = mean, cov

    def update(self, returns: FloatArray) -> None:
        pass

    def mean(self) -> FloatArray:
        return self._mean

    def covariance(self) -> FloatArray:
        return self._cov

    @property
    def ready(self) -> BoolArray:
        return np.ones(self._mean.shape[0], dtype=bool)


def corr_cov(vols: list[float], rho: float) -> FloatArray:
    v = np.asarray(vols)
    corr = np.full((v.size, v.size), rho)
    np.fill_diagonal(corr, 1.0)
    return corr * np.outer(v, v)


def risk_contributions(w: FloatArray, cov: FloatArray) -> FloatArray:
    return w * (cov @ w)


ALL = np.ones(3, dtype=bool)


class TestRiskBudget:
    def test_equal_risk_contribution_on_correlated_assets(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.standard_normal((4, 4))
        cov = a @ a.T + 0.1 * np.eye(4)
        w = risk_budget_weights(cov, np.full(4, 0.25))
        rc = risk_contributions(w, cov)
        np.testing.assert_allclose(rc / rc.sum(), 0.25, atol=1e-8)
        assert w.sum() == pytest.approx(1.0)

    def test_custom_budgets(self) -> None:
        cov = corr_cov([0.1, 0.2, 0.3], 0.4)
        budgets = np.array([0.5, 0.3, 0.2])
        rc = risk_contributions(risk_budget_weights(cov, budgets), cov)
        np.testing.assert_allclose(rc / rc.sum(), budgets, atol=1e-8)

    def test_uncorrelated_is_inverse_vol(self) -> None:
        cov = np.diag(np.array([0.1, 0.2, 0.4]) ** 2)
        w = risk_budget_weights(cov, np.full(3, 1 / 3))
        expected = np.array([10.0, 5.0, 2.5])
        np.testing.assert_allclose(w, expected / expected.sum(), atol=1e-10)


class TestRiskParitySizer:
    def test_hits_vol_target(self) -> None:
        cov = corr_cov([0.15, 0.25, 0.40], 0.6)
        sizer = RiskParitySizer(3, target_vol=0.10, estimator=StubEstimator(np.zeros(3), cov))
        w = sizer.target_weights(np.ones(3), ALL)
        assert np.sqrt(w @ cov @ w) == pytest.approx(0.10)
        rc = risk_contributions(w, cov)
        np.testing.assert_allclose(rc, rc[0], rtol=1e-6)

    def test_long_short_equalizes_signed_risk(self) -> None:
        cov = corr_cov([0.2, 0.3, 0.25], 0.8)
        sizer = RiskParitySizer(3, target_vol=0.08, estimator=StubEstimator(np.zeros(3), cov))
        w = sizer.target_weights(np.array([1.0, -1.0, 0.0]), ALL)
        assert w[0] > 0 > w[1]
        assert w[2] == 0.0
        rc = risk_contributions(w, cov)
        assert rc[0] == pytest.approx(rc[1], rel=1e-6)
        assert np.sqrt(w @ cov @ w) == pytest.approx(0.08)

    def test_untradable_or_flat_assets_get_zero(self) -> None:
        sizer = RiskParitySizer(3, estimator=StubEstimator(np.zeros(3), np.eye(3) * 0.04))
        w = sizer.target_weights(np.array([1.0, 1.0, 0.0]), np.array([True, False, True]))
        assert w[1] == 0.0 and w[2] == 0.0 and w[0] > 0


class TestKellySizer:
    def test_single_asset_closed_form(self) -> None:
        # f* = fraction * (mu - rf) / sigma^2 = 0.5 * (0.10 - 0.02) / 0.04 = 1.0
        sizer = KellySizer(1, fraction=0.5, estimator=StubEstimator(np.array([0.10]), np.array([[0.04]])))
        sizer.set_risk_free(0.02)
        w = sizer.target_weights(np.array([1.0]), np.array([True]))
        assert w[0] == pytest.approx(1.0)

    def test_conviction_scales_edge(self) -> None:
        sizer = KellySizer(1, fraction=0.5, estimator=StubEstimator(np.array([0.10]), np.array([[0.04]])))
        w = sizer.target_weights(np.array([0.5]), np.array([True]))
        assert w[0] == pytest.approx(0.5 * 0.5 * 0.10 / 0.04)

    def test_never_bets_against_signal(self) -> None:
        sizer = KellySizer(2, estimator=StubEstimator(np.array([-0.1, 0.1]), np.eye(2) * 0.04))
        w = sizer.target_weights(np.array([1.0, -1.0]), np.ones(2, dtype=bool))
        np.testing.assert_allclose(w, 0.0)

    def test_short_with_negative_drift(self) -> None:
        sizer = KellySizer(1, fraction=1.0, estimator=StubEstimator(np.array([-0.08]), np.array([[0.04]])))
        w = sizer.target_weights(np.array([-1.0]), np.array([True]))
        assert w[0] == pytest.approx(-2.0)

    def test_risk_free_hurdle_removes_edge(self) -> None:
        sizer = KellySizer(1, estimator=StubEstimator(np.array([0.03]), np.array([[0.04]])))
        sizer.set_risk_free(0.05)
        assert sizer.target_weights(np.array([1.0]), np.array([True]))[0] == 0.0


class TestEWMACovariance:
    def test_recovers_true_covariance(self) -> None:
        rng = np.random.default_rng(3)
        true_daily = corr_cov([0.01, 0.02], 0.5)
        est = EWMACovariance(2, halflife=2_000, min_periods=10, periods_per_year=1)
        for r in rng.multivariate_normal([0.0, 0.0], true_daily, size=20_000):
            est.update(r)
        np.testing.assert_allclose(est.covariance(), true_daily, rtol=0.1)
        assert est.ready.all()

    def test_missing_returns_do_not_update_pairs(self) -> None:
        est = EWMACovariance(2, halflife=5, min_periods=1)
        est.update(np.array([0.01, 0.02]))
        est.update(np.array([0.03, -0.01]))
        before = est.covariance().copy()
        est.update(np.array([np.nan, 0.05]))
        after = est.covariance()
        assert after[0, 0] == before[0, 0]
        assert after[0, 1] == before[0, 1]
        assert after[1, 1] != before[1, 1]
