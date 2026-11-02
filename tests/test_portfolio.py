"""Portfolio accounting, margin and order-generation tests.

Every expected number below is derived by hand in the comments so a reviewer
can audit the trade math without running the engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import T0, make_fill, make_market
from qengine.events import Direction, FinancingEvent, Side, SignalEvent, Universe
from qengine.portfolio import (
    FixedFractionSizer,
    MarginModel,
    Portfolio,
    RiskLimits,
    apply_fill,
)

CAPITAL = 100_000.0


def make_portfolio(
    universe: Universe,
    *,
    fraction: float = 0.1,
    limits: RiskLimits | None = None,
    lot_size: float = 1.0,
    capital: float = CAPITAL,
) -> Portfolio:
    return Portfolio(
        universe,
        capital,
        FixedFractionSizer(fraction),
        limits or RiskLimits(max_position_weight=1.0, max_gross_leverage=10.0, max_net_leverage=10.0),
        lot_size=lot_size,
    )


def signal(symbol: str, direction: Direction, strength: float = 1.0) -> SignalEvent:
    return SignalEvent(timestamp=T0, symbol=symbol, direction=direction, strength=strength)


# --------------------------------------------------------------------------- pure position math
class TestApplyFill:
    def test_open_long(self) -> None:
        assert apply_fill(0.0, 0.0, 100.0, 50.0) == (100.0, 50.0, 0.0)

    def test_increase_uses_weighted_average_cost(self) -> None:
        # (100*50 + 300*54) / 400 = 53.0
        assert apply_fill(100.0, 50.0, 300.0, 54.0) == (400.0, 53.0, 0.0)

    def test_partial_close_realizes_only_closed_quantity(self) -> None:
        # Close 40 of 100 @ 60 vs avg 50 -> 40 * 10 = 400; avg cost unchanged.
        assert apply_fill(100.0, 50.0, -40.0, 60.0) == (60.0, 50.0, 400.0)

    def test_full_close_resets_average(self) -> None:
        assert apply_fill(100.0, 50.0, -100.0, 45.0) == (0.0, 0.0, -500.0)

    def test_short_cover_profit(self) -> None:
        # Short 100 @ 50, cover 100 @ 45 -> +500.
        assert apply_fill(-100.0, 50.0, 100.0, 45.0) == (0.0, 0.0, 500.0)

    def test_flip_long_to_short(self) -> None:
        # Long 100 @ 50, sell 250 @ 60: realize 100 * 10 = 1000; new short 150 @ 60.
        assert apply_fill(100.0, 50.0, -250.0, 60.0) == (-150.0, 60.0, 1000.0)


# --------------------------------------------------------------------------- cash & mark-to-market
class TestCashAndEquity:
    def test_initial_state(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        assert pf.cash == CAPITAL
        assert pf.equity == CAPITAL
        assert pf.gross_exposure == 0.0
        assert pf.margin_requirement() == 0.0
        assert np.all(pf.quantities == 0.0)

    def test_rejects_non_positive_capital(self, universe: Universe) -> None:
        with pytest.raises(ValueError):
            Portfolio(universe, 0.0, FixedFractionSizer())

    def test_buy_debits_notional_plus_commission(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_fill(make_fill("AAA", Side.BUY, 100, 50.0, commission=1.0))
        # 100,000 - 100*50 - 1 = 94,999
        assert pf.cash == pytest.approx(94_999.0)
        assert pf.position("AAA").quantity == 100
        assert pf.position("AAA").avg_price == 50.0
        # Unmarked position is valued at its fill price: equity only loses the commission.
        assert pf.equity == pytest.approx(99_999.0)
        assert pf.total_commission == 1.0

    def test_mark_to_market(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_fill(make_fill("AAA", Side.BUY, 100, 50.0, commission=1.0))
        pf.on_market(make_market(universe, [55.0, 10.0, 10.0]))
        # 94,999 cash + 100 * 55 = 100,499
        assert pf.equity == pytest.approx(100_499.0)
        assert pf.unrealized_pnl == pytest.approx(500.0)
        assert pf.position("AAA").market_value == pytest.approx(5_500.0)

    def test_missing_price_keeps_last_mark(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_fill(make_fill("AAA", Side.BUY, 100, 50.0))
        pf.on_market(make_market(universe, [55.0, 10.0, 10.0]))
        pf.on_market(make_market(universe, [np.nan, 10.0, 10.0]))
        assert pf.position("AAA").last_price == 55.0
        assert pf.equity == pytest.approx(CAPITAL + 500.0)

    def test_partial_sell_cash_and_realized(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_fill(make_fill("AAA", Side.BUY, 100, 50.0))
        pf.on_fill(make_fill("AAA", Side.SELL, 40, 60.0))
        # 100,000 - 5,000 + 2,400 = 97,400
        assert pf.cash == pytest.approx(97_400.0)
        assert pf.realized_pnl == pytest.approx(400.0)
        assert pf.position("AAA").quantity == 60

    def test_short_sale_credits_proceeds_and_marks_inversely(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_fill(make_fill("BBB", Side.SELL, 100, 50.0))
        assert pf.cash == pytest.approx(105_000.0)
        assert pf.position("BBB").quantity == -100
        assert pf.equity == pytest.approx(CAPITAL)
        pf.on_market(make_market(universe, [10.0, 45.0, 10.0]))
        # 105,000 - 100 * 45 = 100,500
        assert pf.equity == pytest.approx(100_500.0)
        assert pf.net_exposure == pytest.approx(-4_500.0)
        assert pf.gross_exposure == pytest.approx(4_500.0)

    def test_borrow_fee_debits_cash(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_fill(make_fill("BBB", Side.SELL, 100, 50.0))
        pf.on_financing(FinancingEvent(timestamp=T0, symbol="BBB", amount=12.5))
        assert pf.cash == pytest.approx(105_000.0 - 12.5)
        assert pf.total_financing == 12.5

    def test_accounting_identity_holds_over_random_fills(self, universe: Universe) -> None:
        """equity == capital + realized + unrealized - commissions - financing, always."""
        rng = np.random.default_rng(0)
        pf = make_portfolio(universe)
        for step in range(500):
            sym = universe.symbols[rng.integers(0, 3)]
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            price = float(rng.uniform(20, 80))
            pf.on_fill(make_fill(sym, side, float(rng.integers(1, 300)), price, commission=float(rng.uniform(0, 3))))
            if step % 7 == 0:
                pf.on_financing(FinancingEvent(timestamp=T0, symbol=sym, amount=float(rng.uniform(0, 5))))
            if step % 5 == 0:
                pf.on_market(make_market(universe, rng.uniform(20, 80, 3)))
            expected = (
                CAPITAL + pf.realized_pnl + pf.unrealized_pnl - pf.total_commission - pf.total_financing
            )
            assert pf.equity == pytest.approx(expected, rel=1e-12, abs=1e-6)


# --------------------------------------------------------------------------- trade records
class TestTradeRecords:
    def test_round_trip_is_net_of_commissions(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_fill(make_fill("AAA", Side.BUY, 100, 50.0, commission=1.0, ts=T0))
        pf.on_fill(make_fill("AAA", Side.SELL, 100, 55.0, commission=1.0, ts=T0 + pd.Timedelta(days=5)))
        (trade,) = pf.closed_trades
        assert trade.direction is Direction.LONG
        assert (trade.entry_price, trade.exit_price) == (50.0, 55.0)
        assert trade.gross_pnl == pytest.approx(500.0)
        assert trade.costs == pytest.approx(2.0)
        assert trade.net_pnl == pytest.approx(498.0)
        assert trade.holding_period == pd.Timedelta(days=5)

    def test_scaling_in_and_out_is_one_trade(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_fill(make_fill("AAA", Side.BUY, 100, 50.0))
        pf.on_fill(make_fill("AAA", Side.BUY, 100, 60.0))   # avg 55
        pf.on_fill(make_fill("AAA", Side.SELL, 50, 70.0))   # +750
        assert pf.closed_trades == []
        pf.on_fill(make_fill("AAA", Side.SELL, 150, 40.0))  # -2,250
        (trade,) = pf.closed_trades
        assert trade.entry_price == pytest.approx(55.0)
        assert trade.exit_price == pytest.approx((50 * 70 + 150 * 40) / 200)
        assert trade.max_quantity == 200
        assert trade.gross_pnl == pytest.approx(-1_500.0)

    def test_flip_closes_trade_and_splits_commission(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_fill(make_fill("AAA", Side.BUY, 100, 50.0, commission=1.0))
        pf.on_fill(make_fill("AAA", Side.SELL, 250, 60.0, commission=5.0))
        (long_trade,) = pf.closed_trades
        # 100/250 of the flip commission belongs to the closing leg: 1 + 2 = 3.
        assert long_trade.costs == pytest.approx(3.0)
        assert long_trade.gross_pnl == pytest.approx(1_000.0)
        (short_trade,) = [t for t in pf.trades() if t.is_open]
        assert short_trade.direction is Direction.SHORT
        assert short_trade.costs == pytest.approx(3.0)
        assert short_trade.entry_price == 60.0

    def test_borrow_fees_are_attributed_to_the_open_trade(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_fill(make_fill("BBB", Side.SELL, 100, 50.0))
        pf.on_financing(FinancingEvent(timestamp=T0, symbol="BBB", amount=4.0))
        pf.on_fill(make_fill("BBB", Side.BUY, 100, 48.0))
        (trade,) = pf.closed_trades
        assert trade.gross_pnl == pytest.approx(200.0)
        assert trade.net_pnl == pytest.approx(196.0)


# --------------------------------------------------------------------------- margin
class TestMargin:
    def test_cash_account_long_requirement(self, universe: Universe) -> None:
        pf = make_portfolio(universe, limits=RiskLimits(margin=MarginModel(long_initial=1.0, long_maintenance=0.25)))
        pf.on_fill(make_fill("AAA", Side.BUY, 100, 50.0))
        assert pf.margin_requirement() == pytest.approx(5_000.0)
        assert pf.margin_requirement(maintenance=True) == pytest.approx(1_250.0)
        assert pf.excess_liquidity == pytest.approx(95_000.0)

    def test_reg_t_short_requirement(self, universe: Universe) -> None:
        pf = make_portfolio(universe, limits=RiskLimits(margin=MarginModel(short_initial=0.5, short_maintenance=0.3)))
        pf.on_fill(make_fill("BBB", Side.SELL, 100, 50.0))
        assert pf.margin_requirement() == pytest.approx(2_500.0)
        assert pf.margin_requirement(maintenance=True) == pytest.approx(1_500.0)

    def test_mixed_book_requirement(self, universe: Universe) -> None:
        pf = make_portfolio(universe, limits=RiskLimits(margin=MarginModel(long_initial=0.5, short_initial=0.5)))
        pf.on_fill(make_fill("AAA", Side.BUY, 200, 50.0))   # 10,000 long
        pf.on_fill(make_fill("BBB", Side.SELL, 100, 40.0))  # 4,000 short
        assert pf.margin_requirement() == pytest.approx(0.5 * 10_000 + 0.5 * 4_000)

    def test_maintenance_cannot_exceed_initial(self) -> None:
        with pytest.raises(ValueError):
            MarginModel(long_initial=0.3, long_maintenance=0.4)

    def test_margin_call_deleverages_proportionally(self, universe: Universe) -> None:
        limits = RiskLimits(
            max_position_weight=10, max_gross_leverage=10, max_net_leverage=10, margin_buffer=0.05,
            margin=MarginModel(long_initial=0.5, long_maintenance=0.25),
        )
        pf = make_portfolio(universe, limits=limits)
        pf.on_fill(make_fill("AAA", Side.BUY, 4_000, 50.0))  # 200k on 100k equity (Reg-T 2x)
        assert pf.cash == pytest.approx(-100_000.0)
        assert not pf.in_margin_call

        pf.on_market(make_market(universe, [30.0, 10.0, 10.0]))
        # MV 120k, equity 20k < maintenance 0.25 * 120k = 30k.
        assert pf.equity == pytest.approx(20_000.0)
        assert pf.in_margin_call

        (order,) = pf.generate_orders(T0)
        # Target scale = 20k * 0.95 / (0.5 * 120k) = 0.31667 -> trunc(4000 * 0.31667) = 1266.
        assert order.reason == "margin_call"
        assert order.side is Side.SELL
        assert order.quantity == 4_000 - 1_266
        pf.on_fill(make_fill("AAA", Side.SELL, order.quantity, 30.0, order_id=order.order_id))
        assert not pf.in_margin_call
        assert pf.margin_requirement() <= pf.equity

    def test_negative_equity_liquidates_and_halts(self, universe: Universe) -> None:
        limits = RiskLimits(margin=MarginModel(long_initial=0.5, long_maintenance=0.25))
        pf = make_portfolio(universe, limits=limits)
        pf.on_fill(make_fill("AAA", Side.BUY, 4_000, 50.0))
        pf.on_fill(make_fill("BBB", Side.SELL, 100, 10.0))
        pf.on_market(make_market(universe, [20.0, 10.0, 10.0]))  # equity = -100k + 80k + 1k - 1k < 0
        orders = pf.generate_orders(T0)
        assert pf.halted
        assert {o.symbol: o.signed_quantity for o in orders} == {"AAA": -4_000, "BBB": 100}
        assert all(o.reason == "liquidation" for o in orders)
        assert pf.generate_orders(T0) == []


# --------------------------------------------------------------------------- risk limits
class TestRiskLimits:
    def test_per_name_cap(self) -> None:
        w = RiskLimits(max_position_weight=0.2).apply(np.array([0.5, -0.5, 0.1]))
        np.testing.assert_allclose(w, [0.2, -0.2, 0.1])

    def test_gross_leverage_scales_book(self) -> None:
        limits = RiskLimits(max_position_weight=1, max_gross_leverage=1.0, max_net_leverage=5,
                            margin=MarginModel(short_initial=0.3))
        w = limits.apply(np.array([0.6, -0.6, 0.3]))
        assert np.abs(w).sum() == pytest.approx(1.0)
        np.testing.assert_allclose(w / w[0], [1.0, -1.0, 0.5])

    def test_net_leverage_cap(self) -> None:
        limits = RiskLimits(max_position_weight=1, max_gross_leverage=5, max_net_leverage=0.5,
                            margin=MarginModel(long_initial=0.1, long_maintenance=0.1))
        assert limits.apply(np.array([0.4, 0.4])).sum() == pytest.approx(0.5)

    def test_no_shorts(self) -> None:
        w = RiskLimits(allow_short=False).apply(np.array([0.1, -0.1]))
        np.testing.assert_allclose(w, [0.1, 0.0])

    def test_margin_capacity_binds_in_cash_account(self) -> None:
        limits = RiskLimits(max_position_weight=1, max_gross_leverage=2, max_net_leverage=2, margin_buffer=0.05)
        w = limits.apply(np.array([0.6, 0.6]))
        assert limits.margin.requirement(w) == pytest.approx(0.95)
        np.testing.assert_allclose(w, [0.475, 0.475])


# --------------------------------------------------------------------------- order generation
class TestOrderGeneration:
    def test_signal_alone_does_not_trade(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_market(make_market(universe, [50.0, 20.0, 10.0]))
        assert pf.on_signal(signal("AAA", Direction.LONG)) == []
        np.testing.assert_allclose(pf.intent, [1.0, 0.0, 0.0])

    def test_rebalance_sizes_from_equity(self, universe: Universe) -> None:
        pf = make_portfolio(universe, fraction=0.1)
        pf.on_market(make_market(universe, [50.0, 20.0, 10.0]))
        pf.on_signal(signal("AAA", Direction.LONG))
        pf.on_signal(signal("BBB", Direction.SHORT, strength=0.5))
        orders = {o.symbol: o for o in pf.generate_orders(T0)}
        # AAA: 0.1 * 100k / 50 = 200 ; BBB: -0.05 * 100k / 20 = -250
        assert orders["AAA"].signed_quantity == 200
        assert orders["BBB"].signed_quantity == -250
        np.testing.assert_allclose(pf.pending, [200, -250, 0])

    def test_pending_orders_are_not_duplicated(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_market(make_market(universe, [50.0, 20.0, 10.0]))
        pf.on_signal(signal("AAA", Direction.LONG))
        assert len(pf.generate_orders(T0)) == 1
        assert pf.generate_orders(T0) == []

    def test_fill_clears_pending(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_market(make_market(universe, [50.0, 20.0, 10.0]))
        pf.on_signal(signal("AAA", Direction.LONG))
        (order,) = pf.generate_orders(T0)
        pf.on_fill(make_fill("AAA", order.side, order.quantity, 50.0, order_id=order.order_id))
        np.testing.assert_allclose(pf.pending, 0.0)
        assert pf.position("AAA").quantity == 200

    def test_rebalance_band_suppresses_small_drift(self, universe: Universe) -> None:
        pf = make_portfolio(universe, limits=RiskLimits(max_position_weight=1, rebalance_band=0.01))
        pf.on_market(make_market(universe, [50.0, 20.0, 10.0]))
        pf.on_signal(signal("AAA", Direction.LONG))
        (order,) = pf.generate_orders(T0)
        pf.on_fill(make_fill("AAA", order.side, order.quantity, 50.0, order_id=order.order_id))
        pf.on_market(make_market(universe, [50.5, 20.0, 10.0]))
        # Target trunc(0.1 * 100,100 / 50.5) = 198 -> 2-share delta (~$101) < 1% of equity.
        assert pf.generate_orders(T0) == []

    def test_exit_ignores_band(self, universe: Universe) -> None:
        pf = make_portfolio(universe, limits=RiskLimits(max_position_weight=1, rebalance_band=0.5))
        pf.on_fill(make_fill("AAA", Side.BUY, 10, 50.0))
        pf.on_market(make_market(universe, [50.0, 20.0, 10.0]))
        pf.on_signal(signal("AAA", Direction.FLAT, 0.0))
        (order,) = pf.generate_orders(T0)
        assert order.signed_quantity == -10

    def test_shorts_blocked_when_disallowed(self, universe: Universe) -> None:
        pf = make_portfolio(universe, limits=RiskLimits(allow_short=False))
        pf.on_market(make_market(universe, [50.0, 20.0, 10.0]))
        pf.on_signal(signal("AAA", Direction.SHORT))
        assert pf.intent[0] == 0.0
        assert pf.generate_orders(T0) == []

    def test_lot_rounding_truncates_toward_zero(self, universe: Universe) -> None:
        pf = make_portfolio(universe, lot_size=10)
        pf.on_market(make_market(universe, [33.0, 33.0, 10.0]))
        pf.on_signal(signal("AAA", Direction.LONG))
        pf.on_signal(signal("BBB", Direction.SHORT))
        orders = {o.symbol: o.signed_quantity for o in pf.generate_orders(T0)}
        # 10,000 / 33 = 303.03 -> 300 in lots of 10 (and symmetric for the short).
        assert orders == {"AAA": 300, "BBB": -300}

    def test_risk_reducing_orders_come_first(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_fill(make_fill("CCC", Side.BUY, 1_000, 10.0))
        pf.on_market(make_market(universe, [50.0, 20.0, 10.0]))
        pf.on_signal(signal("AAA", Direction.LONG))
        orders = pf.generate_orders(T0)  # CCC intent is 0 -> exit
        assert [o.symbol for o in orders] == ["CCC", "AAA"]

    def test_partial_fill_keeps_remainder_pending(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_market(make_market(universe, [50.0, 20.0, 10.0]))
        pf.on_signal(signal("AAA", Direction.LONG))
        (order,) = pf.generate_orders(T0)
        pf.on_fill(make_fill("AAA", Side.BUY, 50, 50.0, order_id=order.order_id))
        np.testing.assert_allclose(pf.pending, [150, 0, 0])
        assert pf.generate_orders(T0) == []
        pf.on_fill(make_fill("AAA", Side.BUY, 150, 50.0, order_id=order.order_id))
        np.testing.assert_allclose(pf.pending, 0.0)

    def test_external_fill_does_not_touch_pending(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_market(make_market(universe, [50.0, 20.0, 10.0]))
        pf.on_signal(signal("AAA", Direction.LONG))
        pf.generate_orders(T0)
        pf.on_fill(make_fill("BBB", Side.BUY, 10, 20.0, order_id=999))
        np.testing.assert_allclose(pf.pending, [200, 0, 0])

    def test_unpriced_symbol_is_not_traded(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_market(make_market(universe, [np.nan, 20.0, 10.0]))
        pf.on_signal(signal("AAA", Direction.LONG))
        assert pf.generate_orders(T0) == []

    def test_snapshot_matches_state(self, universe: Universe) -> None:
        pf = make_portfolio(universe)
        pf.on_fill(make_fill("AAA", Side.BUY, 100, 50.0))
        pf.on_fill(make_fill("BBB", Side.SELL, 100, 20.0))
        snap = pf.snapshot(T0)
        assert snap.equity == pytest.approx(pf.equity)
        assert snap.gross_exposure == pytest.approx(7_000.0)
        assert snap.net_exposure == pytest.approx(3_000.0)
        assert snap.n_positions == 2
