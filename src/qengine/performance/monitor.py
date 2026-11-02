"""PerformanceMonitor: records the equity curve and turns it into analytics."""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np
import pandas as pd

from qengine.events import FillEvent, FinancingEvent
from qengine.performance.metrics import (
    annualized_return,
    annualized_volatility,
    benchmark_stats,
    drawdown_series,
    drawdown_stats,
    sharpe_ratio,
    simple_returns,
    sortino_ratio,
    to_period_rate,
    trade_stats,
)
from qengine.performance.report import MetricValue, PerformanceReport
from qengine.portfolio.accounting import TradeRecord
from qengine.portfolio.portfolio import PortfolioSnapshot


class PerformanceMonitor:
    def __init__(self, periods_per_year: int = 252) -> None:
        self.periods_per_year = periods_per_year
        self._snapshots: list[PortfolioSnapshot] = []
        self._fills: list[FillEvent] = []
        self._financing: list[FinancingEvent] = []

    def record(self, snapshot: PortfolioSnapshot) -> None:
        self._snapshots.append(snapshot)

    def on_fill(self, fill: FillEvent) -> None:
        self._fills.append(fill)

    def on_financing(self, event: FinancingEvent) -> None:
        self._financing.append(event)

    def equity_curve(self) -> pd.DataFrame:
        if not self._snapshots:
            return pd.DataFrame()
        df = pd.DataFrame([asdict(s) for s in self._snapshots]).set_index("timestamp")
        df["returns"] = df["equity"].pct_change()
        df["drawdown"] = drawdown_series(df["equity"])
        df["gross_leverage"] = df["gross_exposure"] / df["equity"]
        df["net_leverage"] = df["net_exposure"] / df["equity"]
        return df

    def fills_frame(self) -> pd.DataFrame:
        if not self._fills:
            return pd.DataFrame(columns=["timestamp", "symbol", "side", "quantity", "fill_price",
                                         "commission", "slippage_cost", "order_id"])
        rows = [
            {"timestamp": f.timestamp, "symbol": f.symbol, "side": f.side.name, "quantity": f.quantity,
             "fill_price": f.fill_price, "commission": f.commission, "slippage_cost": f.slippage_cost,
             "order_id": f.order_id}
            for f in self._fills
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def trades_frame(trades: Sequence[TradeRecord]) -> pd.DataFrame:
        rows = [
            {"symbol": t.symbol, "direction": t.direction.name, "entry_time": t.entry_time,
             "exit_time": t.exit_time, "entry_price": t.entry_price, "exit_price": t.exit_price,
             "max_quantity": t.max_quantity, "gross_pnl": t.gross_pnl, "costs": t.costs,
             "net_pnl": t.net_pnl, "is_open": t.is_open}
            for t in trades
        ]
        return pd.DataFrame(rows)

    def build_report(
        self,
        trades: Sequence[TradeRecord],
        *,
        benchmark: pd.Series | None = None,
        risk_free: pd.Series | None = None,
    ) -> PerformanceReport:
        curve = self.equity_curve()
        if curve.empty:
            raise ValueError("No equity snapshots recorded; did the backtest run?")
        ppy = self.periods_per_year
        equity = curve["equity"]
        rets = simple_returns(equity)
        rf_annual = (
            risk_free.reindex(equity.index).ffill().fillna(0.0)
            if risk_free is not None
            else pd.Series(0.0, index=equity.index)
        )
        rf_period = to_period_rate(rf_annual, ppy)
        assert isinstance(rf_period, pd.Series)

        years = len(rets) / ppy
        dd = drawdown_stats(equity)
        closed = [t for t in trades if not t.is_open]
        ts = trade_stats([t.net_pnl for t in closed])
        holding = [t.holding_period / pd.Timedelta(days=1) for t in closed if t.holding_period is not None]
        fills = self.fills_frame()
        traded_notional = float((fills["quantity"] * fills["fill_price"]).sum()) if len(fills) else 0.0
        cagr = annualized_return(equity, ppy)

        sections: dict[str, dict[str, MetricValue]] = {
            "Summary": {
                "start": str(equity.index[0].date()),
                "end": str(equity.index[-1].date()),
                "bars": int(len(equity)),
                "initial_equity": float(equity.iloc[0]),
                "final_equity": float(equity.iloc[-1]),
                "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
                "cagr": cagr,
                "annual_volatility": annualized_volatility(rets, ppy),
            },
            "Risk-Adjusted (dynamic risk-free)": {
                "sharpe_ratio": sharpe_ratio(rets, rf_period, ppy),
                "sortino_ratio": sortino_ratio(rets, rf_period, ppy),
                "calmar_ratio": cagr / abs(dd.max_drawdown) if dd.max_drawdown < 0 else float("nan"),
                "avg_risk_free_rate": float(rf_annual.mean()),
            },
            "Drawdown": {
                "max_drawdown": dd.max_drawdown,
                "drawdown_peak": str(dd.peak.date()),
                "drawdown_trough": str(dd.trough.date()),
                "drawdown_recovery": str(dd.recovery.date()) if dd.recovery is not None else "not recovered",
                "max_drawdown_duration_bars": dd.max_duration_bars,
                "max_drawdown_duration_days": dd.max_duration_days,
                "longest_drawdown_start": str(dd.longest_start.date()),
                "longest_drawdown_end": str(dd.longest_end.date()),
                "current_drawdown": dd.current_drawdown,
            },
            "Trades (closed round trips, net of costs)": {
                "n_trades": ts.n_trades,
                "open_trades": len(trades) - len(closed),
                "win_rate": ts.win_rate,
                "profit_factor": ts.profit_factor,
                "win_loss_ratio": ts.win_loss_ratio,
                "avg_win": ts.avg_win,
                "avg_loss": ts.avg_loss,
                "expectancy": ts.expectancy,
                "largest_win": ts.largest_win,
                "largest_loss": ts.largest_loss,
                "avg_holding_days": float(np.mean(holding)) if holding else float("nan"),
            },
        }

        if benchmark is not None:
            bench = benchmark.reindex(equity.index).ffill()
            bench_rets = simple_returns(bench)
            bs = benchmark_stats(rets, bench_rets, rf_period, ppy)
            sections["Benchmark"] = {
                "beta": bs.beta,
                "alpha": bs.alpha,
                "correlation": bs.correlation,
                "tracking_error": bs.tracking_error,
                "information_ratio": bs.information_ratio,
                "benchmark_total_return": float(bench.iloc[-1] / bench.dropna().iloc[0] - 1.0),
                "benchmark_sharpe": sharpe_ratio(bench_rets.dropna(), rf_period, ppy),
            }

        sections["Costs & Exposure"] = {
            "total_commission": float(fills["commission"].sum()) if len(fills) else 0.0,
            "total_slippage": float(fills["slippage_cost"].sum()) if len(fills) else 0.0,
            "total_borrow_fees": float(sum(f.amount for f in self._financing)),
            "annual_turnover": traded_notional / float(equity.mean()) / years if years > 0 else float("nan"),
            "avg_gross_leverage": float(curve["gross_leverage"].mean()),
            "max_gross_leverage": float(curve["gross_leverage"].max()),
            "avg_net_leverage": float(curve["net_leverage"].mean()),
            "n_fills": int(len(fills)),
        }

        return PerformanceReport(
            metrics=sections,
            equity_curve=curve,
            trades=self.trades_frame(trades),
            fills=fills,
        )
