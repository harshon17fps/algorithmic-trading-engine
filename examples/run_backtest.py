"""End-to-end backtest: 8-asset synthetic market (or yfinance), trend + sentiment,
risk-parity or Kelly sizing, realistic costs, full analytics export.

    python examples/run_backtest.py                       # synthetic, risk parity
    python examples/run_backtest.py --sizer kelly
    python examples/run_backtest.py --source yfinance     # needs: pip install 'qengine[yfinance]'
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qengine import EventLoop
from qengine.data import (
    FeatureConfig,
    HistoricalDataHandler,
    SyntheticMarketConfig,
    generate_synthetic_market,
    load_yfinance_bars,
    load_yfinance_close,
    load_yfinance_risk_free,
    simulate_nlp_sentiment,
)
from qengine.execution import PerShareCommission, SimulatedBroker, SquareRootImpact
from qengine.performance import PerformanceMonitor
from qengine.portfolio import KellySizer, MarginModel, Portfolio, PositionSizer, RiskLimits, RiskParitySizer
from qengine.strategy import TrendSentimentConfig, TrendSentimentStrategy

YF_UNIVERSE = ("AAPL", "MSFT", "NVDA", "XOM", "CVX", "JPM", "GLD", "TLT")


def build_data(source: str, start: str) -> HistoricalDataHandler:
    features = FeatureConfig(ema_fast=20, ema_slow=100)
    if source == "synthetic":
        ds = generate_synthetic_market(SyntheticMarketConfig())
        return HistoricalDataHandler(
            ds.bars, sentiment=ds.sentiment, benchmark=ds.benchmark, risk_free=ds.risk_free, features=features
        )
    bars = load_yfinance_bars(YF_UNIVERSE, start=start)
    close = HistoricalDataHandler(bars).panel("close")
    return HistoricalDataHandler(
        bars,
        sentiment=simulate_nlp_sentiment(close, seed=7),
        benchmark=load_yfinance_close("^GSPC", start=start),
        risk_free=load_yfinance_risk_free(start=start),
        features=features,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("synthetic", "yfinance"), default="synthetic")
    parser.add_argument("--sizer", choices=("risk_parity", "kelly"), default="risk_parity")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    data = build_data(args.source, args.start)
    universe, ppy = data.universe, data.periods_per_year
    sizer: PositionSizer = (
        RiskParitySizer(universe.n_assets, target_vol=0.12, periods_per_year=ppy)
        if args.sizer == "risk_parity"
        else KellySizer(universe.n_assets, fraction=0.3, periods_per_year=ppy)
    )
    limits = RiskLimits(
        max_position_weight=0.30,
        max_gross_leverage=1.5,
        max_net_leverage=1.0,
        rebalance_band=0.02,
        margin=MarginModel(long_initial=0.5, short_initial=0.5),  # Reg-T
    )
    loop = EventLoop(
        data=data,
        strategy=TrendSentimentStrategy(universe, TrendSentimentConfig()),
        portfolio=Portfolio(universe, args.capital, sizer, limits),
        execution=SimulatedBroker(
            universe,
            commission=PerShareCommission(),
            slippage=SquareRootImpact(spread_bps=4.0, impact_coef=0.5, periods_per_year=ppy),
            borrow_rate=0.01,
        ),
        monitor=PerformanceMonitor(ppy),
    )
    result = loop.run()

    print(result.report.to_markdown(f"{args.source} | {args.sizer}"))
    counts = ", ".join(f"{k.name}={v}" for k, v in result.event_counts.items())
    print(f"Events: {counts}  |  {result.wall_time_s:.2f}s  ({result.events_per_second:,.0f} events/s)")
    for kind, path in result.report.export(args.out / f"{args.source}_{args.sizer}").items():
        print(f"  {kind:>13}: {path}")


if __name__ == "__main__":
    main()
