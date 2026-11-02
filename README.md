# qengine — event-driven, multi-asset trading engine & backtester

A modular, strictly typed (Python 3.11+, `mypy --strict` clean) backtester. Each component
talks to the others only through immutable events, so the strategy, portfolio and broker code
would run unchanged against a live feed.

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest                          # 82 tests
.venv/bin/python examples/run_backtest.py           # 8-asset synthetic market, risk parity
.venv/bin/python examples/run_backtest.py --sizer kelly
.venv/bin/pip install -e '.[yfinance]' && .venv/bin/python examples/run_backtest.py --source yfinance
```

Reports are written to `reports/<source>_<sizer>/`: `report.json`, `report.md`, `equity_curve.csv`,
`trades.csv`, `fills.csv`.

## File structure

```
src/qengine/
├── events.py                 Universe, MarketEvent, SignalEvent, OrderEvent, FillEvent, FinancingEvent
├── data/
│   ├── handler.py            DataHandler ABC; HistoricalDataHandler (builds the (T, N, F) cube, streams events)
│   ├── features.py           Vectorized indicators: returns, EMA, trend z, vol, μ, RSI, ATR, ADV
│   ├── sources.py            CSV / yfinance loaders; synthetic factor-model market with regimes
│   └── sentiment.py          Simulated NLP sentiment for real prices; CSV news-score loader
├── strategy/
│   ├── base.py               Strategy ABC
│   ├── indicators.py         Online EWMA / EW variance, vectorized across assets
│   └── trend_sentiment.py    Trend-following that requires sentiment to confirm
├── portfolio/
│   ├── portfolio.py          Portfolio: cash, positions, intent book, order generation   ← core
│   ├── accounting.py         Average-cost fill math; round-trip TradeRecord bookkeeping
│   ├── sizing.py             FixedFraction, RiskParity (ERC risk budgeting), fractional Kelly
│   ├── estimators.py         Streaming EWMA mean/covariance
│   └── risk.py               RiskLimits (caps, leverage, net) + MarginModel (Reg-T style)
├── execution/
│   ├── broker.py             ExecutionHandler ABC; SimulatedBroker (next-open fills, borrow fees)
│   └── costs.py              Per-share / bps commission; fixed-bps and square-root-impact slippage
├── performance/
│   ├── metrics.py            Sharpe, Sortino, drawdown depth/duration, profit factor, beta/alpha
│   ├── monitor.py            PerformanceMonitor: equity curve, fills, report assembly
│   └── report.py             PerformanceReport: JSON / Markdown / CSV export
└── engine/
    └── event_loop.py         EventLoop controller                                         ← core
tests/                        pytest suite (portfolio, sizing, execution, metrics, data, event loop)
examples/run_backtest.py      CLI entry point
```

## The event loop

There are two loops. The outer one pulls one `MarketEvent` per timestamp from the
`DataHandler`. The inner one drains a FIFO `deque` until no derived events are left. Within
each bar the controller runs four phases in a fixed order:

```
MarketEvent(t)
 ├─ 1. ExecutionHandler.on_market → FinancingEvent*, FillEvent*
 │      overnight borrow fees on shorts held since t-1 (Actual/360),
 │      then orders submitted at t-1 fill at bar t's OPEN (+ slippage, commission)
 │      └─ drain → Portfolio.on_fill / on_financing, Monitor.on_fill
 ├─ 2. Portfolio.on_market → mark to bar t's CLOSE; sizer covariance update
 │      └─ Monitor.record(snapshot)            (one equity-curve point per bar)
 ├─ 3. Strategy.on_market → SignalEvent*
 │      └─ drain → Portfolio.on_signal          (updates the intent book only)
 └─ 4. Portfolio.generate_orders → OrderEvent*
        size → risk limits → margin → diff vs (position + in-flight) → band filter
        └─ drain → ExecutionHandler.on_order    (orders rest until bar t+1)
```

A signal formed on bar `t`'s close can only trade at bar `t+1`'s open, so look-ahead is ruled
out by the structure itself. `tests/test_event_loop.py` asserts this, and
`tests/test_data.py` checks that rewriting future prices changes no past feature.

### Design decisions

* **The data plane is vectorized; the control plane is event-driven.** All indicators are
  computed once, column-wise with pandas/NumPy, and packed into a read-only `(T, N, F)` cube.
  Each `MarketEvent` is a zero-copy `(N, F)` view of it, so strategies also work on whole
  columns (`event.field("trend")`) instead of looping over symbols.
* **Signals set an intent; they don't place orders.** The portfolio keeps an intent book
  (signed conviction per asset) and rebalances it in one pass. Cross-sectional sizers need
  this: risk parity and Kelly have to see the whole book at once.
* **Target positions, not incremental orders.** Orders are `target − (position + pending)`, so
  repeated signals can't stack, and partial or unfilled orders are counted. Pending is tracked
  per `order_id`, so fills the portfolio didn't originate don't distort it.
* **The broker keeps its own ledger** and charges borrow fees from it, the way a prime broker
  does. The tests check that the broker's ledger and the portfolio's books agree.

## Position sizing

| Sizer | Model |
|---|---|
| `RiskParitySizer` | Risk budgeting (each asset's contribution ∝ \|conviction\|), solved by cyclical coordinate descent and scaled to a portfolio vol target. Shorts are handled by solving on `DΣD`, where `D = diag(sign)`. |
| `KellySizer` | Fractional multivariate Kelly `w = f·Σ⁻¹(μ − r_f)` on a diagonally shrunk covariance. `r_f` is the live rate streamed in each event. It never takes a position against the signal. |
| `FixedFractionSizer` | `w = conviction × fraction`. Used as a baseline and in tests. |

`RiskLimits.apply` then enforces, in order: no-short, per-name cap, gross leverage, net
leverage, and initial-margin capacity (`1 − margin_buffer`). Lots are truncated toward zero,
so rounding can never push a position past a limit. If equity falls below maintenance margin,
the book is cut down proportionally. If equity reaches zero or below, everything is
liquidated and trading halts.

## Analytics

| Section | Metrics |
|---|---|
| Risk-adjusted | Sharpe and Sortino on excess returns over a **time-varying** risk-free series (synthetic hiking cycle or `^IRX`), Calmar |
| Drawdown | Max depth; peak, trough and recovery dates; longest duration in bars and calendar days (vectorized) |
| Trades | Round trips net of commission and borrow: profit factor, win/loss ratio, win rate, expectancy, holding period |
| Benchmark | CAPM beta and alpha on excess returns, correlation, tracking error, information ratio (synthetic market factor or `^GSPC`) |
| Costs & exposure | Commission, slippage, borrow fees, annual turnover, gross and net leverage |

## Notes and limitations

* The synthetic market's sentiment is a noisy reading of each asset's hidden drift regime. It
  is informative on purpose, which makes the example's strong Sharpe ratio a demonstration of
  the plumbing, not evidence of an edge. For real prices, `simulate_nlp_sentiment` builds
  scores from trailing returns only.
* yfinance loaders are optional and are not exercised by the test suite (no network in CI).
  To add Alpaca or Binance, write a loader that returns `dict[symbol, OHLCV DataFrame]`.
* Only market orders are supported, and they fill in full at the next open; impact is
  modelled with the square-root law. Limit orders and volume-capped partial fills are natural
  extensions of `SimulatedBroker._match`.
* Because orders fill at the next open, prices can move between rebalances and push realized
  gross leverage slightly above `max_gross_leverage` (≈1.57 vs 1.5 in the Kelly example). The
  margin-call path is the hard backstop.
