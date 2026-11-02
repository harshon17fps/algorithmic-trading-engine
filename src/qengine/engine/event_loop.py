"""EventLoop: the controller that wires every component together via a queue.

Outer loop — one iteration per MarketEvent pulled from the DataHandler.
Inner loop — drain the FIFO queue until no derived events remain.

Within each bar the controller enforces a strict, causal phase order::

    MarketEvent(t)
      ├─ 1. ExecutionHandler.on_market   → FinancingEvent*, FillEvent*   (orders from t-1 fill at t's OPEN)
      │     └─ drain: Portfolio.on_fill / on_financing, Monitor.on_fill
      ├─ 2. Portfolio.on_market          → mark-to-market at t's CLOSE; sizer estimators update
      │     └─ Monitor.record(snapshot)  → equity curve point for bar t
      ├─ 3. Strategy.on_market           → SignalEvent*
      │     └─ drain: Portfolio.on_signal (updates intent book)
      └─ 4. Portfolio.generate_orders    → OrderEvent*
            └─ drain: ExecutionHandler.on_order (rest until t+1)

Because a signal formed on bar t's close can only trade at bar t+1's open,
look-ahead bias is impossible by construction — the same guarantee a live
system gets from wall-clock time.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable

from qengine.data.handler import DataHandler
from qengine.events import (
    Event,
    EventType,
    FillEvent,
    FinancingEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from qengine.execution.broker import ExecutionHandler
from qengine.performance.monitor import PerformanceMonitor
from qengine.performance.report import PerformanceReport
from qengine.portfolio.portfolio import Portfolio
from qengine.strategy.base import Strategy


@dataclass(frozen=True, slots=True)
class BacktestResult:
    report: PerformanceReport
    portfolio: Portfolio
    event_counts: dict[EventType, int] = field(default_factory=dict)
    wall_time_s: float = 0.0

    @property
    def events_per_second(self) -> float:
        total = sum(self.event_counts.values())
        return total / self.wall_time_s if self.wall_time_s > 0 else float("inf")


class EventLoop:
    def __init__(
        self,
        data: DataHandler,
        strategy: Strategy,
        portfolio: Portfolio,
        execution: ExecutionHandler,
        monitor: PerformanceMonitor | None = None,
        *,
        max_events_per_bar: int = 100_000,
    ) -> None:
        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution
        self.monitor = monitor or PerformanceMonitor(data.periods_per_year)
        self._queue: deque[Event] = deque()
        self._counts: Counter[EventType] = Counter()
        self._max_events_per_bar = max_events_per_bar

    # ------------------------------------------------------------------ public
    def run(self) -> BacktestResult:
        started = time.perf_counter()
        for market in self.data.stream():
            self.on_market(market)
        elapsed = time.perf_counter() - started
        report = self.monitor.build_report(
            self.portfolio.trades(),
            benchmark=self.data.benchmark(),
            risk_free=self.data.risk_free(),
        )
        return BacktestResult(
            report=report,
            portfolio=self.portfolio,
            event_counts=dict(self._counts),
            wall_time_s=elapsed,
        )

    def on_market(self, event: MarketEvent) -> None:
        """Process one bar end-to-end. Public so a live feed can push bars in."""
        self._counts[EventType.MARKET] += 1
        self._publish(self.execution.on_market(event))
        self._drain()
        self.portfolio.on_market(event)
        self.monitor.record(self.portfolio.snapshot(event.timestamp))
        self._publish(self.strategy.on_market(event))
        self._drain()
        self._publish(self.portfolio.generate_orders(event.timestamp))
        self._drain()

    # ---------------------------------------------------------------- internal
    def _publish(self, events: Iterable[Event]) -> None:
        self._queue.extend(events)

    def _drain(self) -> None:
        processed = 0
        while self._queue:
            event = self._queue.popleft()
            self._counts[event.type] += 1
            self._publish(self._dispatch(event))
            processed += 1
            if processed > self._max_events_per_bar:
                raise RuntimeError("Event storm: handlers keep generating events within one bar")

    def _dispatch(self, event: Event) -> Iterable[Event]:
        match event:
            case SignalEvent():
                return self.portfolio.on_signal(event)
            case OrderEvent():
                return self.execution.on_order(event)
            case FillEvent():
                self.portfolio.on_fill(event)
                self.monitor.on_fill(event)
                return ()
            case FinancingEvent():
                self.portfolio.on_financing(event)
                self.monitor.on_financing(event)
                return ()
            case MarketEvent():
                raise RuntimeError("MarketEvents must enter through EventLoop.on_market")
        raise TypeError(f"Unhandled event type: {type(event).__name__}")
