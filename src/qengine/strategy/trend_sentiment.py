"""Trend-following with alternative-data confirmation.

A position is only opened when the price trend *and* the smoothed news
sentiment agree; it is held while the trend persists and sentiment does not
flip against it. All decisions are computed for the whole universe at once
with boolean array algebra; Python-level iteration only touches the handful of
symbols whose view actually changed this bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qengine.events import Direction, FloatArray, MarketEvent, SignalEvent, Universe
from qengine.strategy.base import Strategy
from qengine.strategy.indicators import OnlineEWMA


@dataclass(frozen=True, slots=True)
class TrendSentimentConfig:
    entry_trend: float = 0.6
    exit_trend: float = 0.1
    sentiment_span: int = 8
    sentiment_confirm: float = 0.15
    allow_short: bool = True
    min_strength_change: float = 0.15


class TrendSentimentStrategy(Strategy):
    strategy_id = "trend_sentiment"

    def __init__(self, universe: Universe, config: TrendSentimentConfig = TrendSentimentConfig()) -> None:
        super().__init__(universe)
        self.config = config
        n = universe.n_assets
        self._sentiment = OnlineEWMA(n, config.sentiment_span)
        self._state = np.zeros(n, dtype=np.int64)
        self._strength: FloatArray = np.zeros(n)

    def on_market(self, event: MarketEvent) -> list[SignalEvent]:
        cfg = self.config
        trend = event.field("trend")
        sentiment = self._sentiment.update(event.field("sentiment"), event.available)
        ready = event.available & np.isfinite(trend) & np.isfinite(sentiment)

        prev = self._state
        long_entry = (trend > cfg.entry_trend) & (sentiment > cfg.sentiment_confirm)
        short_entry = (trend < -cfg.entry_trend) & (sentiment < -cfg.sentiment_confirm) & cfg.allow_short
        hold_long = (prev == 1) & (trend > cfg.exit_trend) & (sentiment > -cfg.sentiment_confirm)
        hold_short = (prev == -1) & (trend < -cfg.exit_trend) & (sentiment < cfg.sentiment_confirm)

        state = np.select([long_entry | hold_long, short_entry | hold_short], [1, -1], default=0)
        state = np.where(ready, state, prev)

        # Conviction blends trend magnitude with how strongly news agrees.
        agreement = np.clip(np.sign(state) * np.nan_to_num(sentiment), 0.0, 1.0)
        strength = np.where(
            state != 0, np.clip(np.tanh(np.abs(np.nan_to_num(trend))) * (0.5 + 0.5 * agreement), 0.05, 1.0), 0.0
        )

        changed = (state != prev) | (
            (state != 0) & (np.abs(strength - self._strength) >= cfg.min_strength_change)
        )
        signals = [
            SignalEvent(
                timestamp=event.timestamp,
                symbol=self.universe.symbols[i],
                direction=Direction(int(state[i])),
                strength=float(strength[i]) if state[i] != 0 else 0.0,
                strategy_id=self.strategy_id,
            )
            for i in np.flatnonzero(changed)
        ]
        self._state = state.astype(np.int64)
        self._strength = np.where(changed, strength, self._strength)
        return signals
