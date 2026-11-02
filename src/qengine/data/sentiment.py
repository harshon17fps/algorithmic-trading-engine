"""Alternative data: news / NLP sentiment scores aligned to the bar grid.

Output convention: a wide ``time x symbol`` DataFrame of scores in [-1, 1],
with NaN meaning "no news published for this symbol in this bar". The
strategy's online indicators simply skip NaNs, so stale news decays naturally.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


def simulate_nlp_sentiment(
    close: pd.DataFrame,
    *,
    lookback: int = 10,
    sensitivity: float = 0.8,
    noise: float = 0.7,
    news_probability: float = 0.6,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate headline sentiment for *real* price data (e.g. from yfinance).

    Scores react to trailing risk-adjusted performance (news flow tends to
    follow price) plus idiosyncratic noise, and only exist on "news days".
    Built purely from trailing windows, so it introduces no look-ahead.
    """
    rng = np.random.default_rng(seed)
    log_ret = cast(pd.DataFrame, np.log(close / close.shift(1)))
    trailing = log_ret.rolling(lookback, min_periods=lookback).sum()
    scale = log_ret.rolling(60, min_periods=20).std() * np.sqrt(lookback)
    z = (trailing / scale).to_numpy()
    score = np.tanh(sensitivity * z + noise * rng.standard_normal(z.shape))
    has_news = rng.random(z.shape) < news_probability
    return pd.DataFrame(np.where(has_news, score, np.nan), index=close.index, columns=close.columns)


def load_sentiment_csv(
    path: str | Path,
    *,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    score_col: str = "score",
    publication_lag: pd.Timedelta = pd.Timedelta(0),
    bar_frequency: str = "D",
) -> pd.DataFrame:
    """Load long-format scores (one row per article) and aggregate per bar.

    ``publication_lag`` shifts each score to when it was actually knowable
    (e.g. NLP pipeline latency), *then* timestamps are floored to the bar grid,
    so an article published after a bar's close lands on the next bar.
    """
    raw = pd.read_csv(path)
    ts = pd.to_datetime(raw[timestamp_col]) + publication_lag
    raw = raw.assign(_bar=ts.dt.floor(bar_frequency))
    wide = raw.pivot_table(index="_bar", columns=symbol_col, values=score_col, aggfunc="mean")
    wide.index.name = None
    wide.columns.name = None
    return wide.clip(-1.0, 1.0).sort_index()
