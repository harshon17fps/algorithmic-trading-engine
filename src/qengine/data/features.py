"""Vectorized feature engineering over wide (time x symbol) panels.

All indicators are computed once, column-wise across every asset, before a
single event is streamed. Every transform uses only trailing information
(``shift``, ``rolling``, ``ewm``) so the cube contains no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, cast

import numpy as np
import pandas as pd

Panel = Mapping[str, pd.DataFrame]


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    ema_fast: int = 20
    ema_slow: int = 100
    vol_window: int = 20
    mu_halflife: int = 60
    rsi_window: int = 14
    atr_window: int = 14
    adv_window: int = 20
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if self.ema_fast >= self.ema_slow:
            raise ValueError("ema_fast must be shorter than ema_slow")


FEATURE_FIELDS: tuple[str, ...] = (
    "ret",
    "log_ret",
    "ema_fast",
    "ema_slow",
    "trend",
    "vol",
    "mu",
    "rsi",
    "atr",
    "adv",
)


def compute_features(panel: Panel, cfg: FeatureConfig) -> dict[str, pd.DataFrame]:
    """Return a dict of wide DataFrames, one per feature in ``FEATURE_FIELDS``."""
    close = panel["close"]
    high = panel["high"]
    low = panel["low"]
    volume = panel["volume"]
    prev_close = close.shift(1)
    ann = np.sqrt(cfg.periods_per_year)

    ret = close / prev_close - 1.0
    log_ret = cast(pd.DataFrame, np.log(close / prev_close))

    ema_fast = close.ewm(span=cfg.ema_fast, adjust=False, min_periods=cfg.ema_fast).mean()
    ema_slow = close.ewm(span=cfg.ema_slow, adjust=False, min_periods=cfg.ema_slow).mean()

    daily_vol = log_ret.rolling(cfg.vol_window, min_periods=cfg.vol_window).std(ddof=1)
    vol = daily_vol * ann

    # Dimensionless trend strength: EMA spread scaled by the noise expected over
    # the slow window, so one threshold works across assets of different vols.
    trend = cast(pd.DataFrame, np.log(ema_fast / ema_slow)) / (daily_vol * np.sqrt(cfg.ema_slow))

    mu = log_ret.ewm(halflife=cfg.mu_halflife, min_periods=cfg.mu_halflife).mean() * cfg.periods_per_year

    delta = close.diff()
    alpha_rsi = 1.0 / cfg.rsi_window
    avg_gain = delta.clip(lower=0.0).ewm(alpha=alpha_rsi, adjust=False, min_periods=cfg.rsi_window).mean()
    avg_loss = (-delta.clip(upper=0.0)).ewm(alpha=alpha_rsi, adjust=False, min_periods=cfg.rsi_window).mean()
    rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    true_range = np.fmax(
        np.fmax((high - low).to_numpy(), (high - prev_close).abs().to_numpy()),
        (low - prev_close).abs().to_numpy(),
    )
    true_range[prev_close.isna().to_numpy()] = np.nan
    atr = (
        pd.DataFrame(true_range, index=close.index, columns=close.columns)
        .ewm(alpha=1.0 / cfg.atr_window, adjust=False, min_periods=cfg.atr_window)
        .mean()
    )

    adv = (close * volume).rolling(cfg.adv_window, min_periods=cfg.adv_window).mean()

    out = {
        "ret": ret,
        "log_ret": log_ret,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "trend": trend,
        "vol": vol,
        "mu": mu,
        "rsi": rsi,
        "atr": atr,
        "adv": adv,
    }
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in out.items()}
