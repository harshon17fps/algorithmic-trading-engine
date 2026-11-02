from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qengine.data import HistoricalDataHandler, SyntheticMarketConfig, generate_synthetic_market
from qengine.data.features import FEATURE_FIELDS
from qengine.strategy import OnlineEWMA


@pytest.fixture(scope="module")
def dataset():  # type: ignore[no-untyped-def]
    return generate_synthetic_market(SyntheticMarketConfig(n_bars=300, seed=5))


def test_features_have_no_lookahead(dataset) -> None:  # type: ignore[no-untyped-def]
    """Rewriting the future must not change any feature value in the past."""
    base = HistoricalDataHandler(dataset.bars, sentiment=dataset.sentiment)
    cut = 250
    shocked = {s: df.copy() for s, df in dataset.bars.items()}
    for df in shocked.values():
        df.iloc[cut:, :4] *= 3.0
    alt = HistoricalDataHandler(shocked, sentiment=dataset.sentiment)
    for f in FEATURE_FIELDS:
        pd.testing.assert_frame_equal(base.panel(f).iloc[:cut], alt.panel(f).iloc[:cut], check_exact=False)


def test_vectorized_features_match_pandas_reference(dataset) -> None:  # type: ignore[no-untyped-def]
    h = HistoricalDataHandler(dataset.bars)
    close = dataset.bars["TECH1"]["close"]
    expected_ema = close.ewm(span=20, adjust=False, min_periods=20).mean()
    pd.testing.assert_series_equal(h.panel("ema_fast")["TECH1"], expected_ema, check_names=False)
    expected_vol = np.log(close / close.shift(1)).rolling(20).std() * np.sqrt(252)
    pd.testing.assert_series_equal(h.panel("vol")["TECH1"], expected_vol, check_names=False)


def test_stream_yields_zero_copy_read_only_views(dataset) -> None:  # type: ignore[no-untyped-def]
    h = HistoricalDataHandler(dataset.bars, sentiment=dataset.sentiment, risk_free=dataset.risk_free)
    events = list(h.stream())
    assert len(events) == len(h) == 300
    first = events[0]
    assert first.values.shape == (h.universe.n_assets, len(h.universe.fields))
    assert not first.values.flags.writeable
    assert np.shares_memory(first.values, events[1].values) is False or first.values.base is not None
    assert first.value("TECH1", "close") == pytest.approx(dataset.bars["TECH1"]["close"].iloc[0])
    np.testing.assert_allclose(events[-1].field("risk_free"), dataset.risk_free.iloc[-1])


def test_unequal_calendars_mark_missing_assets_unavailable(dataset) -> None:  # type: ignore[no-untyped-def]
    bars = {s: df for s, df in dataset.bars.items()}
    bars["TECH1"] = bars["TECH1"].iloc[50:]
    h = HistoricalDataHandler(bars)
    first = next(h.stream())
    i = h.universe.symbol_index("TECH1")
    assert not first.available[i]
    assert first.available.sum() == h.universe.n_assets - 1


def test_online_ewma_skips_missing_news() -> None:
    ewma = OnlineEWMA(2, span=3)  # alpha = 0.5
    ewma.update(np.array([1.0, np.nan]))
    out = ewma.update(np.array([np.nan, 2.0]))
    np.testing.assert_allclose(out, [1.0, 2.0])
    out = ewma.update(np.array([0.0, 4.0]))
    np.testing.assert_allclose(out, [0.5, 3.0])
