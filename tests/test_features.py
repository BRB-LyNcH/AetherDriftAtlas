import numpy as np
import pandas as pd

from aetherdrift_atlas.features import FeatureBuilder


def _ohlcv(rows: int = 80) -> pd.DataFrame:
    index = pd.bdate_range("2023-01-02", periods=rows)
    close = np.linspace(100.0, 140.0, rows) + np.sin(np.arange(rows))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(10_000.0, 20_000.0, rows),
        },
        index=index,
    )


def test_target_is_one_bar_forward_return() -> None:
    raw = _ohlcv()
    features = FeatureBuilder().build(raw)
    timestamp = features.index[10]
    position = raw.index.get_loc(timestamp)
    expected = raw["close"].iloc[position + 1] / raw["close"].iloc[position] - 1.0
    assert features.loc[timestamp, "target_return"] == expected
    assert set(FeatureBuilder().feature_columns).isdisjoint({"target_return", "target_direction"})
