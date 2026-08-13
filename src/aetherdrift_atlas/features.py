"""Point-in-time technical features and forward-return labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "return_1d",
    "return_5d",
    "sma_ratio_10",
    "sma_ratio_30",
    "volatility_20d",
    "rsi_14",
    "range_pct",
    "volume_zscore_20",
)


@dataclass(frozen=True)
class FeatureConfig:
    """Feature windows and label horizon for a daily-bar forecasting problem."""

    target_horizon: int = 1
    short_window: int = 10
    long_window: int = 30
    volatility_window: int = 20
    rsi_window: int = 14

    def __post_init__(self) -> None:
        if self.target_horizon < 1:
            raise ValueError("target_horizon must be at least one.")
        if self.short_window < 2 or self.long_window <= self.short_window:
            raise ValueError("long_window must be greater than short_window >= 2.")


class FeatureBuilder:
    """Create only features observable at each bar's decision timestamp."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Columns emitted as model predictors."""
        return FEATURE_COLUMNS

    def build(self, ohlcv: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """Build a finite feature table, optionally with the future-return label.

        Every predictor uses data through timestamp *t* only. ``target_return``
        is explicitly shifted into the future and never included in predictors.
        """
        self._validate_ohlcv(ohlcv)
        frame = ohlcv.copy().sort_index()
        close = pd.to_numeric(frame["close"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
        config = self.config

        result = pd.DataFrame(index=frame.index)
        result["return_1d"] = close.pct_change(fill_method=None)
        result["return_5d"] = close.pct_change(5, fill_method=None)
        result["sma_ratio_10"] = close / close.rolling(config.short_window).mean() - 1.0
        result["sma_ratio_30"] = close / close.rolling(config.long_window).mean() - 1.0
        result["volatility_20d"] = result["return_1d"].rolling(
            config.volatility_window
        ).std(ddof=0)
        result["rsi_14"] = self._rsi(close, config.rsi_window)
        result["range_pct"] = (frame["high"] - frame["low"]) / close
        rolling_volume = volume.rolling(config.volatility_window)
        volume_std = rolling_volume.std(ddof=0).replace(0.0, np.nan)
        result["volume_zscore_20"] = (volume - rolling_volume.mean()) / volume_std
        # An absent/uninformative volume feed (normal for some FX series) is
        # represented by a neutral feature rather than discarded data.
        result["volume_zscore_20"] = result["volume_zscore_20"].fillna(0.0)

        if include_target:
            result["target_return"] = close.shift(-config.target_horizon) / close - 1.0
            result["target_direction"] = (result["target_return"] > 0.0).astype(float)

        result = result.replace([np.inf, -np.inf], np.nan)
        required = [*self.feature_columns]
        if include_target:
            required.extend(["target_return", "target_direction"])
        result = result.dropna(subset=required)
        if result.empty:
            raise ValueError("No usable feature rows remain; provide more valid history.")
        return result

    @staticmethod
    def _rsi(close: pd.Series, window: int) -> pd.Series:
        delta = close.diff()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)
        average_gain = gains.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
        average_loss = losses.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
        relative_strength = average_gain / average_loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        return rsi.mask((average_loss == 0.0) & average_gain.notna(), 100.0)

    @staticmethod
    def _validate_ohlcv(frame: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "volume"}
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("ohlcv must be a non-empty DataFrame.")
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"ohlcv is missing required columns: {sorted(missing)}")
        if not frame.index.is_monotonic_increasing:
            # Sorting is safe, but duplicate timestamps are not: their order
            # would make feature values depend on provider ordering.
            frame = frame.sort_index()
        if frame.index.has_duplicates:
            raise ValueError("ohlcv index cannot contain duplicate timestamps.")
