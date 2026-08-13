"""Purged expanding-window validation with nested threshold selection."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import DirectionForecaster, ForecasterConfig


@dataclass(frozen=True)
class WalkForwardConfig:
    """Parameters for rolling out-of-sample evaluation.

    ``embargo_bars`` leaves a gap between training labels and each test window.
    Set it at least to the target horizon to avoid a training outcome crossing
    into the test period.
    """

    min_train_size: int = 504
    test_size: int = 63
    step_size: int = 63
    embargo_bars: int = 1

    def __post_init__(self) -> None:
        if self.min_train_size < 20:
            raise ValueError("min_train_size must be at least 20.")
        if self.test_size < 1 or self.step_size < 1:
            raise ValueError("test_size and step_size must be positive.")
        if self.embargo_bars < 0:
            raise ValueError("embargo_bars cannot be negative.")


@dataclass(frozen=True)
class ThresholdSelectionConfig:
    """Nested, cost-aware selection of a probability confidence threshold.

    The threshold is selected only from an earlier validation slice inside an
    outer training window. ``reselect_every_folds`` limits repeated fitting and
    intentionally reuses the last previously selected threshold in between.
    """

    candidates: tuple[float, ...] = (0.52, 0.55, 0.58, 0.61)
    validation_fraction: float = 0.25
    min_trades: int = 20
    cost_bps: float = 7.0
    reselect_every_folds: int = 4

    def __post_init__(self) -> None:
        if not self.candidates or any(not 0.5 <= value < 1.0 for value in self.candidates):
            raise ValueError("Threshold candidates must be in [0.5, 1.0).")
        if not 0.1 <= self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in [0.1, 0.5).")
        if self.min_trades < 1:
            raise ValueError("min_trades must be positive.")
        if self.cost_bps < 0.0:
            raise ValueError("cost_bps cannot be negative.")
        if self.reselect_every_folds < 1:
            raise ValueError("reselect_every_folds must be positive.")


@dataclass(frozen=True)
class ThresholdDecision:
    """The nested validation decision attached to an outer test prediction."""

    long_threshold: float
    validation_score: float
    validation_observations: int
    source: str


class ExpandingWindowSplitter:
    """Yield expanding train and later test slices without temporal overlap."""

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    def split(self, n_samples: int) -> Iterator[tuple[slice, slice]]:
        """Produce chronological fold slices for a frame of ``n_samples`` rows."""
        config = self.config
        train_stop = config.min_train_size
        while train_stop + config.embargo_bars < n_samples:
            test_start = train_stop + config.embargo_bars
            test_stop = min(test_start + config.test_size, n_samples)
            yield slice(0, train_stop), slice(test_start, test_stop)
            if test_stop == n_samples:
                break
            train_stop += config.step_size


def probabilities_to_signals(probabilities: pd.Series, long_threshold: float) -> pd.Series:
    """Convert probabilities into symmetric long/flat/short signals."""
    if not 0.5 <= long_threshold < 1.0:
        raise ValueError("long_threshold must be in [0.5, 1.0).")
    short_threshold = 1.0 - long_threshold
    values = np.select(
        [probabilities >= long_threshold, probabilities <= short_threshold],
        [1.0, -1.0],
        default=0.0,
    )
    return pd.Series(values, index=probabilities.index, name="signal")


def walk_forward_predict(
    frame: pd.DataFrame,
    validation: WalkForwardConfig | None = None,
    forecaster: ForecasterConfig | None = None,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    threshold_selection: ThresholdSelectionConfig | None = None,
) -> pd.DataFrame:
    """Generate outer OOS forecasts, optionally with nested threshold choices.

    Model fitting, threshold selection, and the final test prediction are
    strictly time ordered. A selected threshold is a hyperparameter and is
    never evaluated on the outer window used to choose it.
    """
    if not 0.0 <= short_threshold < long_threshold <= 1.0:
        raise ValueError("Thresholds must satisfy 0 <= short < long <= 1.")
    splitter = ExpandingWindowSplitter(validation)
    config = splitter.config
    predictions: list[pd.DataFrame] = []
    last_decision = ThresholdDecision(
        long_threshold=long_threshold,
        validation_score=float("nan"),
        validation_observations=0,
        source="fixed",
    )
    for fold_number, (train_slice, test_slice) in enumerate(splitter.split(len(frame))):
        training_frame = frame.iloc[train_slice]
        if threshold_selection and fold_number % threshold_selection.reselect_every_folds == 0:
            last_decision = _select_threshold(
                training_frame,
                forecaster,
                threshold_selection,
                embargo_bars=config.embargo_bars,
                fallback=long_threshold,
            )
        model = DirectionForecaster(forecaster).fit(training_frame)
        test_frame = frame.iloc[test_slice]
        probabilities = model.predict_probabilities(test_frame)
        predicted = test_frame.copy()
        predicted["probability_up"] = probabilities
        if threshold_selection:
            predicted["signal"] = probabilities_to_signals(
                probabilities, last_decision.long_threshold
            )
        else:
            predicted["signal"] = np.select(
                [probabilities >= long_threshold, probabilities <= short_threshold],
                [1.0, -1.0],
                default=0.0,
            )
        predicted["long_threshold"] = last_decision.long_threshold
        predicted["threshold_validation_score"] = last_decision.validation_score
        predicted["threshold_validation_observations"] = last_decision.validation_observations
        predicted["threshold_source"] = last_decision.source
        predictions.append(predicted)
    if not predictions:
        raise ValueError(
            "No walk-forward fold could be created. Increase history or reduce "
            f"min_train_size ({config.min_train_size})."
        )
    result = pd.concat(predictions).sort_index()
    if result.index.has_duplicates:
        raise RuntimeError("Walk-forward configuration produced duplicate predictions.")
    return result


def _select_threshold(
    training_frame: pd.DataFrame,
    forecaster: ForecasterConfig | None,
    selection: ThresholdSelectionConfig,
    embargo_bars: int,
    fallback: float,
) -> ThresholdDecision:
    """Choose a threshold from a nested chronological validation segment."""
    split_at = int(len(training_frame) * (1.0 - selection.validation_fraction))
    validation_start = split_at + embargo_bars
    if split_at < 60 or len(training_frame) - validation_start < 20:
        return ThresholdDecision(fallback, float("nan"), 0, "fallback_insufficient_history")
    inner_train = training_frame.iloc[:split_at]
    inner_validation = training_frame.iloc[validation_start:]
    model = DirectionForecaster(forecaster).fit(inner_train)
    probabilities = model.predict_probabilities(inner_validation)
    scored: list[tuple[float, float]] = []
    for candidate in selection.candidates:
        signals = probabilities_to_signals(probabilities, candidate)
        score = _cost_adjusted_sharpe(
            signals,
            inner_validation["target_return"],
            cost_bps=selection.cost_bps,
            min_trades=selection.min_trades,
        )
        scored.append((score, candidate))
    viable = [(score, candidate) for score, candidate in scored if np.isfinite(score)]
    if not viable:
        return ThresholdDecision(
            fallback,
            float("nan"),
            len(inner_validation),
            "fallback_no_viable_threshold",
        )
    # When performance ties, the higher confidence threshold trades less and
    # is preferred. This is a deterministic rule fixed before outer testing.
    score, threshold = max(viable, key=lambda item: (item[0], item[1]))
    return ThresholdDecision(threshold, float(score), len(inner_validation), "nested_validation")


def _cost_adjusted_sharpe(
    signals: pd.Series,
    forward_returns: pd.Series,
    cost_bps: float,
    min_trades: int,
) -> float:
    trades = int((signals != 0.0).sum())
    if trades < min_trades:
        return float("nan")
    turnover = signals.diff().abs().fillna(signals.abs())
    net_returns = signals * forward_returns - turnover * cost_bps * 1e-4
    standard_deviation = float(net_returns.std(ddof=1))
    if standard_deviation == 0.0 or not np.isfinite(standard_deviation):
        return float("nan")
    return float(net_returns.mean() / standard_deviation * np.sqrt(365.0))
