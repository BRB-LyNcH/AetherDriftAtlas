"""Probabilistic next-bar direction forecasting and accuracy diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .features import FEATURE_COLUMNS


@dataclass(frozen=True)
class ForecasterConfig:
    """Conservative defaults for a nonlinear tabular directional model."""

    n_estimators: int = 300
    max_depth: int | None = 5
    min_samples_leaf: int = 8
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_estimators < 10:
            raise ValueError("n_estimators must be at least 10.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be positive.")


@dataclass(frozen=True)
class ForecastMetrics:
    """Out-of-sample classification calibration and directional accuracy."""

    direction_accuracy: float
    balanced_accuracy: float
    long_precision: float
    long_recall: float
    brier_score: float
    roc_auc: float
    observations: int

    def as_dict(self) -> dict[str, float | int]:
        """Convert metrics to JSON-safe primitive values."""
        return {
            "direction_accuracy": self.direction_accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "long_precision": self.long_precision,
            "long_recall": self.long_recall,
            "brier_score": self.brier_score,
            "roc_auc": self.roc_auc,
            "observations": self.observations,
        }


class DirectionForecaster:
    """Random-forest probability model trained only on prior observations."""

    def __init__(self, config: ForecasterConfig | None = None) -> None:
        self.config = config or ForecasterConfig()
        self._model: RandomForestClassifier | None = None

    def fit(self, training_frame: pd.DataFrame) -> DirectionForecaster:
        """Fit the model on a labeled historical window."""
        self._validate_frame(training_frame, require_target=True)
        labels = training_frame["target_direction"].astype(int)
        if labels.nunique() < 2:
            raise ValueError("Training labels need both upward and downward observations.")
        config = self.config
        self._model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.random_state,
        )
        self._model.fit(training_frame.loc[:, FEATURE_COLUMNS], labels)
        return self

    def predict_probabilities(self, frame: pd.DataFrame) -> pd.Series:
        """Return the probability that the next bar closes higher."""
        self._validate_frame(frame, require_target=False)
        if self._model is None:
            raise RuntimeError("Call fit before requesting predictions.")
        probability_matrix = self._model.predict_proba(frame.loc[:, FEATURE_COLUMNS])
        classes = self._model.classes_
        up_index = int(np.where(classes == 1)[0][0])
        return pd.Series(probability_matrix[:, up_index], index=frame.index, name="probability_up")

    def predict_frame(
        self,
        frame: pd.DataFrame,
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
    ) -> pd.DataFrame:
        """Attach probabilities and tradable long/flat/short signals to a frame."""
        if not 0.0 <= short_threshold < long_threshold <= 1.0:
            raise ValueError("Thresholds must satisfy 0 <= short < long <= 1.")
        probabilities = self.predict_probabilities(frame)
        signals = np.select(
            [probabilities >= long_threshold, probabilities <= short_threshold],
            [1.0, -1.0],
            default=0.0,
        )
        result = frame.copy()
        result["probability_up"] = probabilities
        result["signal"] = signals
        return result

    @staticmethod
    def evaluate(predictions: pd.DataFrame) -> ForecastMetrics:
        """Measure stock/asset prediction quality on strictly out-of-sample rows."""
        required = {"target_direction", "probability_up"}
        missing = required.difference(predictions.columns)
        if missing:
            raise ValueError(f"predictions is missing: {sorted(missing)}")
        truth = predictions["target_direction"].astype(int)
        probabilities = predictions["probability_up"].astype(float).clip(0.0, 1.0)
        labels = (probabilities >= 0.5).astype(int)
        roc_auc = (
            float(roc_auc_score(truth, probabilities))
            if truth.nunique() == 2
            else float("nan")
        )
        return ForecastMetrics(
            direction_accuracy=float(accuracy_score(truth, labels)),
            balanced_accuracy=float(balanced_accuracy_score(truth, labels)),
            long_precision=float(precision_score(truth, labels, zero_division=0)),
            long_recall=float(recall_score(truth, labels, zero_division=0)),
            brier_score=float(brier_score_loss(truth, probabilities)),
            roc_auc=roc_auc,
            observations=len(predictions),
        )

    @staticmethod
    def _validate_frame(frame: pd.DataFrame, require_target: bool) -> None:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("A non-empty feature DataFrame is required.")
        required = set(FEATURE_COLUMNS)
        if require_target:
            required.add("target_direction")
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Feature frame is missing: {sorted(missing)}")
        if not np.isfinite(frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
            raise ValueError("Feature frame contains non-finite predictor values.")
