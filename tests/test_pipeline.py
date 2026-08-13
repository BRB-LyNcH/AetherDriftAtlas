from aetherdrift_atlas.demo import make_demo_ohlcv
from aetherdrift_atlas.features import FeatureBuilder
from aetherdrift_atlas.model import DirectionForecaster, ForecasterConfig
from aetherdrift_atlas.walkforward import (
    ThresholdSelectionConfig,
    WalkForwardConfig,
    walk_forward_predict,
)


def test_walkforward_predictions_are_out_of_sample_and_scored() -> None:
    features = FeatureBuilder().build(make_demo_ohlcv("TEST", periods=300))
    predictions = walk_forward_predict(
        features,
        validation=WalkForwardConfig(min_train_size=80, test_size=40, step_size=40),
        forecaster=ForecasterConfig(n_estimators=20, min_samples_leaf=3),
    )
    metrics = DirectionForecaster.evaluate(predictions)
    assert predictions.index.is_unique
    assert len(predictions) >= 40
    assert predictions["probability_up"].between(0.0, 1.0).all()
    assert 0.0 <= metrics.direction_accuracy <= 1.0


def test_nested_threshold_selection_is_recorded_on_oos_predictions() -> None:
    features = FeatureBuilder().build(make_demo_ohlcv("SELECT", periods=350))
    predictions = walk_forward_predict(
        features,
        validation=WalkForwardConfig(min_train_size=100, test_size=40, step_size=40),
        forecaster=ForecasterConfig(n_estimators=20, min_samples_leaf=3),
        threshold_selection=ThresholdSelectionConfig(
            candidates=(0.52, 0.55), min_trades=2, reselect_every_folds=2
        ),
    )
    assert predictions["long_threshold"].isin([0.52, 0.55]).all()
    assert predictions["threshold_source"].eq("nested_validation").any()
