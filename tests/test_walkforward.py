from aetherdrift_atlas.walkforward import ExpandingWindowSplitter, WalkForwardConfig


def test_embargo_separates_training_and_test() -> None:
    splitter = ExpandingWindowSplitter(
        WalkForwardConfig(min_train_size=50, test_size=20, step_size=20, embargo_bars=2)
    )
    folds = list(splitter.split(120))
    assert folds
    for train_slice, test_slice in folds:
        assert train_slice.start == 0
        assert train_slice.stop + 2 == test_slice.start
        assert test_slice.stop > test_slice.start
