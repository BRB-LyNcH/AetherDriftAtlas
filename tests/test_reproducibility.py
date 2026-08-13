import pandas as pd

from aetherdrift_atlas.reproducibility import frame_fingerprint, market_data_manifest


def test_frame_fingerprint_is_stable_and_detects_data_changes() -> None:
    frame = pd.DataFrame(
        {"close": [100.0, 101.0]}, index=pd.date_range("2024-01-01", periods=2)
    )
    original = frame_fingerprint(frame)
    assert frame_fingerprint(frame.copy()) == original
    changed = frame.copy()
    changed.loc[changed.index[-1], "close"] = 102.0
    assert frame_fingerprint(changed) != original
    manifest = market_data_manifest(frame)
    assert manifest["rows"] == 2
    assert manifest["sha256"] == original
