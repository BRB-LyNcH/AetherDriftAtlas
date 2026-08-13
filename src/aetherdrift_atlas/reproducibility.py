"""Stable fingerprints for auditable research inputs and configurations."""

from __future__ import annotations

from hashlib import sha256

import numpy as np
import pandas as pd


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Return a SHA-256 fingerprint of a sorted DataFrame and its schema.

    A research run can therefore record the exact normalized market-data input
    it used without copying a second, potentially divergent data file.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a non-empty DataFrame.")
    stable = frame.sort_index().copy()
    digest = sha256()
    digest.update("|".join(map(str, stable.columns)).encode("utf-8"))
    digest.update("|".join(map(str, stable.dtypes)).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(stable, index=True, categorize=True)
    digest.update(np.asarray(row_hashes, dtype=np.uint64).tobytes())
    return digest.hexdigest()


def market_data_manifest(frame: pd.DataFrame) -> dict[str, str | int]:
    """Describe the exact normalized market-data series used in a run."""
    return {
        "rows": len(frame),
        "first_timestamp": str(frame.index.min()),
        "last_timestamp": str(frame.index.max()),
        "sha256": frame_fingerprint(frame),
    }
