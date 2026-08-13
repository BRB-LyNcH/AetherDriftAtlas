"""Deterministic synthetic data for offline demonstrations and tests."""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd


def make_demo_ohlcv(symbol: str, periods: int = 900) -> pd.DataFrame:
    """Create a reproducible daily OHLCV series with several market regimes.

    It is intentionally for plumbing demonstrations only—not simulated evidence
    that a strategy has investable edge.
    """
    if periods < 100:
        raise ValueError("periods must be at least 100.")
    seed = zlib.adler32(symbol.encode("utf-8"))
    generator = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=periods)
    time = np.arange(periods, dtype=float)
    regime = 0.00015 + 0.0015 * np.sin(time / 55.0)
    innovations = generator.normal(0.0, 0.010, periods)
    close = 100.0 * np.exp(np.cumsum(regime + innovations))
    overnight = generator.normal(0.0, 0.0025, periods)
    open_ = close * (1.0 + overnight)
    intraday_range = np.abs(generator.normal(0.008, 0.003, periods))
    high = np.maximum(open_, close) * (1.0 + intraday_range)
    low = np.minimum(open_, close) * (1.0 - intraday_range)
    volume = generator.lognormal(mean=13.0, sigma=0.35, size=periods)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
