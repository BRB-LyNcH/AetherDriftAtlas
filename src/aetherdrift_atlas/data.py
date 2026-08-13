"""Market-data adapters with validation and a transparent local CSV cache."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .assets import Instrument


class MarketDataError(RuntimeError):
    """Raised when market data cannot be downloaded or normalized safely."""


class YahooFinanceProvider:
    """Download OHLCV data from Yahoo Finance on demand.

    The provider is deliberately isolated from the rest of the application so
    a broker, exchange, or paid data source can replace it without changing
    features, model validation, or portfolio accounting.
    """

    REQUIRED_COLUMNS = ("open", "high", "low", "close")

    def download(
        self,
        instrument: Instrument,
        start: str,
        end: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download and normalize a single series to canonical OHLCV columns."""
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - dependency message
            raise MarketDataError(
                "yfinance is required. Install the project dependencies first."
            ) from exc

        raw = yf.download(
            instrument.symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        if raw is None or raw.empty:
            raise MarketDataError(
                f"No data returned for {instrument.symbol}. Check symbol and dates."
            )
        return self.normalize(raw, instrument.symbol)

    @classmethod
    def normalize(cls, raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate OHLCV fields, flatten provider columns, and sort timestamps."""
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            raise MarketDataError(f"{symbol}: expected a non-empty DataFrame.")
        frame = raw.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [str(column[0]) for column in frame.columns]
        frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
        missing = set(cls.REQUIRED_COLUMNS).difference(frame.columns)
        if missing:
            raise MarketDataError(f"{symbol}: missing required columns: {sorted(missing)}")
        if "volume" not in frame:
            # FX feeds commonly omit reliable volume. Zero communicates that
            # volume-derived features should carry no information for this asset.
            frame["volume"] = 0.0

        canonical = frame.loc[:, [*cls.REQUIRED_COLUMNS, "volume"]].apply(
            pd.to_numeric, errors="coerce"
        )
        canonical.index = pd.to_datetime(canonical.index, utc=True).tz_convert(None)
        canonical = canonical[~canonical.index.duplicated(keep="last")].sort_index()
        invalid_price = (~np.isfinite(canonical[list(cls.REQUIRED_COLUMNS)])).any(axis=1)
        invalid_price |= (canonical[list(cls.REQUIRED_COLUMNS)] <= 0.0).any(axis=1)
        canonical = canonical.loc[~invalid_price]
        canonical["volume"] = canonical["volume"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if canonical.empty:
            raise MarketDataError(f"{symbol}: no valid OHLC rows remain after cleaning.")
        return canonical.astype(float)


class CsvMarketCache:
    """Small explicit cache that makes downloaded research inputs reproducible."""

    def __init__(self, directory: str | Path = "data_cache") -> None:
        self.directory = Path(directory)

    def path_for(self, instrument: Instrument, start: str, end: str | None) -> Path:
        """Return a stable, filesystem-safe cache location for one request."""
        safe_symbol = re.sub(r"[^A-Za-z0-9_.-]+", "_", instrument.symbol)
        end_label = end or "latest"
        return self.directory / f"{safe_symbol}_{start}_{end_label}.csv"

    def load_or_download(
        self,
        provider: YahooFinanceProvider,
        instrument: Instrument,
        start: str,
        end: str | None = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Read cached normalized data or download and persist it as CSV."""
        cache_path = self.path_for(instrument, start, end)
        if cache_path.exists() and not refresh:
            frame = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return provider.normalize(frame, instrument.symbol)

        frame = provider.download(instrument, start=start, end=end)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index_label="timestamp")
        return frame
