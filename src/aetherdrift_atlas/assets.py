"""Asset metadata and calendar conventions used by the research stack."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AssetClass(str, Enum):
    """Supported investable asset categories."""

    EQUITY = "equity"
    ETF = "etf"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"


_PERIODS_PER_YEAR: dict[AssetClass, int] = {
    AssetClass.EQUITY: 252,
    AssetClass.ETF: 252,
    AssetClass.CRYPTO: 365,
    AssetClass.FOREX: 260,
    AssetClass.COMMODITY: 252,
}


@dataclass(frozen=True)
class Instrument:
    """A tradable series and the assumptions needed to evaluate it.

    Symbols use the provider's naming convention. Yahoo Finance examples are
    ``AAPL`` (equity), ``SPY`` (ETF), ``BTC-USD`` (crypto), and ``EURUSD=X``
    (foreign exchange).
    """

    symbol: str
    asset_class: AssetClass
    quote_currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string.")

    @property
    def periods_per_year(self) -> int:
        """Return the conventional annualization factor for daily bars."""
        return _PERIODS_PER_YEAR[self.asset_class]


def parse_asset_class(value: str) -> AssetClass:
    """Parse a case-insensitive asset class supplied through the CLI."""
    try:
        return AssetClass(value.strip().lower())
    except (AttributeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in AssetClass)
        raise ValueError(f"Unsupported asset class '{value}'. Use: {allowed}.") from exc
