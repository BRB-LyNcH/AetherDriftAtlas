"""Cost-aware portfolio accounting and fair multi-asset benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    """Execution, leverage, and annualization assumptions for a portfolio run."""

    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    max_gross_leverage: float = 1.0
    max_asset_weight: float = 0.35
    volatility_lookback: int = 20
    periods_per_year: int = 365

    def __post_init__(self) -> None:
        if self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be positive.")
        if self.transaction_cost_bps < 0.0 or self.slippage_bps < 0.0:
            raise ValueError("Costs cannot be negative.")
        if self.max_gross_leverage <= 0.0:
            raise ValueError("max_gross_leverage must be positive.")
        if not 0.0 < self.max_asset_weight <= self.max_gross_leverage:
            raise ValueError("max_asset_weight must be in (0, max_gross_leverage].")
        if self.volatility_lookback < 2 or self.periods_per_year < 1:
            raise ValueError("volatility_lookback and periods_per_year must be positive.")

    @property
    def cost_rate(self) -> float:
        """Return the assumed one-way cost per unit of portfolio turnover."""
        return (self.transaction_cost_bps + self.slippage_bps) * 1e-4


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """Strategy path, executed weights, benchmarks, and performance metrics."""

    history: pd.DataFrame
    weights: pd.DataFrame
    metrics: dict[str, float]
    benchmarks: dict[str, dict[str, float]]


class MultiAssetBacktester:
    """Turn point-in-time signals into risk-balanced, costed portfolio returns.

    ``forward_returns`` is indexed at the decision bar: the value at *t* is
    the realized return from *t* to *t+1*. A signal at *t* therefore applies to
    that explicitly aligned forward return, not to an already observed bar.
    """

    def __init__(self, config: PortfolioBacktestConfig | None = None) -> None:
        self.config = config or PortfolioBacktestConfig()

    def run(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> PortfolioBacktestResult:
        """Backtest signals and produce matched and full-exposure baselines.

        ``same_exposure_long`` uses the strategy's *absolute* weights. It has
        precisely the same per-day exposure and asset mix, but is always long.
        It is therefore a fair test of whether the model's directional calls add
        value. ``full_exposure_long`` is a separate long-only market reference
        that invests up to configured portfolio exposure whenever data exists.
        """
        self._validate_inputs(signals, forward_returns)
        returns = forward_returns.sort_index().astype(float)
        signal_frame = signals.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
        tradable = returns.notna()
        signal_frame = signal_frame.where(tradable, 0.0)
        returns = returns.fillna(0.0)

        weights = self._risk_balanced_weights(signal_frame, returns)
        strategy_path = self._path_from_weights(weights, returns)
        matched_long_path = self._path_from_weights(weights.abs(), returns)
        full_long_weights = self._risk_balanced_weights(tradable.astype(float), returns)
        full_long_path = self._path_from_weights(full_long_weights, returns)

        history = pd.DataFrame(index=returns.index)
        self._add_path_columns(history, strategy_path, prefix="")
        self._add_path_columns(history, matched_long_path, prefix="same_exposure_long_")
        self._add_path_columns(history, full_long_path, prefix="full_exposure_long_")
        # Backwards-compatible aliases now refer to the transparent full-market
        # reference; research gates use the fair same-exposure baseline instead.
        history["benchmark_return"] = history["full_exposure_long_net_return"]
        history["benchmark_equity"] = history["full_exposure_long_equity"]

        metrics = self._strategy_metrics(strategy_path)
        benchmarks = {
            "same_exposure_long": self._net_metrics(matched_long_path),
            "full_exposure_long": self._net_metrics(full_long_path),
        }
        return PortfolioBacktestResult(
            history=history,
            weights=weights,
            metrics=metrics,
            benchmarks=benchmarks,
        )

    def _risk_balanced_weights(
        self, signals: pd.DataFrame, returns: pd.DataFrame
    ) -> pd.DataFrame:
        annualized_vol = returns.shift(1).rolling(
            self.config.volatility_lookback, min_periods=2
        ).std(ddof=0) * np.sqrt(self.config.periods_per_year)
        inverse_vol = 1.0 / annualized_vol.replace(0.0, np.nan)
        desired = signals * inverse_vol
        gross = desired.abs().sum(axis=1).replace(0.0, np.nan)
        weights = desired.div(gross, axis=0) * self.config.max_gross_leverage
        # A warm-up should not quietly create future knowledge or abandon a
        # valid forecast. Equal risk is unavailable, so use equal notional.
        fallback_gross = signals.abs().sum(axis=1).replace(0.0, np.nan)
        fallback = signals.div(fallback_gross, axis=0) * self.config.max_gross_leverage
        weights = weights.fillna(fallback).fillna(0.0)
        return weights.clip(
            lower=-self.config.max_asset_weight,
            upper=self.config.max_asset_weight,
        )

    def _path_from_weights(
        self, weights: pd.DataFrame, returns: pd.DataFrame
    ) -> pd.DataFrame:
        gross_returns = (weights * returns).sum(axis=1)
        previous_weights = weights.shift(1).fillna(0.0)
        turnover = (weights - previous_weights).abs().sum(axis=1)
        trading_cost = turnover * self.config.cost_rate
        net_returns = gross_returns - trading_cost
        gross_equity = self.config.initial_capital * (1.0 + gross_returns).cumprod()
        equity = self.config.initial_capital * (1.0 + net_returns).cumprod()
        previous_equity = equity.shift(1).fillna(self.config.initial_capital)
        return pd.DataFrame(
            {
                "gross_return": gross_returns,
                "trading_cost": trading_cost,
                "net_return": net_returns,
                "turnover": turnover,
                "gross_exposure": weights.abs().sum(axis=1),
                "gross_equity": gross_equity,
                "equity": equity,
                "estimated_cost_dollars": previous_equity * trading_cost,
                "drawdown": equity / equity.cummax() - 1.0,
            },
            index=returns.index,
        )

    @staticmethod
    def _add_path_columns(history: pd.DataFrame, path: pd.DataFrame, prefix: str) -> None:
        for column in path.columns:
            history[f"{prefix}{column}"] = path[column]

    def _strategy_metrics(self, path: pd.DataFrame) -> dict[str, float]:
        """Report net performance together with explicit gross/cost attribution."""
        metrics = self._net_metrics(path)
        gross_metrics = self._return_metrics(path["gross_return"], path["gross_equity"])
        metrics.update(
            {
                "gross_total_return": gross_metrics["total_return"],
                "gross_annualized_return": gross_metrics["annualized_return"],
                "gross_sharpe_ratio": gross_metrics["sharpe_ratio"],
                "cost_drag_total_return": gross_metrics["total_return"] - metrics["total_return"],
                "cumulative_cost_return": float(path["trading_cost"].sum()),
                "estimated_total_cost_dollars": float(path["estimated_cost_dollars"].sum()),
                # Kept for compatibility with existing output readers.
                "total_cost": float(path["trading_cost"].sum()),
            }
        )
        return metrics

    def _net_metrics(self, path: pd.DataFrame) -> dict[str, float]:
        metrics = self._return_metrics(path["net_return"], path["equity"])
        metrics.update(
            {
                "maximum_drawdown": float(path["drawdown"].min()),
                "calmar_ratio": self._calmar(
                    metrics["annualized_return"], float(path["drawdown"].min())
                ),
                "average_turnover": float(path["turnover"].mean()),
                "final_equity": float(path["equity"].iloc[-1]),
            }
        )
        return metrics

    def _return_metrics(self, returns: pd.Series, equity: pd.Series) -> dict[str, float]:
        periods = len(returns)
        annualization = self.config.periods_per_year
        standard_deviation = float(returns.std(ddof=1)) if periods > 1 else 0.0
        total_return = float(equity.iloc[-1] / self.config.initial_capital - 1.0)
        annual_return = (
            float((1.0 + total_return) ** (annualization / periods) - 1.0)
            if periods
            else float("nan")
        )
        sharpe = (
            float(returns.mean() / standard_deviation * np.sqrt(annualization))
            if standard_deviation > 0.0
            else float("nan")
        )
        return {
            "total_return": total_return,
            "annualized_return": annual_return,
            "annualized_volatility": standard_deviation * np.sqrt(annualization),
            "sharpe_ratio": sharpe,
        }

    @staticmethod
    def _calmar(annualized_return: float, maximum_drawdown: float) -> float:
        return (
            float(annualized_return / abs(maximum_drawdown))
            if maximum_drawdown < 0.0
            else float("nan")
        )

    @staticmethod
    def _validate_inputs(signals: pd.DataFrame, forward_returns: pd.DataFrame) -> None:
        if not isinstance(signals, pd.DataFrame) or not isinstance(forward_returns, pd.DataFrame):
            raise TypeError("signals and forward_returns must be DataFrames.")
        if forward_returns.empty:
            raise ValueError("forward_returns cannot be empty.")
        if forward_returns.index.has_duplicates:
            raise ValueError("forward_returns index cannot contain duplicate timestamps.")
        if not set(forward_returns.columns).issubset(signals.columns):
            raise ValueError("signals must contain every asset in forward_returns.")
