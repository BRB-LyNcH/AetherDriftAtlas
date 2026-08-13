"""Diagnostics and rejection gates for trustworthy strategy research."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResearchGateConfig:
    """Minimum evidence required before a strategy becomes a paper-trading candidate."""

    min_roc_auc: float = 0.52
    min_net_sharpe: float = 0.5
    max_drawdown: float = 0.25
    min_trades: int = 60

    def __post_init__(self) -> None:
        if not 0.5 <= self.min_roc_auc <= 1.0:
            raise ValueError("min_roc_auc must be between 0.5 and 1.0.")
        if self.max_drawdown <= 0.0 or self.max_drawdown > 1.0:
            raise ValueError("max_drawdown must be in (0, 1].")
        if self.min_trades < 1:
            raise ValueError("min_trades must be positive.")


def signal_diagnostics(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Measure whether entered signals, rather than raw labels, add value."""
    required = {"signal", "target_return"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions is missing: {sorted(missing)}")
    signals = predictions["signal"].astype(float)
    forward_returns = predictions["target_return"].astype(float)
    traded = signals != 0.0
    signed_returns = (signals * forward_returns).loc[traded]
    long_trades = int((signals == 1.0).sum())
    short_trades = int((signals == -1.0).sum())
    hit_rate = float((signed_returns > 0.0).mean()) if not signed_returns.empty else float("nan")
    mean_return = float(signed_returns.mean()) if not signed_returns.empty else float("nan")
    standard_deviation = float(signed_returns.std(ddof=1)) if len(signed_returns) > 1 else 0.0
    return {
        "trades": int(traded.sum()),
        "trade_rate": float(traded.mean()),
        "long_trades": long_trades,
        "short_trades": short_trades,
        "trade_hit_rate": hit_rate,
        "mean_signed_return": mean_return,
        "median_signed_return": (
            float(signed_returns.median()) if not signed_returns.empty else float("nan")
        ),
        "signal_sharpe_before_cost": (
            float(signed_returns.mean() / standard_deviation * np.sqrt(365.0))
            if standard_deviation > 0.0
            else float("nan")
        ),
    }


def research_gate(
    forecast_accuracy: dict[str, dict[str, float | int]],
    diagnostics: dict[str, dict[str, float | int]],
    portfolio_metrics: dict[str, float],
    same_exposure_benchmark: dict[str, float],
    config: ResearchGateConfig | None = None,
) -> dict[str, object]:
    """Return an explicit approve/reject decision with auditable evidence.

    Passing this gate does not establish future profitability. It only confirms
    that the historical research clears predeclared minimum controls and can be
    considered for a separate paper-trading stage.
    """
    policy = config or ResearchGateConfig()
    auc_values = [
        float(metrics["roc_auc"])
        for metrics in forecast_accuracy.values()
        if math.isfinite(float(metrics["roc_auc"]))
    ]
    best_auc = max(auc_values, default=float("nan"))
    trades = sum(int(metric["trades"]) for metric in diagnostics.values())
    strategy_total_return = portfolio_metrics["total_return"]
    baseline_total_return = same_exposure_benchmark["total_return"]
    checks = {
        "model_discrimination": {
            "passed": bool(best_auc >= policy.min_roc_auc),
            "observed": best_auc,
            "minimum": policy.min_roc_auc,
        },
        "sufficient_trades": {
            "passed": trades >= policy.min_trades,
            "observed": trades,
            "minimum": policy.min_trades,
        },
        "risk_adjusted_return": {
            "passed": bool(portfolio_metrics["sharpe_ratio"] >= policy.min_net_sharpe),
            "observed": portfolio_metrics["sharpe_ratio"],
            "minimum": policy.min_net_sharpe,
        },
        "drawdown_limit": {
            "passed": bool(portfolio_metrics["maximum_drawdown"] >= -policy.max_drawdown),
            "observed": portfolio_metrics["maximum_drawdown"],
            "minimum": -policy.max_drawdown,
        },
        "adds_directional_value": {
            "passed": bool(strategy_total_return > baseline_total_return),
            "observed": strategy_total_return - baseline_total_return,
            "minimum": 0.0,
        },
    }
    failures = [name for name, value in checks.items() if not bool(value["passed"])]
    return {
        "status": "CANDIDATE_FOR_PAPER_TRADING" if not failures else "REJECTED",
        "checks": checks,
        "failed_checks": failures,
        "policy": {
            "min_roc_auc": policy.min_roc_auc,
            "min_net_sharpe": policy.min_net_sharpe,
            "max_drawdown": policy.max_drawdown,
            "min_trades": policy.min_trades,
        },
    }
