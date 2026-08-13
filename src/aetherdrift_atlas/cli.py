"""Command-line orchestration for reproducible Atlas research runs."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .assets import Instrument, parse_asset_class
from .backtest import MultiAssetBacktester, PortfolioBacktestConfig
from .data import CsvMarketCache, YahooFinanceProvider
from .demo import make_demo_ohlcv
from .features import FEATURE_COLUMNS, FeatureBuilder, FeatureConfig
from .model import DirectionForecaster, ForecasterConfig
from .reproducibility import market_data_manifest
from .research import ResearchGateConfig, research_gate, signal_diagnostics
from .walkforward import (
    ThresholdSelectionConfig,
    WalkForwardConfig,
    walk_forward_predict,
)

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the public command-line interface."""
    parser = argparse.ArgumentParser(
        description="Leakage-aware, multi-asset walk-forward research and backtesting."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "BTC-USD", "EURUSD=X"],
        help="Provider symbols (for example: AAPL BTC-USD EURUSD=X).",
    )
    parser.add_argument(
        "--asset-classes",
        nargs="+",
        default=["equity", "crypto", "forex"],
        help="One class per symbol: equity, etf, crypto, forex, or commodity.",
    )
    parser.add_argument("--start", default="2018-01-01", help="Inclusive YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Exclusive YYYY-MM-DD.")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached downloads.")
    parser.add_argument("--demo", action="store_true", help="Run offline against synthetic data.")
    parser.add_argument("--output", default="outputs/latest", help="Folder for CSV and JSON artifacts.")
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=504,
        help="Minimum outer training history; 504 daily bars is roughly two years.",
    )
    parser.add_argument("--test-size", type=int, default=63)
    parser.add_argument("--step-size", type=int, default=63)
    parser.add_argument("--target-horizon", type=int, default=1)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--long-threshold", type=float, default=0.55)
    parser.add_argument("--short-threshold", type=float, default=0.45)
    parser.add_argument(
        "--auto-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Select confidence thresholds only inside prior nested validation windows.",
    )
    parser.add_argument(
        "--threshold-candidates",
        nargs="+",
        type=float,
        default=[0.52, 0.55, 0.58, 0.61],
        help="Long probability thresholds considered by nested validation.",
    )
    parser.add_argument(
        "--threshold-reselect-every-folds",
        type=int,
        default=4,
        help="Re-evaluate the nested threshold after this many outer folds.",
    )
    parser.add_argument("--threshold-min-trades", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--minimum-roc-auc", type=float, default=0.52)
    parser.add_argument("--minimum-sharpe", type=float, default=0.5)
    parser.add_argument("--maximum-drawdown", type=float, default=0.25)
    parser.add_argument("--minimum-gate-trades", type=int, default=60)
    return parser


def parse_instruments(symbols: Sequence[str], asset_classes: Sequence[str]) -> list[Instrument]:
    """Pair CLI symbols to their asset classes without accidental assumptions."""
    if len(symbols) != len(asset_classes):
        raise ValueError("--symbols and --asset-classes must contain the same number of values.")
    return [
        Instrument(symbol=symbol, asset_class=parse_asset_class(asset_class))
        for symbol, asset_class in zip(symbols, asset_classes, strict=True)
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the requested research run and write all reproducibility artifacts."""
    instruments = parse_instruments(args.symbols, args.asset_classes)
    feature_config = FeatureConfig(target_horizon=args.target_horizon)
    builder = FeatureBuilder(feature_config)
    validation = WalkForwardConfig(
        min_train_size=args.min_train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        embargo_bars=args.target_horizon,
    )
    forecaster = ForecasterConfig(n_estimators=args.trees)
    threshold_selection = (
        ThresholdSelectionConfig(
            candidates=tuple(args.threshold_candidates),
            min_trades=args.threshold_min_trades,
            cost_bps=args.transaction_cost_bps + args.slippage_bps,
            reselect_every_folds=args.threshold_reselect_every_folds,
        )
        if args.auto_threshold
        else None
    )
    provider = YahooFinanceProvider()
    cache = CsvMarketCache()
    per_asset_predictions: dict[str, pd.DataFrame] = {}
    accuracy: dict[str, dict[str, float | int]] = {}
    diagnostics: dict[str, dict[str, float | int]] = {}
    data_manifest: dict[str, dict[str, str | int]] = {}

    for instrument in instruments:
        raw = (
            make_demo_ohlcv(instrument.symbol)
            if args.demo
            else cache.load_or_download(
                provider,
                instrument,
                start=args.start,
                end=args.end,
                refresh=args.refresh,
            )
        )
        data_manifest[instrument.symbol] = market_data_manifest(raw)
        features = builder.build(raw, include_target=True)
        predictions = walk_forward_predict(
            features,
            validation=validation,
            forecaster=forecaster,
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            threshold_selection=threshold_selection,
        )
        per_asset_predictions[instrument.symbol] = predictions
        accuracy[instrument.symbol] = DirectionForecaster.evaluate(predictions).as_dict()
        diagnostics[instrument.symbol] = signal_diagnostics(predictions)
        LOGGER.info(
            "%s: %d OOS forecasts, direction accuracy %.2f%%, trade rate %.1f%%",
            instrument.symbol,
            len(predictions),
            100.0 * float(accuracy[instrument.symbol]["direction_accuracy"]),
            100.0 * float(diagnostics[instrument.symbol]["trade_rate"]),
        )

    forward_returns = pd.concat(
        {symbol: frame["target_return"] for symbol, frame in per_asset_predictions.items()},
        axis=1,
    ).sort_index()
    signals = pd.concat(
        {symbol: frame["signal"] for symbol, frame in per_asset_predictions.items()}, axis=1
    ).sort_index()
    result = MultiAssetBacktester(
        PortfolioBacktestConfig(
            initial_capital=args.initial_capital,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
        )
    ).run(signals, forward_returns)
    gate = research_gate(
        forecast_accuracy=accuracy,
        diagnostics=diagnostics,
        portfolio_metrics=result.metrics,
        same_exposure_benchmark=result.benchmarks["same_exposure_long"],
        config=ResearchGateConfig(
            min_roc_auc=args.minimum_roc_auc,
            min_net_sharpe=args.minimum_sharpe,
            max_drawdown=args.maximum_drawdown,
            min_trades=args.minimum_gate_trades,
        ),
    )
    artifacts = {
        "forecast_accuracy": accuracy,
        "signal_diagnostics": diagnostics,
        "portfolio_metrics": result.metrics,
        "benchmarks": result.benchmarks,
        "research_gate": gate,
        "research_configuration": {
            "auto_threshold": args.auto_threshold,
            "threshold_candidates": args.threshold_candidates,
            "threshold_reselect_every_folds": args.threshold_reselect_every_folds,
            "threshold_min_trades": args.threshold_min_trades,
            "transaction_cost_bps": args.transaction_cost_bps,
            "slippage_bps": args.slippage_bps,
        },
        "run_manifest": {
            "package_version": "0.1.0",
            "market_data": data_manifest,
            "feature_columns": list(FEATURE_COLUMNS),
            "feature_config": asdict(feature_config),
            "walk_forward_config": asdict(validation),
            "forecaster_config": asdict(forecaster),
        },
        "instruments": [
            {"symbol": item.symbol, "asset_class": item.asset_class.value}
            for item in instruments
        ],
    }
    _write_artifacts(Path(args.output), per_asset_predictions, result.history, result.weights, artifacts)
    return artifacts


def _write_artifacts(
    output_dir: Path,
    predictions: dict[str, pd.DataFrame],
    history: pd.DataFrame,
    weights: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for symbol, frame in predictions.items():
        safe_symbol = symbol.replace("/", "_").replace("=", "_")
        frame.to_csv(output_dir / f"{safe_symbol}_predictions.csv", index_label="timestamp")
    history.to_csv(output_dir / "portfolio_history.csv", index_label="timestamp")
    weights.to_csv(output_dir / "portfolio_weights.csv", index_label="timestamp")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, sort_keys=True)


def _json_safe(value: object) -> object:
    """Replace NaN metrics with JSON null so artifacts are standards-compliant."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run research, and provide a concise final report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        artifacts = run(args)
    except (RuntimeError, ValueError) as exc:
        LOGGER.error("Run failed: %s", exc)
        return 1
    metrics = artifacts["portfolio_metrics"]
    print("\nPortfolio results (walk-forward out-of-sample)")
    print(f"Final equity: ${metrics['final_equity']:,.2f}")
    print(f"Total return: {metrics['total_return']:.2%}")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Maximum drawdown: {metrics['maximum_drawdown']:.2%}")
    print(f"Gross return before costs: {metrics['gross_total_return']:.2%}")
    print(f"Cost drag: {metrics['cost_drag_total_return']:.2%}")
    print(f"Research gate: {artifacts['research_gate']['status']}")
    print(f"Artifacts: {Path(args.output).resolve()}")
    return 0
