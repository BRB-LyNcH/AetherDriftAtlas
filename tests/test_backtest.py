import pandas as pd

from aetherdrift_atlas.backtest import MultiAssetBacktester, PortfolioBacktestConfig


def test_backtest_applies_return_and_costs() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    signals = pd.DataFrame({"AAPL": [1.0, 1.0, 1.0]}, index=index)
    returns = pd.DataFrame({"AAPL": [0.01, 0.01, 0.01]}, index=index)
    result = MultiAssetBacktester(
        PortfolioBacktestConfig(
            initial_capital=100.0,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
            max_asset_weight=1.0,
            max_gross_leverage=1.0,
        )
    ).run(signals, returns)
    assert abs(result.history["equity"].iloc[-1] - 100.0 * 1.01**3) < 1e-10
    assert result.history["turnover"].iloc[0] == 1.0
    assert result.metrics["total_cost"] == 0.0


def test_same_exposure_baseline_keeps_the_strategy_exposure() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    signals = pd.DataFrame({"AAPL": [1.0, -1.0]}, index=index)
    returns = pd.DataFrame({"AAPL": [0.01, 0.01]}, index=index)
    result = MultiAssetBacktester(
        PortfolioBacktestConfig(
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
            max_asset_weight=1.0,
            max_gross_leverage=1.0,
        )
    ).run(signals, returns)
    assert result.history["gross_exposure"].equals(
        result.history["same_exposure_long_gross_exposure"]
    )
    assert result.benchmarks["same_exposure_long"]["total_return"] > result.metrics["total_return"]
