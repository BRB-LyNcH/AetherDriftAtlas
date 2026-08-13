from aetherdrift_atlas.research import (
    ResearchGateConfig,
    research_gate,
    signal_diagnostics,
)


def test_research_gate_rejects_a_non_discriminating_losing_strategy() -> None:
    gate = research_gate(
        forecast_accuracy={"AAPL": {"roc_auc": 0.5}},
        diagnostics={"AAPL": {"trades": 100}},
        portfolio_metrics={
            "total_return": -0.1,
            "sharpe_ratio": -0.2,
            "maximum_drawdown": -0.3,
        },
        same_exposure_benchmark={"total_return": 0.05},
        config=ResearchGateConfig(),
    )
    assert gate["status"] == "REJECTED"
    assert "model_discrimination" in gate["failed_checks"]


def test_signal_diagnostics_reports_entered_trade_quality() -> None:
    import pandas as pd

    predictions = pd.DataFrame(
        {"signal": [1.0, -1.0, 0.0], "target_return": [0.01, -0.02, 0.03]}
    )
    diagnostics = signal_diagnostics(predictions)
    assert diagnostics["trades"] == 2
    assert diagnostics["trade_hit_rate"] == 1.0
