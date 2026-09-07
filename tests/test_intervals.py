import numpy as np
import pandas as pd

from backtest.intervals import evaluate_intervals, interval_quality_flags
from models.ensemble import interval_coverage_metrics


def test_interval_metrics_expected_values():
    actual = [80, 100, 120, 100, 100]
    p10 = [70, 90, 110, 90, 90]
    p50 = [80, 100, 120, 100, 100]
    p90 = [90, 110, 130, 110, 110]
    metrics = interval_coverage_metrics(actual, p10, p50, p90)
    assert metrics["observations"] == 5
    assert metrics["coverage"] == 1.0
    assert metrics["mae_p50"] == 0.0
    assert metrics["mean_interval_width"] == 20.0


def test_interval_evaluation_can_split_by_checkpoint():
    df = pd.DataFrame({
        "regioncode": ["A", "A", "B", "B"],
        "checkpoint": [4, 7, 4, 7],
        "actual": [100, 110, 200, 190],
        "forecast_p10": [90, 100, 180, 170],
        "forecast_p50": [100, 105, 195, 190],
        "forecast_p90": [110, 120, 210, 210],
    })
    result = evaluate_intervals(df, ["regioncode"])
    assert len(result) == 4
    assert set(result["checkpoint"]) == {4, 7}


def test_interval_quality_flags():
    metrics = pd.DataFrame({"coverage": [0.80, 0.65, 0.95]})
    result = interval_quality_flags(metrics)
    assert list(result["coverage_status"]) == ["OK", "REVIEW", "REVIEW"]


def test_invalid_interval_rows_are_ignored():
    metrics = interval_coverage_metrics(
        [100, np.nan], [90, 90], [100, 100], [110, 110]
    )
    assert metrics["observations"] == 1
