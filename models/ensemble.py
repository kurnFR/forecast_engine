"""Combine candidate forecasts and derive a conservative prediction interval."""
from typing import Optional

import numpy as np
import pandas as pd

from config import MODEL_CONFIG

CANDIDATE_MODELS = ("baseline", "ets", "sarima", "xgboost")


def _row_backtest_weights(row: pd.Series) -> dict:
    """Derive inverse-WAPE weights from checkpoint backtest scores when present."""
    scores = {}
    for model in CANDIDATE_MODELS:
        col = f"{model}_checkpoint_score"
        if col in row and pd.notna(row[col]) and float(row[col]) > 0:
            scores[model] = float(row[col])
    if not scores:
        return MODEL_CONFIG["ensemble_weights_default"]

    inverse = {model: 1.0 / score for model, score in scores.items()}
    total = sum(inverse.values())
    return {model: value / total for model, value in inverse.items()}


def combine_forecasts(row: pd.Series, weights: Optional[dict] = None) -> float:
    weights = weights or _row_backtest_weights(row)
    available = {
        k: float(row[f"forecast_{k}"])
        for k in weights
        if f"forecast_{k}" in row and pd.notna(row[f"forecast_{k}"])
    }
    if not available:
        return np.nan
    w_sum = sum(weights[k] for k in available)
    return float(sum(weights[k] * value for k, value in available.items()) / w_sum)


def prediction_interval(row: pd.Series, weights: Optional[dict] = None) -> tuple[float, float, float]:
    """Return a provisional P10/P50/P90 interval.

    P50 uses inverse-WAPE backtest weights when checkpoint scores are attached
    to the row.  P10/P90 remain explicitly provisional until residual-based
    calibration is added from out-of-sample checkpoint errors.
    """
    p50 = combine_forecasts(row, weights)
    if not np.isfinite(p50):
        return np.nan, np.nan, np.nan

    effective_weights = weights or _row_backtest_weights(row)
    values = [
        float(row[f"forecast_{k}"])
        for k in effective_weights
        if f"forecast_{k}" in row and pd.notna(row[f"forecast_{k}"])
    ]
    if len(values) < 2:
        spread = abs(p50) * 0.15
    else:
        spread = max(float(np.std(values, ddof=1)), abs(p50) * 0.05)

    return max(p50 - 1.28 * spread, 0.0), p50, max(p50 + 1.28 * spread, 0.0)


def build_ensemble(df: pd.DataFrame, weights: Optional[dict] = None) -> pd.DataFrame:
    out = df.copy()
    intervals = out.apply(lambda r: prediction_interval(r, weights), axis=1, result_type="expand")
    intervals.columns = ["forecast_p10", "forecast_p50", "forecast_p90"]
    return pd.concat([out, intervals], axis=1)
