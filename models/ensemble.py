"""Combine candidate forecasts and derive a conservative prediction interval."""
from typing import Optional

import numpy as np
import pandas as pd

from config import MODEL_CONFIG


def combine_forecasts(row: pd.Series, weights: Optional[dict] = None) -> float:
    weights = weights or MODEL_CONFIG["ensemble_weights_default"]
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
    """Return P10/P50/P90 from model consensus and cross-model dispersion.

    This is a provisional interval until residual-quantile calibration is
    added from the full checkpoint backtest.  It is deliberately conservative
    and never returns a negative Sell-In forecast.
    """
    p50 = combine_forecasts(row, weights)
    if not np.isfinite(p50):
        return np.nan, np.nan, np.nan

    weights = weights or MODEL_CONFIG["ensemble_weights_default"]
    values = [
        float(row[f"forecast_{k}"])
        for k in weights
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
