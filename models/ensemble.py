"""Backtest-driven ensemble and residual-calibrated prediction intervals."""
from typing import Optional

import numpy as np
import pandas as pd

from config import MODEL_CONFIG

CANDIDATE_MODELS = ("baseline", "ets", "sarima", "xgboost")


def _row_backtest_weights(row: pd.Series) -> dict:
    """Derive normalized inverse-WAPE weights from checkpoint backtest scores."""
    scores = {}
    for model in CANDIDATE_MODELS:
        col = f"{model}_checkpoint_score"
        if col in row and pd.notna(row[col]) and np.isfinite(float(row[col])) and float(row[col]) > 0:
            scores[model] = float(row[col])
    if not scores:
        scores = {k: float(v) for k, v in MODEL_CONFIG["ensemble_weights_default"].items() if k in CANDIDATE_MODELS}

    inverse = {model: 1.0 / score for model, score in scores.items() if score > 0}
    total = sum(inverse.values())
    return {model: value / total for model, value in inverse.items()} if total > 0 else {"baseline": 1.0}


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


def _residual_interval(row: pd.Series, weights: dict, p50: float) -> tuple[float, float]:
    """Combine model-specific OOS residual quantiles around the ensemble P50."""
    lower, upper, used = [], [], []
    for model, weight in weights.items():
        q10 = row.get(f"{model}_residual_q10")
        q90 = row.get(f"{model}_residual_q90")
        if pd.notna(q10) and pd.notna(q90) and np.isfinite(float(q10)) and np.isfinite(float(q90)):
            lower.append(float(q10))
            upper.append(float(q90))
            used.append(float(weight))

    if not used:
        spread = max(abs(p50) * 0.15, 1.0)
        return max(p50 - 1.28 * spread, 0.0), max(p50 + 1.28 * spread, 0.0)

    weights_arr = np.asarray(used, dtype=float)
    weights_arr /= weights_arr.sum()
    q10 = float(np.dot(weights_arr, np.asarray(lower)))
    q90 = float(np.dot(weights_arr, np.asarray(upper)))
    return max(p50 + q10, 0.0), max(p50 + q90, 0.0)


def prediction_interval(row: pd.Series, weights: Optional[dict] = None) -> tuple[float, float, float]:
    """Return P10/P50/P90 using backtest-derived weights and OOS residuals."""
    effective_weights = weights or _row_backtest_weights(row)
    p50 = combine_forecasts(row, effective_weights)
    if not np.isfinite(p50):
        return np.nan, np.nan, np.nan
    p10, p90 = _residual_interval(row, effective_weights, p50)
    return p10, p50, p90


def build_ensemble(df: pd.DataFrame, weights: Optional[dict] = None) -> pd.DataFrame:
    """Build production P10/P50/P90 from candidate forecasts and OOS residual calibration."""
    out = df.copy()
    intervals = out.apply(
        lambda r: prediction_interval(r, weights), axis=1, result_type="expand"
    )
    intervals.columns = ["forecast_p10", "forecast_p50", "forecast_p90"]
    return pd.concat([out, intervals], axis=1)


def interval_coverage_metrics(actual, p10, p50, p90) -> dict:
    """Calculate empirical interval diagnostics for out-of-sample forecasts.

    P10/P90 are interpreted as quantile bounds: roughly 10% of observations
    should fall below P10 and 10% above P90, while the central P10-P90 interval
    should cover roughly 80% of observations. Metrics are intentionally
    descriptive; production acceptance thresholds belong in backtest policy.
    """
    a = np.asarray(actual, dtype=float)
    lo = np.asarray(p10, dtype=float)
    mid = np.asarray(p50, dtype=float)
    hi = np.asarray(p90, dtype=float)
    valid = np.isfinite(a) & np.isfinite(lo) & np.isfinite(mid) & np.isfinite(hi)
    if not valid.any():
        return {
            "observations": 0,
            "p10_below_rate": np.nan,
            "p90_above_rate": np.nan,
            "coverage": np.nan,
            "mean_interval_width": np.nan,
            "mae_p50": np.nan,
        }

    a, lo, mid, hi = a[valid], lo[valid], mid[valid], hi[valid]
    return {
        "observations": int(len(a)),
        "p10_below_rate": float(np.mean(a < lo)),
        "p90_above_rate": float(np.mean(a > hi)),
        "coverage": float(np.mean((a >= lo) & (a <= hi))),
        "mean_interval_width": float(np.mean(hi - lo)),
        "mae_p50": float(np.mean(np.abs(a - mid))),
    }
