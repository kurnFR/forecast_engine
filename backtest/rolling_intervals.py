"""Strictly leakage-safe rolling P10/P50/P90 backtesting."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from backtest.metrics import wape
from backtest.xgb_checkpoint import (
    _as_key,
    _checkpoint_date as xgb_checkpoint_date,
    _build_training_frame,
    _checkpoint_row,
)
from config import CANDIDATE_MODELS, FORECAST_CONFIG
from models.baseline import forecast_from_mtd
from models.ets import forecast_ets
from models.sarima import forecast_sarima
from models.xgboost_model import predict_xgboost, train_xgboost_checkpoint

logger = logging.getLogger(__name__)


def _checkpoint_date(calendar: pd.DataFrame, periode: pd.Timestamp, checkpoint: int):
    return xgb_checkpoint_date(calendar, periode, checkpoint)


def _safe_forecast(fn, series):
    try:
        value = fn(series)
        if value is None or not np.isfinite(float(value)):
            return None
        return max(float(value), 0.0)
    except Exception as exc:
        logger.debug("%s failed during interval backtest: %s", getattr(fn, "__name__", "model"), exc)
        return None


def _prior_weights(rows: pd.DataFrame) -> dict:
    """Calculate inverse-WAPE weights from calibration rows strictly before T."""
    if rows is None or rows.empty:
        return {"baseline": 1.0}
    scores = {}
    for model in CANDIDATE_MODELS:
        pred_col = f"forecast_{model}"
        if pred_col not in rows:
            continue
        valid = rows[["actual", pred_col]].dropna()
        if valid.empty:
            continue
        score = wape(valid["actual"], valid[pred_col])
        if np.isfinite(score) and score > 0:
            scores[model] = float(score)
    if not scores:
        return {"baseline": 1.0}
    inverse = {model: 1.0 / score for model, score in scores.items()}
    total = sum(inverse.values())
    return {model: value / total for model, value in inverse.items()}


def _ensemble_with_prior_calibration(target_row: pd.Series, prior_rows: pd.DataFrame, checkpoint: int):
    """Build P10/P50/P90 using prior OOS residuals only."""
    del checkpoint
    if prior_rows is None or prior_rows.empty:
        available = {
            model: float(target_row[f"forecast_{model}"])
            for model in CANDIDATE_MODELS
            if pd.notna(target_row.get(f"forecast_{model}"))
        }
        if not available:
            return np.nan, np.nan, np.nan
        p50 = float(np.mean(list(available.values())))
        return np.nan, p50, np.nan

    weights = _prior_weights(prior_rows)
    available = {
        model: float(target_row[f"forecast_{model}"])
        for model in weights
        if pd.notna(target_row.get(f"forecast_{model}"))
    }
    if not available:
        return np.nan, np.nan, np.nan
    weight_sum = sum(weights[m] for m in available)
    effective = {m: weights[m] / weight_sum for m in available}
    p50 = float(sum(effective[m] * available[m] for m in available))

    lower, upper, used = [], [], []
    for model, weight in effective.items():
        history = prior_rows[["actual", f"forecast_{model}"]].dropna()
        if history.empty:
            continue
        residuals = history["actual"].to_numpy(dtype=float) - history[f"forecast_{model}"].to_numpy(dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        if len(residuals):
            lower.append(float(np.quantile(residuals, 0.10)))
            upper.append(float(np.quantile(residuals, 0.90)))
            used.append(weight)

    if not used:
        return np.nan, p50, np.nan
    weights_arr = np.asarray(used, dtype=float)
    weights_arr /= weights_arr.sum()
    residual_scale = float(FORECAST_CONFIG.get("interval_residual_scale", 1.0))
    if not np.isfinite(residual_scale) or residual_scale < 1.0:
        raise ValueError("interval_residual_scale must be finite and >= 1.0")
    lower_residual = float(np.dot(weights_arr, lower)) * residual_scale
    upper_residual = float(np.dot(weights_arr, upper)) * residual_scale
    p10 = max(p50 + lower_residual, 0.0)
    p90 = max(p50 + upper_residual, 0.0)
    return (min(p10, p90), p50, max(p10, p90))


def _checkpoint_xgb_prediction(monthly, daily, calendar, targets, group_cols, target_month, checkpoint, min_train_months, key):
    """Train and score checkpoint XGBoost using target months strictly before T."""
    train_frame = _build_training_frame(
        monthly, daily, calendar, targets, group_cols,
        target_month, checkpoint, min_train_months,
    )
    if train_frame.empty or len(train_frame) < min_train_months:
        return None
    try:
        model = train_xgboost_checkpoint(train_frame, target_col="monthly_value")
        row = _checkpoint_row(
            monthly, daily, calendar, targets, group_cols,
            target_month, checkpoint, _as_key(key), min_train_months,
        )
        if row is None:
            return None
        pred = predict_xgboost(model, pd.DataFrame([row]))
        return float(pred) if pred is not None and np.isfinite(pred) else None
    except (ValueError, TypeError) as exc:
        logger.debug("Checkpoint XGBoost unavailable for %s WD%s: %s", target_month, checkpoint, exc)
        return None
