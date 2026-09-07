"""Leakage-safe XGBoost models for region-month Sell-In forecasting."""
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from config import MODEL_CONFIG

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6",
    "mom_growth", "rolling_mean_3", "rolling_std_3", "seasonal_index",
    "calendar_month", "total_working_days",
]

CHECKPOINT_FEATURE_COLS = FEATURE_COLS + [
    "mtd_value",
    "elapsed_working_days",
    "remaining_working_days",
    "avg_daily_rate",
    "run_rate_forecast",
    "target_sellin",
    "mtd_target_pct",
    "checkpoint",
]


def _training_frame(
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    cols = [c for c in feature_cols if c in train_df.columns]
    if not cols:
        raise ValueError("No XGBoost feature columns are present.")
    d = train_df.dropna(subset=cols + [target_col]).copy()
    if d.empty:
        raise ValueError("No rows left after dropping NaNs - not enough history to train XGBoost.")
    return d, cols


def train_xgboost(train_df: pd.DataFrame, target_col: str = "monthly_value") -> XGBRegressor:
    """Fit one pooled history-only region-month model."""
    d, cols = _training_frame(train_df, target_col, FEATURE_COLS)
    model = XGBRegressor(**MODEL_CONFIG["xgboost"])
    model.fit(d[cols], d[target_col])
    model._feature_cols_used = cols
    return model


def train_xgboost_checkpoint(train_df: pd.DataFrame, target_col: str = "monthly_value") -> XGBRegressor:
    """Fit a checkpoint model whose features may include target-month MTD data."""
    d, cols = _training_frame(train_df, target_col, CHECKPOINT_FEATURE_COLS)
    model = XGBRegressor(**MODEL_CONFIG["xgboost"])
    model.fit(d[cols], d[target_col])
    model._feature_cols_used = cols
    model._checkpoint_aware = True
    return model


def train_xgboost_checkpoint_models(
    training_frame: pd.DataFrame,
    checkpoints: List[int],
    target_col: str = "monthly_value",
) -> dict:
    """Train one pooled XGBoost model per working-day checkpoint."""
    models = {}
    for checkpoint in checkpoints:
        frame = training_frame[training_frame["checkpoint"] == checkpoint].copy()
        if frame.empty:
            continue
        try:
            models[int(checkpoint)] = train_xgboost_checkpoint(frame, target_col)
        except (ValueError, TypeError) as exc:
            logger.warning("Checkpoint XGBoost unavailable for WD%s: %s", checkpoint, exc)
    return models


def predict_xgboost(model: XGBRegressor, features_row: pd.DataFrame) -> Optional[float]:
    """Predict one row; return None when required features are unavailable."""
    cols = getattr(model, "_feature_cols_used", FEATURE_COLS)
    if not all(c in features_row.columns for c in cols):
        return None
    values = features_row[cols]
    if values.isna().any(axis=None):
        return None
    pred = float(model.predict(values)[0])
    return max(pred, 0.0) if np.isfinite(pred) else None
