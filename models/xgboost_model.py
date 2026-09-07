"""Leakage-safe XGBoost model for region-month Sell-In forecasting.

The model is trained on historical rows whose features describe only information
available before the forecast month. Working-day checkpoint backtesting can
construct checkpoint-specific feature rows from the same feature builder.
"""
import logging
from typing import Optional

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


def _training_frame(train_df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in FEATURE_COLS if c in train_df.columns]
    if not cols:
        raise ValueError("No XGBoost feature columns are present.")
    d = train_df.dropna(subset=cols + [target_col]).copy()
    if d.empty:
        raise ValueError("No rows left after dropping NaNs - not enough history to train XGBoost.")
    return d, cols


def train_xgboost(train_df: pd.DataFrame, target_col: str = "monthly_value") -> XGBRegressor:
    """Fit one pooled region-month model using only supplied historical rows."""
    d, cols = _training_frame(train_df, target_col)
    model = XGBRegressor(**MODEL_CONFIG["xgboost"])
    model.fit(d[cols], d[target_col])
    model._feature_cols_used = cols
    return model


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
