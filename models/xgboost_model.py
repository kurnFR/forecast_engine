"""XGBoost regressor trained across all (region, branch) series at once,
using the engineered lag / seasonality / working-day features."""
import logging
from typing import Optional
import pandas as pd
from xgboost import XGBRegressor

from config import MODEL_CONFIG

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "mom_growth", "rolling_mean_3",
    "rolling_std_3", "seasonal_index", "calendar_month",
]


def train_xgboost(train_df: pd.DataFrame, target_col: str = "monthly_value") -> XGBRegressor:
    cols_present = [c for c in FEATURE_COLS if c in train_df.columns]
    d = train_df.dropna(subset=cols_present + [target_col])
    if d.empty:
        raise ValueError("No rows left after dropping NaNs - not enough history to train XGBoost.")
    model = XGBRegressor(**MODEL_CONFIG["xgboost"])
    model.fit(d[cols_present], d[target_col])
    model._feature_cols_used = cols_present  # remember for prediction time
    return model


def predict_xgboost(model: XGBRegressor, features_row: pd.DataFrame) -> Optional[float]:
    cols = getattr(model, "_feature_cols_used", FEATURE_COLS)
    cols = [c for c in cols if c in features_row.columns]
    if not cols or features_row[cols].isna().any(axis=None):
        return None
    return max(float(model.predict(features_row[cols])[0]), 0.0)
