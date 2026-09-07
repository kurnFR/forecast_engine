"""Generate the current region-month end-of-month forecast."""
import logging

import pandas as pd

from config import FORECAST_CONFIG
from features.current_month import build_current_month_features
from models.baseline import forecast_baseline
from models.ensemble import build_ensemble
from models.ets import forecast_ets
from models.sarima import forecast_sarima
from models.xgboost_model import predict_xgboost

logger = logging.getLogger(__name__)
GROUP_COLS = FORECAST_CONFIG["grain"]


def run_prediction_pipeline(trained: dict) -> pd.DataFrame:
    daily = trained["daily"]
    monthly = trained["monthly"]
    calendar = trained["calendar"]
    hist_features = trained["hist_features"]
    xgb_model = trained["xgb_model"]
    targets = trained["targets"]

    current_month = pd.Timestamp.today().normalize().replace(day=1)

    # Baseline is the only candidate that intentionally consumes current-MTD
    # actuals.  All historical models below use closed months only.
    cur = build_current_month_features(daily, calendar, GROUP_COLS)
    cur = forecast_baseline(cur)

    closed_monthly = monthly[monthly["periode"] < current_month].copy()
    ts_rows = []
    for key, group in closed_monthly.groupby(GROUP_COLS):
        key_vals = key if isinstance(key, tuple) else (key,)
        series = group.sort_values("periode")["monthly_value"].astype(float)
        row = dict(zip(GROUP_COLS, key_vals))
        row["forecast_ets"] = forecast_ets(series)
        row["forecast_sarima"] = forecast_sarima(series)
        ts_rows.append(row)
    ts_forecasts = pd.DataFrame(ts_rows)

    # XGBoost was trained to predict a month from lagged history.  Use the
    # latest CLOSED feature row and move its calendar month one month forward;
    # never use the current partial month as an ML target or feature.
    latest = (
        hist_features[hist_features["periode"] < current_month]
        .sort_values("periode")
        .groupby(GROUP_COLS, as_index=False)
        .tail(1)
        .copy()
    )
    latest["calendar_month"] = (current_month.month)

    xgb_rows = []
    for _, row in latest.iterrows():
        feature_row = pd.DataFrame([row])
        pred = predict_xgboost(xgb_model, feature_row)
        xgb_rows.append({**{c: row[c] for c in GROUP_COLS}, "forecast_xgboost": pred})
    xgb_df = pd.DataFrame(xgb_rows)

    result = (
        cur.merge(ts_forecasts, on=GROUP_COLS, how="left")
        .merge(xgb_df, on=GROUP_COLS, how="left")
    )
    result = build_ensemble(result)

    # Authoritative region-level target: no branch allocation or inferred
    # target is permitted in V2.
    current_targets = targets[targets["periode"] == current_month]
    result = result.merge(
        current_targets[GROUP_COLS + ["target_sellin"]].drop_duplicates(),
        on=GROUP_COLS,
        how="left",
    )
    result["achievement_pct_forecast"] = (
        result["forecast_p50"]
        / result["target_sellin"].replace(0, pd.NA)
        * 100
    )
    result["periode"] = current_month
    result["generated_at"] = pd.Timestamp.now()

    keep = GROUP_COLS + [
        "periode",
        "mtd_value",
        "elapsed_working_days",
        "remaining_working_days",
        "total_working_days",
        "forecast_baseline",
        "forecast_ets",
        "forecast_sarima",
        "forecast_xgboost",
        "forecast_p10",
        "forecast_p50",
        "forecast_p90",
        "target_sellin",
        "achievement_pct_forecast",
        "generated_at",
    ]
    return result[[c for c in keep if c in result.columns]]
