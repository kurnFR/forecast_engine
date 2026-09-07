"""Generate the current-month end-of-month forecast using the ensemble of models."""
import logging
import pandas as pd

from features.current_month import build_current_month_features
from models.baseline import forecast_baseline
from models.ets import forecast_ets
from models.sarima import forecast_sarima
from models.xgboost_model import predict_xgboost
from models.ensemble import build_ensemble
from config import FORECAST_CONFIG

logger = logging.getLogger(__name__)

GROUP_COLS = FORECAST_CONFIG["grain"]


def run_prediction_pipeline(trained: dict) -> pd.DataFrame:
    daily, monthly, calendar = trained["daily"], trained["monthly"], trained["calendar"]
    hist_features, xgb_model = trained["hist_features"], trained["xgb_model"]
    branch_targets = trained["branch_targets"]

    cur = build_current_month_features(daily, calendar, GROUP_COLS)
    cur = forecast_baseline(cur)

    ts_rows = []
    for key, g in monthly.groupby(GROUP_COLS):
        key_vals = key if isinstance(key, tuple) else (key,)
        g = g.sort_values("periode")
        series = g["monthly_value"]

        row = dict(zip(GROUP_COLS, key_vals))
        row["forecast_ets"] = forecast_ets(series)
        row["forecast_sarima"] = forecast_sarima(series)
        ts_rows.append(row)
    ts_forecasts = pd.DataFrame(ts_rows)

    latest_features = (
        hist_features.sort_values("periode").groupby(GROUP_COLS, as_index=False).tail(1)
    )
    xgb_rows = []
    for _, r in latest_features.iterrows():
        row_df = pd.DataFrame([r])
        pred = predict_xgboost(xgb_model, row_df)
        xgb_rows.append({**{c: r[c] for c in GROUP_COLS}, "forecast_xgboost": pred})
    xgb_df = pd.DataFrame(xgb_rows)

    merged = (
        cur.merge(ts_forecasts, on=GROUP_COLS, how="left")
        .merge(xgb_df, on=GROUP_COLS, how="left")
    )

    result = build_ensemble(merged)

    result = result.merge(
        branch_targets[GROUP_COLS + ["branch_target_sellin"]].drop_duplicates(),
        on=GROUP_COLS, how="left",
    )
    result["achievement_pct_forecast"] = (
        result["forecast_ensemble"] / result["branch_target_sellin"].replace(0, pd.NA) * 100
    )
    result["periode"] = pd.Timestamp.today().normalize().replace(day=1)
    result["generated_at"] = pd.Timestamp.now()

    keep = GROUP_COLS + [
        "periode", "mtd_value", "elapsed_working_days", "remaining_working_days",
        "total_working_days", "forecast_baseline", "forecast_ets", "forecast_sarima",
        "forecast_xgboost", "forecast_ensemble", "branch_target_sellin",
        "achievement_pct_forecast", "generated_at",
    ]
    return result[[c for c in keep if c in result.columns]]
