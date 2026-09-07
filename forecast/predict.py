"""Generate the current region-month end-of-month forecast."""
import logging

import numpy as np
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


def _next_month_xgb_features(hist_features: pd.DataFrame, current_month: pd.Timestamp) -> pd.DataFrame:
    """Construct a feature row for current_month from closed history only."""
    rows = []
    for key, group in hist_features[hist_features["periode"] < current_month].groupby(GROUP_COLS):
        key_vals = key if isinstance(key, tuple) else (key,)
        g = group.sort_values("periode").copy()
        if g.empty:
            continue
        latest = g.iloc[-1].copy()
        values = g["monthly_value"].astype(float).tail(3).tolist()
        previous = g["monthly_value"].astype(float).tail(4).tolist()

        latest["calendar_month"] = current_month.month
        for lag in range(1, 7):
            latest[f"lag_{lag}"] = (
                previous[-lag] if len(previous) >= lag else np.nan
            )
        if len(previous) >= 2 and previous[-2] != 0:
            latest["mom_growth"] = previous[-1] / previous[-2] - 1.0
        else:
            latest["mom_growth"] = np.nan
        latest["rolling_mean_3"] = np.mean(values) if values else np.nan
        latest["rolling_std_3"] = np.std(values, ddof=1) if len(values) >= 2 else np.nan
        rows.append({**{c: latest[c] for c in GROUP_COLS}, **latest.to_dict()})

    return pd.DataFrame(rows)


def run_prediction_pipeline(trained: dict) -> pd.DataFrame:
    daily = trained["daily"]
    monthly = trained["monthly"]
    calendar = trained["calendar"]
    hist_features = trained["hist_features"]
    xgb_model = trained["xgb_model"]
    targets = trained["targets"]

    current_month = pd.Timestamp.today().normalize().replace(day=1)

    # Only the baseline consumes current-MTD actuals. Historical models use
    # closed months exclusively, preventing current-month leakage.
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

    xgb_features = _next_month_xgb_features(hist_features, current_month)
    xgb_rows = []
    for _, row in xgb_features.iterrows():
        pred = predict_xgboost(xgb_model, pd.DataFrame([row]))
        xgb_rows.append({**{c: row[c] for c in GROUP_COLS}, "forecast_xgboost": pred})
    xgb_df = pd.DataFrame(xgb_rows)

    result = (
        cur.merge(ts_forecasts, on=GROUP_COLS, how="left")
        .merge(xgb_df, on=GROUP_COLS, how="left")
    )
    result = build_ensemble(result)

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
        "periode", "mtd_value", "elapsed_working_days", "remaining_working_days",
        "total_working_days", "forecast_baseline", "forecast_ets", "forecast_sarima",
        "forecast_xgboost", "forecast_p10", "forecast_p50", "forecast_p90",
        "target_sellin", "achievement_pct_forecast", "generated_at",
    ]
    return result[[c for c in keep if c in result.columns]]
