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


def _next_month_xgb_features(hist_features, current_features, calendar, targets, current_month):
    """Construct current-month checkpoint features using MTD plus closed history.

    The historical feature row for ``current_month`` is already leakage-safe:
    build_historical_features() creates its lag/rolling/seasonal features from
    months strictly before the current month. Using the latest *closed* row
    instead would shift every lag back by one month (for example, lag_1 would
    become two-months-ago), producing the wrong XGBoost input.
    """
    current = current_features.copy()
    target = targets[targets["periode"] == current_month][GROUP_COLS + ["target_sellin"]].drop_duplicates()

    # IMPORTANT: use the current-month historical-feature row. Its features
    # are generated with shift/rolling operations, so the current target is
    # not included and leakage is avoided.
    base = hist_features[hist_features["periode"] == current_month].copy()
    base = base.sort_values("periode").drop_duplicates(GROUP_COLS, keep="last")

    base = base.merge(current[GROUP_COLS + [
        "mtd_value", "elapsed_working_days", "remaining_working_days",
        "total_working_days", "avg_daily_rate", "run_rate_forecast",
    ]], on=GROUP_COLS, how="left", validate="one_to_one")
    base = base.merge(target, on=GROUP_COLS, how="left", validate="one_to_one")
    base["mtd_target_pct"] = base["mtd_value"] / base["target_sellin"].replace(0, np.nan) * 100.0
    base["checkpoint"] = base["elapsed_working_days"].astype(int)
    base["periode"] = current_month
    return base


def _active_checkpoint(elapsed_working_days, checkpoints):
    valid = [cp for cp in checkpoints if cp <= int(elapsed_working_days)]
    return max(valid) if valid else None


def run_prediction_pipeline(trained):
    daily = trained["daily"]
    monthly = trained["monthly"]
    calendar = trained["calendar"]
    hist_features = trained["hist_features"]
    xgb_model = trained["xgb_model"]
    xgb_checkpoint_models = trained.get("xgb_checkpoint_models", {})
    targets = trained["targets"]
    best_models = trained.get("best_models", pd.DataFrame())
    backtest_results = trained.get("backtest_results", pd.DataFrame())
    current_month = pd.Timestamp.today().normalize().replace(day=1)

    cur_features = build_current_month_features(daily, calendar, GROUP_COLS)
    cur = forecast_baseline(cur_features)
    closed_monthly = monthly[monthly["periode"] < current_month].copy()
    ts_rows = []
    for key, group in closed_monthly.groupby(GROUP_COLS):
        key_vals = key if isinstance(key, tuple) else (key,)
        series = group.sort_values("periode")["monthly_value"].astype(float)
        ts_rows.append({
            **{c: v for c, v in zip(GROUP_COLS, key_vals)},
            "forecast_ets": forecast_ets(series),
            "forecast_sarima": forecast_sarima(series),
        })
    ts_forecasts = pd.DataFrame(ts_rows)

    xgb_features = _next_month_xgb_features(hist_features, cur_features, calendar, targets, current_month)
    xgb_rows = []
    checkpoints = FORECAST_CONFIG.get("backtest_checkpoints", [4, 7, 10, 15, 20])
    for _, row in xgb_features.iterrows():
        checkpoint = _active_checkpoint(row["elapsed_working_days"], checkpoints)
        model = xgb_checkpoint_models.get(checkpoint) if checkpoint is not None else None
        if model is None:
            model = xgb_model
            feature_row = row
        else:
            feature_row = row.copy()
            feature_row["checkpoint"] = checkpoint
        pred = predict_xgboost(model, pd.DataFrame([feature_row]))
        if pred is None:
            logger.warning(
                "XGBoost prediction unavailable for %s at WD%s; required feature values are missing.",
                row.get(GROUP_COLS[0]), checkpoint,
            )
        xgb_rows.append({
            **{c: row[c] for c in GROUP_COLS},
            "forecast_xgboost": pred,
        })
    xgb_df = pd.DataFrame(xgb_rows)

    result = cur.merge(ts_forecasts, on=GROUP_COLS, how="left").merge(xgb_df, on=GROUP_COLS, how="left")
    if not best_models.empty:
        result = result.merge(best_models, on=GROUP_COLS, how="left")
    if not backtest_results.empty:
        calibration_cols = GROUP_COLS + [
            f"{model}_{suffix}"
            for model in ("baseline", "ets", "sarima", "xgboost")
            for suffix in ("residual_q10", "residual_q90")
            if f"{model}_{suffix}" in backtest_results.columns
        ]
        for model in ("baseline", "ets", "sarima", "xgboost"):
            for checkpoint in checkpoints:
                for suffix in ("residual_q10", "residual_q90", "residual_observations"):
                    col = f"{model}_wd{checkpoint}_{suffix}"
                    if col in backtest_results.columns:
                        calibration_cols.append(col)
        if len(calibration_cols) > len(GROUP_COLS):
            result = result.merge(
                backtest_results[calibration_cols].drop_duplicates(GROUP_COLS),
                on=GROUP_COLS,
                how="left",
            )

    result = build_ensemble(result)
    current_targets = targets[targets["periode"] == current_month]
    result = result.merge(
        current_targets[GROUP_COLS + ["target_sellin"]].drop_duplicates(),
        on=GROUP_COLS,
        how="left",
    )
    result["achievement_pct_forecast"] = result["forecast_p50"] / result["target_sellin"].replace(0, pd.NA) * 100
    result["periode"] = current_month
    result["generated_at"] = pd.Timestamp.now()
    keep = GROUP_COLS + [
        "periode", "mtd_value", "elapsed_working_days", "remaining_working_days",
        "total_working_days", "forecast_baseline", "forecast_ets", "forecast_sarima",
        "forecast_xgboost", "forecast_p10", "forecast_p50", "forecast_p90",
        "target_sellin", "achievement_pct_forecast", "generated_at",
    ]
    return result[[c for c in keep if c in result.columns]]
