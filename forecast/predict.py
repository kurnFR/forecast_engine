"""Generate the current region-month end-of-month forecast."""
import logging
import numpy as np
import pandas as pd

from config import FORECAST_CONFIG
from features.current_month import build_current_month_features
from features.working_day import month_working_day_stats
from models.baseline import forecast_baseline
from models.ensemble import build_ensemble
from models.ets import forecast_ets
from models.sarima import forecast_sarima
from models.xgboost_model import predict_xgboost

logger = logging.getLogger(__name__)
GROUP_COLS = FORECAST_CONFIG["grain"]


def _next_month_xgb_features(hist_features, current_features, calendar, targets, current_month):
    """Construct current-month checkpoint features using MTD plus closed history.

    The historical feature row for ``current_month`` is leakage-safe because
    build_historical_features() creates its lag/rolling/seasonal features from
    months strictly before the current month.
    """
    current = current_features.copy()
    target = targets[targets["periode"] == current_month][GROUP_COLS + ["target_sellin"]].drop_duplicates()

    base = hist_features[hist_features["periode"] == current_month].copy()
    base = base.sort_values("periode").drop_duplicates(GROUP_COLS, keep="last")

    base = base.merge(current[GROUP_COLS + [
        "mtd_value", "elapsed_working_days", "remaining_working_days",
        "total_working_days", "avg_daily_rate", "run_rate_forecast",
    ]], on=GROUP_COLS, how="left", validate="one_to_one")
    base = base.merge(target, on=GROUP_COLS, how="left", validate="one_to_one")

    # The working-day calendar is a month-level feature, not a region-level
    # feature.  Re-derive it directly from the authoritative calendar here so
    # an incomplete/partially populated current_features frame can never make
    # XGBoost lose total_working_days for every region.
    wd = month_working_day_stats(calendar, pd.Timestamp(current_month))
    base["total_working_days"] = pd.to_numeric(
        base["total_working_days"], errors="coerce"
    ).fillna(float(wd["total_working_days"]))
    base["elapsed_working_days"] = pd.to_numeric(
        base["elapsed_working_days"], errors="coerce"
    ).fillna(float(wd["elapsed_working_days"]))
    base["remaining_working_days"] = pd.to_numeric(
        base["remaining_working_days"], errors="coerce"
    ).fillna(float(wd["remaining_working_days"]))

    if base["total_working_days"].isna().any() or (base["total_working_days"] <= 0).any():
        raise ValueError(
            f"Invalid total_working_days for {current_month:%Y-%m}; "
            "cannot build XGBoost prediction features."
        )

    base["mtd_target_pct"] = base["mtd_value"] / base["target_sellin"].replace(0, np.nan) * 100.0
    base["checkpoint"] = base["elapsed_working_days"].astype(int)
    base["periode"] = current_month
    return base


def _active_checkpoint(elapsed_working_days, checkpoints):
    valid = [cp for cp in checkpoints if cp <= int(elapsed_working_days)]
    return max(valid) if valid else None


def _missing_features(model, row):
    """Return required XGBoost features that are missing or null."""
    cols = getattr(model, "_feature_cols_used", [])
    missing = [c for c in cols if c not in row.index or pd.isna(row.get(c))]
    return missing


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
        checkpoint_model = xgb_checkpoint_models.get(checkpoint) if checkpoint is not None else None
        pred = None

        if checkpoint_model is not None:
            feature_row = row.copy()
            feature_row["checkpoint"] = checkpoint
            pred = predict_xgboost(checkpoint_model, pd.DataFrame([feature_row]))
            if pred is None:
                missing = _missing_features(checkpoint_model, feature_row)
                logger.warning(
                    "Checkpoint XGBoost unavailable for %s at WD%s; missing features: %s. Falling back to pooled XGBoost.",
                    row.get(GROUP_COLS[0]), checkpoint, ", ".join(missing) if missing else "prediction failure",
                )

        if pred is None:
            pooled_missing = _missing_features(xgb_model, row)
            pred = predict_xgboost(xgb_model, pd.DataFrame([row]))
            if pred is None:
                logger.warning(
                    "Pooled XGBoost prediction unavailable for %s; missing features: %s.",
                    row.get(GROUP_COLS[0]), ", ".join(pooled_missing) if pooled_missing else "prediction failure",
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
