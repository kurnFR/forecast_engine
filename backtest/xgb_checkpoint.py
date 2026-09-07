"""Leakage-safe XGBoost predictions at working-day checkpoints."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from backtest.metrics import bias, bias_pct, mae, rmse, wape
from features.historical import build_historical_features
from models.xgboost_model import train_xgboost, predict_xgboost
logger = logging.getLogger(__name__)

def _as_key(value):
    return value if isinstance(value, tuple) else (value,)

def _checkpoint_date(calendar, periode, checkpoint):
    month = calendar[(calendar["date"] >= periode) & (calendar["date"] < periode + pd.offsets.MonthBegin(1)) & calendar["is_working_day"].astype(bool)].sort_values("date")
    return month.iloc[checkpoint - 1]["date"] if len(month) >= checkpoint else None

def _residual_stats(pairs):
    if not pairs:
        return np.nan, np.nan, 0
    residuals = np.asarray([a - p for a, p in pairs], dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if not len(residuals):
        return np.nan, np.nan, 0
    return float(np.quantile(residuals, .10)), float(np.quantile(residuals, .90)), len(residuals)

def _xgb_checkpoint_features(monthly_history, calendar, group_cols, target_month, min_train_months):
    history = monthly_history[monthly_history["periode"] < target_month].copy()
    if history.empty:
        return pd.DataFrame()
    rows = []
    for key, g in history.groupby(group_cols):
        key_vals = _as_key(key)
        g = g.sort_values("periode")
        if len(g) < min_train_months:
            continue
        values = g["monthly_value"].astype(float).tolist()
        row = dict(zip(group_cols, key_vals)); row["periode"] = target_month
        for lag in range(1, 7): row[f"lag_{lag}"] = values[-lag] if len(values) >= lag else np.nan
        row["mom_growth"] = values[-1] / values[-2] - 1.0 if len(values) >= 2 and values[-2] != 0 else np.nan
        recent = values[-3:]
        row["rolling_mean_3"] = np.mean(recent) if recent else np.nan
        row["rolling_std_3"] = np.std(recent, ddof=1) if len(recent) >= 2 else np.nan
        same_month = g.loc[g["periode"].dt.month == target_month.month, "monthly_value"].astype(float)
        overall_mean = np.mean(values) if values else 0.0
        row["seasonal_index"] = float(same_month.mean()) / overall_mean if len(same_month) and overall_mean != 0 else np.nan
        row["calendar_month"] = target_month.month
        row["total_working_days"] = int(calendar.loc[(calendar["date"] >= target_month) & (calendar["date"] < target_month + pd.offsets.MonthBegin(1)), "is_working_day"].astype(bool).sum())
        rows.append(row)
    return pd.DataFrame(rows)

def rolling_xgb_checkpoint_backtest(monthly_history, daily_history, calendar, group_cols, checkpoints, min_train_months=24):
    """Evaluate XGBoost at WD checkpoints and retain checkpoint residual quantiles."""
    monthly = monthly_history.copy(); monthly["periode"] = pd.to_datetime(monthly["periode"])
    daily = daily_history.copy(); daily["invoice_date"] = pd.to_datetime(daily["invoice_date"])
    calendar = calendar.copy(); calendar["date"] = pd.to_datetime(calendar["date"])
    pair_store = {_as_key(key): {cp: [] for cp in checkpoints} for key in monthly[group_cols].drop_duplicates().itertuples(index=False, name=None)}

    for target_month in sorted(monthly["periode"].drop_duplicates()):
        train = monthly[monthly["periode"] < target_month].copy()
        eligible_keys = {_as_key(key) for key, g in train.groupby(group_cols) if len(g) >= min_train_months}
        if not eligible_keys: continue
        try:
            model = train_xgboost(build_historical_features(train, group_cols))
        except (ValueError, TypeError) as exc:
            logger.debug("XGBoost unavailable for %s: %s", target_month, exc); continue
        feature_rows = _xgb_checkpoint_features(monthly, calendar, group_cols, target_month, min_train_months)
        if feature_rows.empty: continue
        actuals = monthly.loc[monthly["periode"] == target_month].groupby(group_cols)["monthly_value"].sum()
        actual_map = {_as_key(k): float(v) for k, v in actuals.items()}
        for cp in checkpoints:
            if _checkpoint_date(calendar, target_month, cp) is None: continue
            for key, row in feature_rows.groupby(group_cols):
                key = _as_key(key)
                if key not in eligible_keys or key not in actual_map: continue
                pred = predict_xgboost(model, row)
                if pred is not None and np.isfinite(pred): pair_store[key][cp].append((actual_map[key], float(pred)))

    results = []
    for key, cp_pairs in pair_store.items():
        out = dict(zip(group_cols, key)); all_pairs = [p for pairs in cp_pairs.values() for p in pairs]
        for cp in checkpoints:
            pairs = cp_pairs[cp]
            for metric in ("wape", "mae", "rmse", "bias", "bias_pct"):
                out[f"xgboost_wd{cp}_{metric}"] = np.nan if not pairs else {"wape": wape, "mae": mae, "rmse": rmse, "bias": bias, "bias_pct": bias_pct}[metric](*zip(*pairs))
            out[f"xgboost_wd{cp}_observations"] = len(pairs)
            q10, q90, count = _residual_stats(pairs)
            out[f"xgboost_wd{cp}_residual_q10"] = q10
            out[f"xgboost_wd{cp}_residual_q90"] = q90
            out[f"xgboost_wd{cp}_residual_observations"] = count
        if all_pairs:
            y, p = zip(*all_pairs)
            out.update({"xgboost_wape": wape(y,p), "xgboost_mae": mae(y,p), "xgboost_rmse": rmse(y,p), "xgboost_bias": bias(y,p), "xgboost_bias_pct": bias_pct(y,p), "xgboost_observations": len(all_pairs)})
            q10, q90, _ = _residual_stats(all_pairs); out["xgboost_residual_q10"] = q10; out["xgboost_residual_q90"] = q90
        else:
            for metric in ("wape", "mae", "rmse", "bias", "bias_pct"): out[f"xgboost_{metric}"] = np.nan
            out["xgboost_observations"] = 0; out["xgboost_residual_q10"] = np.nan; out["xgboost_residual_q90"] = np.nan
        results.append(out)
    return pd.DataFrame(results)
