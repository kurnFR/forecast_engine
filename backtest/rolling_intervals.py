"""Strictly leakage-safe rolling P10/P50/P90 backtesting."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from backtest.metrics import wape
from backtest.xgb_checkpoint import _checkpoint_date as xgb_checkpoint_date
from backtest.xgb_checkpoint import _xgb_checkpoint_features
from features.historical import build_historical_features
from models.baseline import forecast_from_mtd
from models.ensemble import CANDIDATE_MODELS
from models.ets import forecast_ets
from models.sarima import forecast_sarima
from models.xgboost_model import predict_xgboost, train_xgboost

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
    """Build P10/P50/P90 using calibration rows strictly before the target."""
    del checkpoint
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
    p10 = max(p50 + float(np.dot(weights_arr, lower)), 0.0)
    p90 = max(p50 + float(np.dot(weights_arr, upper)), 0.0)
    return (min(p10, p90), p50, max(p10, p90))


def build_oos_checkpoint_predictions(monthly_history, daily_history, calendar, group_cols, checkpoints, min_train_months=24):
    """Build raw candidate OOS predictions at each target/checkpoint."""
    monthly = monthly_history.copy(); monthly["periode"] = pd.to_datetime(monthly["periode"])
    daily = daily_history.copy(); daily["invoice_date"] = pd.to_datetime(daily["invoice_date"])
    daily["periode"] = daily["invoice_date"].dt.to_period("M").dt.to_timestamp()
    calendar = calendar.copy(); calendar["date"] = pd.to_datetime(calendar["date"])
    rows = []

    for key, group in monthly.groupby(group_cols):
        key_vals = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(group_cols, key_vals))
        g = group.sort_values("periode").reset_index(drop=True)
        for target_month in g["periode"].drop_duplicates().sort_values():
            train = g[g["periode"] < target_month]
            if len(train) < min_train_months:
                continue
            target_daily = daily[(daily["periode"] == target_month) & daily[group_cols].eq(pd.Series(key_vals, index=group_cols)).all(axis=1)]
            if target_daily.empty:
                continue
            actual = float(g.loc[g["periode"] == target_month, "monthly_value"].sum())
            series = train["monthly_value"].astype(float)
            ts_preds = {"ets": _safe_forecast(forecast_ets, series), "sarima": _safe_forecast(forecast_sarima, series)}
            xgb_pred = None
            try:
                xgb_train = train_xgboost(build_historical_features(train, group_cols))
                xgb_features = _xgb_checkpoint_features(monthly, calendar, group_cols, target_month, min_train_months)
                target_features = xgb_features[xgb_features[group_cols].eq(pd.Series(key_vals, index=group_cols)).all(axis=1)]
                if not target_features.empty:
                    xgb_pred = predict_xgboost(xgb_train, target_features)
            except (ValueError, TypeError) as exc:
                logger.debug("XGBoost unavailable for %s/%s: %s", target_month, key_vals, exc)

            month_mask = (calendar["date"] >= target_month) & (calendar["date"] < target_month + pd.offsets.MonthBegin(1))
            total = int(calendar.loc[month_mask, "is_working_day"].astype(bool).sum())
            for checkpoint in checkpoints:
                cp_date = _checkpoint_date(calendar, target_month, checkpoint)
                if cp_date is None:
                    continue
                elapsed = int(calendar.loc[month_mask & (calendar["date"] <= cp_date), "is_working_day"].astype(bool).sum())
                mtd = float(target_daily.loc[target_daily["invoice_date"] <= cp_date, "sellin_value"].sum())
                preds = {"baseline": forecast_from_mtd(mtd, elapsed, total), "ets": ts_preds["ets"], "sarima": ts_preds["sarima"], "xgboost": xgb_pred}
                row = dict(key_dict); row.update({"periode": target_month, "checkpoint": checkpoint, "actual": actual})
                for model in CANDIDATE_MODELS:
                    value = preds.get(model)
                    row[f"forecast_{model}"] = float(value) if value is not None and np.isfinite(float(value)) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def rolling_interval_backtest(monthly_history, daily_history, calendar, group_cols, checkpoints, min_train_months=24):
    """Evaluate intervals with target-month calibration restricted to prior months."""
    raw = build_oos_checkpoint_predictions(monthly_history, daily_history, calendar, group_cols, checkpoints, min_train_months)
    if raw.empty:
        return pd.DataFrame()
    scored = []
    for key, target in raw.groupby(group_cols + ["periode", "checkpoint"], dropna=False):
        key_vals = key if isinstance(key, tuple) else (key,)
        target_month = key_vals[len(group_cols)]; checkpoint = int(key_vals[len(group_cols) + 1])
        prior = raw[(raw["periode"] < target_month) & (raw["checkpoint"] == checkpoint)]
        for col, value in zip(group_cols, key_vals[:len(group_cols)]):
            prior = prior[prior[col] == value]
        p10, p50, p90 = _ensemble_with_prior_calibration(target.iloc[0], prior, checkpoint)
        row = {col: value for col, value in zip(group_cols, key_vals[:len(group_cols)])}
        row.update({"periode": target_month, "checkpoint": checkpoint, "actual": float(target.iloc[0]["actual"]), "forecast_p10": p10, "forecast_p50": p50, "forecast_p90": p90, "calibration_observations": int(len(prior)), "calibration_cutoff": target_month})
        scored.append(row)
    result = pd.DataFrame(scored)
    if result.empty:
        return result
    result = result[result["forecast_p10"].notna() & result["forecast_p90"].notna()].copy()
    if result.empty:
        return result
    result["interval_width"] = result["forecast_p90"] - result["forecast_p10"]
    result["interval_width_pct_actual"] = result["interval_width"] / result["actual"].abs().replace(0, np.nan)
    return result
