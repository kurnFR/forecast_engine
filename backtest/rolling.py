"""Leakage-safe working-day checkpoint backtesting for monthly forecasts."""
import logging

import numpy as np
import pandas as pd

from backtest.metrics import bias, bias_pct, mae, rmse, wape
from models.baseline import forecast_from_mtd
from models.ets import forecast_ets
from models.sarima import forecast_sarima
from backtest.xgb_checkpoint import rolling_xgb_checkpoint_backtest

logger = logging.getLogger(__name__)


def _checkpoint_date(calendar: pd.DataFrame, periode: pd.Timestamp, checkpoint: int):
    month = calendar[(calendar["date"] >= periode) & (calendar["date"] < periode + pd.offsets.MonthBegin(1)) & calendar["is_working_day"].astype(bool)].sort_values("date")
    if len(month) < checkpoint:
        return None
    return month.iloc[checkpoint - 1]["date"]


def _safe_forecast(fn, series):
    try:
        value = fn(series)
        if value is None or not np.isfinite(float(value)):
            return None
        return max(float(value), 0.0)
    except Exception as exc:
        logger.debug("%s failed during backtest: %s", getattr(fn, "__name__", "model"), exc)
        return None


def _residual_stats(pairs):
    """Return robust out-of-sample residual quantiles for interval calibration."""
    if not pairs:
        return np.nan, np.nan, 0
    residuals = np.asarray([actual - pred for actual, pred in pairs], dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if not len(residuals):
        return np.nan, np.nan, 0
    return float(np.quantile(residuals, 0.10)), float(np.quantile(residuals, 0.90)), len(residuals)


def rolling_backtest(monthly_history, group_cols, min_train_months=24, daily_history=None, calendar=None, checkpoints=None):
    """Evaluate candidate models at WD checkpoints with residual calibration."""
    checkpoints = checkpoints or [4, 7, 10, 15, 20]
    if daily_history is None or calendar is None:
        raise ValueError("V2 backtest requires daily_history and calendar for WD checkpoints.")
    required = set(group_cols + ["periode", "monthly_value"])
    if not required.issubset(monthly_history.columns):
        raise ValueError(f"monthly_history missing columns: {required - set(monthly_history.columns)}")

    monthly = monthly_history.copy()
    monthly["periode"] = pd.to_datetime(monthly["periode"])
    daily = daily_history.copy()
    daily["invoice_date"] = pd.to_datetime(daily["invoice_date"])
    daily["periode"] = daily["invoice_date"].dt.to_period("M").dt.to_timestamp()
    calendar = calendar.copy()
    calendar["date"] = pd.to_datetime(calendar["date"])

    rows = []
    for key, group in monthly.groupby(group_cols):
        key_vals = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(group_cols, key_vals))
        g = group.sort_values("periode").reset_index(drop=True)
        observations = {model: [] for model in ("baseline", "ets", "sarima")}
        checkpoint_values = {model: {cp: [] for cp in checkpoints} for model in observations}

        for target_month in g["periode"].drop_duplicates().sort_values():
            train = g[g["periode"] < target_month]
            if len(train) < min_train_months:
                continue
            actual = g.loc[g["periode"] == target_month, "monthly_value"].sum()
            target_daily = daily[(daily["periode"] == target_month) & daily[group_cols].eq(pd.Series(key_vals, index=group_cols)).all(axis=1)]
            if target_daily.empty:
                continue
            series = train["monthly_value"].astype(float)
            ts_preds = {"ets": _safe_forecast(forecast_ets, series), "sarima": _safe_forecast(forecast_sarima, series)}

            for checkpoint in checkpoints:
                cp_date = _checkpoint_date(calendar, target_month, checkpoint)
                if cp_date is None:
                    continue
                elapsed = int(calendar.loc[(calendar["date"] >= target_month) & (calendar["date"] <= cp_date), "is_working_day"].astype(bool).sum())
                total = int(calendar.loc[(calendar["date"] >= target_month) & (calendar["date"] < target_month + pd.offsets.MonthBegin(1)), "is_working_day"].astype(bool).sum())
                mtd = target_daily.loc[target_daily["invoice_date"] <= cp_date, "sellin_value"].sum()
                preds = {"baseline": forecast_from_mtd(float(mtd), elapsed, total), "ets": ts_preds["ets"], "sarima": ts_preds["sarima"]}
                for model, pred in preds.items():
                    if pred is not None and np.isfinite(float(pred)):
                        pair = (float(actual), float(pred))
                        observations[model].append(pair)
                        checkpoint_values[model][checkpoint].append(pair)

        row = dict(key_dict)
        for model, pairs in observations.items():
            if not pairs:
                for metric in ("wape", "mae", "rmse", "bias", "bias_pct"):
                    row[f"{model}_{metric}"] = np.nan
                row[f"{model}_observations"] = 0
                row[f"{model}_residual_q10"] = np.nan
                row[f"{model}_residual_q90"] = np.nan
            else:
                y_true, y_pred = zip(*pairs)
                row.update({f"{model}_wape": wape(y_true, y_pred), f"{model}_mae": mae(y_true, y_pred), f"{model}_rmse": rmse(y_true, y_pred), f"{model}_bias": bias(y_true, y_pred), f"{model}_bias_pct": bias_pct(y_true, y_pred), f"{model}_observations": len(pairs)})
                q10, q90, _ = _residual_stats(pairs)
                row[f"{model}_residual_q10"] = q10
                row[f"{model}_residual_q90"] = q90
            for checkpoint in checkpoints:
                cp_pairs = checkpoint_values[model][checkpoint]
                row[f"{model}_wd{checkpoint}_wape"] = wape(*zip(*cp_pairs)) if cp_pairs else np.nan
                q10, q90, count = _residual_stats(cp_pairs)
                row[f"{model}_wd{checkpoint}_residual_q10"] = q10
                row[f"{model}_wd{checkpoint}_residual_q90"] = q90
                row[f"{model}_wd{checkpoint}_residual_observations"] = count
        rows.append(row)

    base_results = pd.DataFrame(rows)
    xgb_results = rolling_xgb_checkpoint_backtest(monthly_history=monthly, daily_history=daily, calendar=calendar, group_cols=group_cols, checkpoints=checkpoints, min_train_months=min_train_months)
    if base_results.empty:
        return xgb_results
    return base_results.merge(xgb_results, on=group_cols, how="left")
