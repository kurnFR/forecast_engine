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


def _checkpoint_date(calendar: pd.DataFrame, periode: pd.Timestamp, checkpoint: int):
    month = calendar[
        (calendar["date"] >= periode)
        & (calendar["date"] < periode + pd.offsets.MonthBegin(1))
        & calendar["is_working_day"].astype(bool)
    ].sort_values("date")
    if len(month) < checkpoint:
        return None
    return month.iloc[checkpoint - 1]["date"]


def _xgb_checkpoint_features(
    monthly_history: pd.DataFrame,
    calendar: pd.DataFrame,
    group_cols: list[str],
    target_month: pd.Timestamp,
    min_train_months: int,
) -> pd.DataFrame:
    """Build one feature row per eligible region without target-month sales."""
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
        row = dict(zip(group_cols, key_vals))
        row["periode"] = target_month
        for lag in range(1, 7):
            row[f"lag_{lag}"] = values[-lag] if len(values) >= lag else np.nan
        row["mom_growth"] = (
            values[-1] / values[-2] - 1.0 if len(values) >= 2 and values[-2] != 0 else np.nan
        )
        recent = values[-3:]
        row["rolling_mean_3"] = np.mean(recent) if recent else np.nan
        row["rolling_std_3"] = np.std(recent, ddof=1) if len(recent) >= 2 else np.nan
        same_month = g.loc[g["periode"].dt.month == target_month.month, "monthly_value"].astype(float)
        overall_mean = np.mean(values) if values else 0.0
        row["seasonal_index"] = (
            float(same_month.mean()) / overall_mean
            if len(same_month) and overall_mean != 0
            else np.nan
        )
        row["calendar_month"] = target_month.month
        row["total_working_days"] = int(
            calendar.loc[
                (calendar["date"] >= target_month)
                & (calendar["date"] < target_month + pd.offsets.MonthBegin(1)),
                "is_working_day",
            ].astype(bool).sum()
        )
        rows.append(row)

    return pd.DataFrame(rows)


def rolling_xgb_checkpoint_backtest(
    monthly_history: pd.DataFrame,
    daily_history: pd.DataFrame,
    calendar: pd.DataFrame,
    group_cols: list[str],
    checkpoints: list[int],
    min_train_months: int = 24,
) -> pd.DataFrame:
    """Evaluate XGBoost at WD checkpoints with one pooled model per target month.

    Each target-month model is fitted once on all eligible regions using only
    observations strictly before that month. The target-month feature rows use
    closed history plus the target month's calendar only. Daily Sell-In is
    accepted for API compatibility and checkpoint-date validation, but is not
    used as an XGBoost feature in this history-only candidate.
    """
    monthly = monthly_history.copy()
    monthly["periode"] = pd.to_datetime(monthly["periode"])
    daily = daily_history.copy()
    daily["invoice_date"] = pd.to_datetime(daily["invoice_date"])
    calendar = calendar.copy()
    calendar["date"] = pd.to_datetime(calendar["date"])

    target_months = sorted(monthly["periode"].drop_duplicates())
    pair_store = {
        _as_key(key): {cp: [] for cp in checkpoints}
        for key in monthly[group_cols].drop_duplicates().itertuples(index=False, name=None)
    }

    for target_month in target_months:
        train = monthly[monthly["periode"] < target_month].copy()
        eligible_keys = {
            _as_key(key)
            for key, g in train.groupby(group_cols)
            if len(g) >= min_train_months
        }
        if not eligible_keys:
            continue

        train_features = build_historical_features(train, group_cols)
        try:
            model = train_xgboost(train_features)
        except (ValueError, TypeError) as exc:
            logger.debug("XGBoost unavailable for %s: %s", target_month, exc)
            continue

        feature_rows = _xgb_checkpoint_features(
            monthly, calendar, group_cols, target_month, min_train_months
        )
        if feature_rows.empty:
            continue

        actuals = monthly.loc[monthly["periode"] == target_month].groupby(group_cols)["monthly_value"].sum()
        if len(group_cols) == 1:
            actual_map = {(key,): float(value) for key, value in actuals.items()}
        else:
            actual_map = {_as_key(key): float(value) for key, value in actuals.items()}

        for cp in checkpoints:
            cp_date = _checkpoint_date(calendar, target_month, cp)
            if cp_date is None:
                continue
            for key, row in feature_rows.groupby(group_cols):
                key = _as_key(key)
                if key not in eligible_keys:
                    continue
                actual = actual_map.get(key)
                if actual is None or pd.isna(actual):
                    continue
                pred = predict_xgboost(model, row)
                if pred is not None and np.isfinite(pred):
                    pair_store[key][cp].append((actual, float(pred)))

    results = []
    for key, cp_pairs in pair_store.items():
        out = dict(zip(group_cols, key))
        all_pairs = [pair for pairs in cp_pairs.values() for pair in pairs]
        for cp in checkpoints:
            pairs = cp_pairs[cp]
            if not pairs:
                out[f"xgboost_wd{cp}_wape"] = np.nan
                out[f"xgboost_wd{cp}_mae"] = np.nan
                out[f"xgboost_wd{cp}_rmse"] = np.nan
                out[f"xgboost_wd{cp}_bias"] = np.nan
                out[f"xgboost_wd{cp}_bias_pct"] = np.nan
                out[f"xgboost_wd{cp}_observations"] = 0
                continue
            y, p = zip(*pairs)
            out.update({
                f"xgboost_wd{cp}_wape": wape(y, p),
                f"xgboost_wd{cp}_mae": mae(y, p),
                f"xgboost_wd{cp}_rmse": rmse(y, p),
                f"xgboost_wd{cp}_bias": bias(y, p),
                f"xgboost_wd{cp}_bias_pct": bias_pct(y, p),
                f"xgboost_wd{cp}_observations": len(pairs),
            })
        if all_pairs:
            y, p = zip(*all_pairs)
            out["xgboost_wape"] = wape(y, p)
            out["xgboost_mae"] = mae(y, p)
            out["xgboost_rmse"] = rmse(y, p)
            out["xgboost_bias"] = bias(y, p)
            out["xgboost_bias_pct"] = bias_pct(y, p)
            out["xgboost_observations"] = len(all_pairs)
        else:
            for metric in ("wape", "mae", "rmse", "bias", "bias_pct"):
                out[f"xgboost_{metric}"] = np.nan
            out["xgboost_observations"] = 0
        results.append(out)

    return pd.DataFrame(results)
