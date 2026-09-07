"""Leakage-safe XGBoost predictions at working-day checkpoints."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from backtest.metrics import bias, bias_pct, mae, rmse, wape
from features.historical import build_historical_features
from models.xgboost_model import train_xgboost, predict_xgboost

logger = logging.getLogger(__name__)


def _checkpoint_date(calendar: pd.DataFrame, periode: pd.Timestamp, checkpoint: int):
    m = calendar[
        (calendar["date"] >= periode)
        & (calendar["date"] < periode + pd.offsets.MonthBegin(1))
        & calendar["is_working_day"].astype(bool)
    ].sort_values("date")
    if len(m) < checkpoint:
        return None
    return m.iloc[checkpoint - 1]["date"]


def _xgb_checkpoint_features(
    monthly_history: pd.DataFrame,
    daily_history: pd.DataFrame,
    calendar: pd.DataFrame,
    group_cols: list[str],
    target_month: pd.Timestamp,
    checkpoint: int,
    train_min_months: int,
) -> tuple[pd.DataFrame | None, float | None]:
    """Build a target-month feature row without using target-month future sales."""
    key = monthly_history[group_cols].drop_duplicates()
    target_daily = daily_history[
        (daily_history["invoice_date"] >= target_month)
        & (daily_history["invoice_date"] < target_month + pd.offsets.MonthBegin(1))
    ]
    cp_date = _checkpoint_date(calendar, target_month, checkpoint)
    if cp_date is None:
        return None, None

    target_daily = target_daily[target_daily["invoice_date"] <= cp_date]
    if target_daily.empty:
        return None, None

    # Historical feature values are generated from closed months only.
    history = monthly_history[monthly_history["periode"] < target_month].copy()
    if history.groupby(group_cols)["periode"].nunique().min() < train_min_months:
        return None, None

    # One synthetic row represents the target month. Its lag/seasonality
    # features are derived from history, while working-day count comes from
    # the target month's calendar. No target-month Sell-In is used as a lag.
    rows = []
    for _, group in key.iterrows():
        mask = np.ones(len(history), dtype=bool)
        for c in group_cols:
            mask &= history[c].eq(group[c]).to_numpy()
        g = history.loc[mask].sort_values("periode").copy()
        if len(g) < train_min_months:
            continue

        values = g["monthly_value"].astype(float).tolist()
        row = {c: group[c] for c in group_cols}
        row["periode"] = target_month
        for lag in range(1, 7):
            row[f"lag_{lag}"] = values[-lag] if len(values) >= lag else np.nan
        row["mom_growth"] = (
            values[-1] / values[-2] - 1 if len(values) >= 2 and values[-2] != 0 else np.nan
        )
        row["rolling_mean_3"] = np.mean(values[-3:]) if len(values) >= 1 else np.nan
        row["rolling_std_3"] = np.std(values[-3:], ddof=1) if len(values[-3:]) >= 2 else np.nan
        same_month = g[g["periode"].dt.month == target_month.month]["monthly_value"].astype(float)
        row["seasonal_index"] = (
            same_month.mean() / np.mean(values) if len(same_month) and np.mean(values) != 0 else np.nan
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

    if not rows:
        return None, None
    return pd.DataFrame(rows), cp_date


def rolling_xgb_checkpoint_backtest(
    monthly_history: pd.DataFrame,
    daily_history: pd.DataFrame,
    calendar: pd.DataFrame,
    group_cols: list[str],
    checkpoints: list[int],
    min_train_months: int = 24,
) -> pd.DataFrame:
    """Evaluate XGBoost at WD checkpoints using a fresh model per target month.

    A model is fitted only on months before the target month.  The checkpoint
    row contains calendar information for the target month but never target
    month Sell-In, preventing future leakage.
    """
    monthly = monthly_history.copy()
    monthly["periode"] = pd.to_datetime(monthly["periode"])
    daily = daily_history.copy()
    daily["invoice_date"] = pd.to_datetime(daily["invoice_date"])
    calendar = calendar.copy()
    calendar["date"] = pd.to_datetime(calendar["date"])

    results = []
    for key, g in monthly.groupby(group_cols):
        key_vals = key if isinstance(key, tuple) else (key,)
        key_mask = monthly[group_cols].eq(pd.Series(key_vals, index=group_cols)).all(axis=1)
        group_months = monthly.loc[key_mask, "periode"].drop_duplicates().sort_values()
        pairs = {cp: [] for cp in checkpoints}

        for target_month in group_months:
            train = monthly[(monthly["periode"] < target_month)]
            train_group = train[train[group_cols].eq(pd.Series(key_vals, index=group_cols)).all(axis=1)]
            if len(train_group) < min_train_months:
                continue

            model = train_xgboost(build_historical_features(train, group_cols).query("periode < @target_month"))
            for cp in checkpoints:
                feature_row, cp_date = _xgb_checkpoint_features(
                    monthly, daily, calendar, group_cols, target_month, cp, min_train_months
                )
                if feature_row is None or cp_date is None:
                    continue
                row = feature_row[feature_row[group_cols].eq(pd.Series(key_vals, index=group_cols)).all(axis=1)]
                if row.empty:
                    continue
                pred = predict_xgboost(model, row)
                actual = g.loc[g["periode"] == target_month, "monthly_value"].sum()
                if pred is not None and np.isfinite(pred):
                    pairs[cp].append((float(actual), float(pred)))

        out = dict(zip(group_cols, key_vals))
        all_pairs = [p for cp in checkpoints for p in pairs[cp]]
        for cp in checkpoints:
            cp_pairs = pairs[cp]
            if not cp_pairs:
                out[f"xgboost_wd{cp}_wape"] = np.nan
                out[f"xgboost_wd{cp}_observations"] = 0
                continue
            y, p = zip(*cp_pairs)
            out.update({
                f"xgboost_wd{cp}_wape": wape(y, p),
                f"xgboost_wd{cp}_mae": mae(y, p),
                f"xgboost_wd{cp}_rmse": rmse(y, p),
                f"xgboost_wd{cp}_bias": bias(y, p),
                f"xgboost_wd{cp}_bias_pct": bias_pct(y, p),
                f"xgboost_wd{cp}_observations": len(cp_pairs),
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
