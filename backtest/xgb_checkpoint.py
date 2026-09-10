"""Leakage-safe XGBoost predictions at working-day checkpoints."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from backtest.metrics import bias, bias_pct, mae, rmse, wape
from config import FORECAST_CONFIG
from models.xgboost_model import train_xgboost_checkpoint, predict_xgboost

logger = logging.getLogger(__name__)


def _as_key(value):
    return value if isinstance(value, tuple) else (value,)


def _checkpoint_date(calendar, periode, checkpoint):
    month = calendar[
        (calendar["date"] >= periode)
        & (calendar["date"] < periode + pd.offsets.MonthBegin(1))
        & calendar["is_working_day"].astype(bool)
    ].sort_values("date")
    return month.iloc[checkpoint - 1]["date"] if len(month) >= checkpoint else None


def _residual_stats(pairs):
    if not pairs:
        return np.nan, np.nan, 0
    residuals = np.asarray([a - p for a, p in pairs], dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if not len(residuals):
        return np.nan, np.nan, 0
    return float(np.quantile(residuals, .10)), float(np.quantile(residuals, .90)), len(residuals)


def _history_features(monthly, group_cols, target_month, key):
    history = monthly[monthly["periode"] < target_month]
    g = history
    for col, value in zip(group_cols, key):
        g = g[g[col] == value]
    g = g.sort_values("periode")
    values = g["monthly_value"].astype(float).tolist()
    if not values:
        return None
    row = dict(zip(group_cols, key))
    row["periode"] = target_month
    for lag in range(1, 7):
        row[f"lag_{lag}"] = values[-lag] if len(values) >= lag else np.nan
    row["mom_growth"] = (
        values[-1] / values[-2] - 1.0
        if len(values) >= 2 and values[-2] != 0
        else np.nan
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
    return row


def _checkpoint_row(monthly, daily, calendar, targets, group_cols, target_month, checkpoint, key, min_train_months):
    history = monthly[monthly["periode"] < target_month]
    g = history
    for col, value in zip(group_cols, key):
        g = g[g[col] == value]
    if len(g) < min_train_months:
        return None

    checkpoint_date = _checkpoint_date(calendar, target_month, checkpoint)
    if checkpoint_date is None:
        return None

    row = _history_features(monthly, group_cols, target_month, key)
    if row is None:
        return None

    cur = daily[
        (daily["invoice_date"] >= target_month)
        & (daily["invoice_date"] <= checkpoint_date)
    ]
    for col, value in zip(group_cols, key):
        cur = cur[cur[col] == value]
    mtd_value = float(cur["sellin_value"].sum()) if not cur.empty else 0.0

    month_calendar = calendar[
        (calendar["date"] >= target_month)
        & (calendar["date"] < target_month + pd.offsets.MonthBegin(1))
    ]
    total_wd = int(month_calendar["is_working_day"].astype(bool).sum())
    elapsed_wd = int(
        month_calendar.loc[month_calendar["date"] <= checkpoint_date, "is_working_day"]
        .astype(bool)
        .sum()
    )

    target = targets
    for col, value in zip(group_cols, key):
        target = target[target[col] == value]
    target = target[target["periode"] == target_month]
    target_value = float(target["target_sellin"].iloc[0]) if not target.empty and pd.notna(target["target_sellin"].iloc[0]) else np.nan

    row.update(
        {
            "mtd_value": mtd_value,
            "elapsed_working_days": elapsed_wd,
            "remaining_working_days": max(total_wd - elapsed_wd, 0),
            "total_working_days": total_wd,
            "avg_daily_rate": mtd_value / max(elapsed_wd, 1),
            "run_rate_forecast": mtd_value / max(elapsed_wd, 1) * total_wd,
            "target_sellin": target_value,
            "mtd_target_pct": mtd_value / target_value * 100.0 if pd.notna(target_value) and np.isfinite(target_value) and target_value != 0 else np.nan,
            "checkpoint": checkpoint,
        }
    )
    actual = monthly[monthly["periode"] == target_month]
    for col, value in zip(group_cols, key):
        actual = actual[actual[col] == value]
    row["actual"] = float(actual["monthly_value"].iloc[0]) if not actual.empty else np.nan
    return row


def _build_training_frame(monthly, daily, calendar, targets, group_cols, target_month, checkpoint, min_train_months):
    """Build training examples from target months strictly before target_month."""
    prior_months = sorted(m for m in monthly["periode"].drop_duplicates() if m < target_month)
    rows = []
    for historical_target in prior_months:
        keys = monthly.loc[monthly["periode"] == historical_target, group_cols].drop_duplicates().itertuples(index=False, name=None)
        for key in keys:
            row = _checkpoint_row(
                monthly, daily, calendar, targets, group_cols,
                historical_target, checkpoint, _as_key(key), min_train_months
            )
            if row is not None and pd.notna(row.get("actual")):
                row["monthly_value"] = row["actual"]
                rows.append(row)
    return pd.DataFrame(rows)


def _build_all_checkpoint_training_frames(monthly, daily, calendar, targets, group_cols, checkpoints, min_train_months):
    """Build each checkpoint snapshot once, avoiding repeated historical scans."""
    if monthly.empty:
        return {int(cp): pd.DataFrame() for cp in checkpoints}

    last_target = monthly["periode"].max()
    snapshot_end = last_target + pd.offsets.MonthBegin(1)
    return {
        int(cp): _build_training_frame(
            monthly, daily, calendar, targets, group_cols,
            snapshot_end, int(cp), min_train_months
        )
        for cp in checkpoints
    }


def rolling_xgb_checkpoint_backtest(
    monthly_history,
    daily_history,
    calendar,
    group_cols,
    checkpoints,
    min_train_months=24,
    targets=None,
    return_predictions=False,
):
    """Evaluate checkpoint-aware XGBoost using only prior target-month information.

    When return_predictions=True, also return the leakage-safe OOS prediction
    pairs keyed by region and checkpoint for downstream ensemble auditing.
    """
    monthly = monthly_history.copy()
    monthly["periode"] = pd.to_datetime(monthly["periode"])
    daily = daily_history.copy()
    daily["invoice_date"] = pd.to_datetime(daily["invoice_date"])
    calendar = calendar.copy()
    calendar["date"] = pd.to_datetime(calendar["date"])
    if targets is None:
        targets = pd.DataFrame(columns=["periode", *group_cols, "target_sellin"])
    else:
        targets = targets.copy()
        targets["periode"] = pd.to_datetime(targets["periode"])

    pair_store = {
        _as_key(key): {cp: [] for cp in checkpoints}
        for key in monthly[group_cols].drop_duplicates().itertuples(index=False, name=None)
    }
    checkpoint_frames = _build_all_checkpoint_training_frames(
        monthly, daily, calendar, targets, group_cols, checkpoints, min_train_months
    )

    for target_month in sorted(monthly["periode"].drop_duplicates()):
        for cp in checkpoints:
            checkpoint_date = _checkpoint_date(calendar, target_month, cp)
            if checkpoint_date is None:
                continue
            train_frame = checkpoint_frames[int(cp)]
            if not train_frame.empty:
                train_frame = train_frame[train_frame["periode"] < target_month]
            if train_frame.empty or len(train_frame) < min_train_months:
                continue
            try:
                model = train_xgboost_checkpoint(train_frame, target_col="monthly_value")
            except (ValueError, TypeError) as exc:
                logger.debug("Checkpoint XGBoost unavailable for %s WD%s: %s", target_month, cp, exc)
                continue

            keys = monthly.loc[monthly["periode"] == target_month, group_cols].drop_duplicates().itertuples(index=False, name=None)
            for key in keys:
                key = _as_key(key)
                row = _checkpoint_row(
                    monthly, daily, calendar, targets, group_cols,
                    target_month, cp, key, min_train_months
                )
                if row is None or pd.isna(row.get("actual")):
                    continue
                pred = predict_xgboost(model, pd.DataFrame([row]))
                if pred is not None and np.isfinite(pred):
                    pair_store[key][cp].append((float(row["actual"]), float(pred)))

    results = []
    for key, cp_pairs in pair_store.items():
        out = dict(zip(group_cols, key))
        all_pairs = [p for pairs in cp_pairs.values() for p in pairs]
        for cp in checkpoints:
            pairs = cp_pairs[cp]
            if pairs:
                y, p = zip(*pairs)
                out[f"xgboost_wd{cp}_wape"] = wape(y, p)
                out[f"xgboost_wd{cp}_mae"] = mae(y, p)
                out[f"xgboost_wd{cp}_rmse"] = rmse(y, p)
                out[f"xgboost_wd{cp}_bias"] = bias(y, p)
                out[f"xgboost_wd{cp}_bias_pct"] = bias_pct(y, p)
            else:
                for metric in ("wape", "mae", "rmse", "bias", "bias_pct"):
                    out[f"xgboost_wd{cp}_{metric}"] = np.nan
            out[f"xgboost_wd{cp}_observations"] = len(pairs)
            q10, q90, count = _residual_stats(pairs)
            out[f"xgboost_wd{cp}_residual_q10"] = q10
            out[f"xgboost_wd{cp}_residual_q90"] = q90
            out[f"xgboost_wd{cp}_residual_observations"] = count
        if all_pairs:
            y, p = zip(*all_pairs)
            out.update({
                "xgboost_wape": wape(y, p),
                "xgboost_mae": mae(y, p),
                "xgboost_rmse": rmse(y, p),
                "xgboost_bias": bias(y, p),
                "xgboost_bias_pct": bias_pct(y, p),
                "xgboost_observations": len(all_pairs),
            })
            q10, q90, _ = _residual_stats(all_pairs)
            out["xgboost_residual_q10"] = q10
            out["xgboost_residual_q90"] = q90
        else:
            for metric in ("wape", "mae", "rmse", "bias", "bias_pct"):
                out[f"xgboost_{metric}"] = np.nan
            out["xgboost_observations"] = 0
            out["xgboost_residual_q10"] = np.nan
            out["xgboost_residual_q90"] = np.nan
        results.append(out)
    result_df = pd.DataFrame(results)
    if return_predictions:
        return result_df, pair_store
    return result_df
