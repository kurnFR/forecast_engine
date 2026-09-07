"""Training orchestration for the V2 region-month forecast engine."""
import logging

import pandas as pd

from data.extract import get_daily_sellin, get_region_monthly_targets, get_calendar
from data.validation import (
    validate_daily_sellin,
    validate_targets,
    validate_region_alignment,
    validate_calendar,
)
from data.aggregation import to_monthly, complete_month_panel
from features.historical import build_historical_features
from features.working_day import historical_working_days_per_month
from backtest.rolling import rolling_backtest
from backtest.rolling_intervals import rolling_interval_backtest
from backtest.xgb_checkpoint import _build_training_frame
from backtest.model_selection import select_best_model
from models.xgboost_model import train_xgboost, train_xgboost_checkpoint_models
from config import FORECAST_CONFIG

logger = logging.getLogger(__name__)
GROUP_COLS = FORECAST_CONFIG["grain"]


def run_training_pipeline() -> dict:
    """Extract, validate, panelize, backtest and train the V2 model set."""
    history_months = FORECAST_CONFIG["history_months"]
    logger.info("Extracting %d months of region-level Sell-In history...", history_months)
    daily = validate_daily_sellin(get_daily_sellin(history_months))
    targets = validate_targets(get_region_monthly_targets(history_months))
    calendar = validate_calendar(
        get_calendar(history_months),
        required_checkpoints=FORECAST_CONFIG["backtest_checkpoints"],
    )
    validate_region_alignment(daily, targets)

    if daily.empty:
        raise ValueError("No Sell-In history available for the configured period.")
    if calendar.empty:
        raise ValueError("No calendar rows available for the configured period.")

    current_month = pd.Timestamp.today().normalize().replace(day=1)
    start = current_month - pd.DateOffset(months=history_months - 1)
    monthly_raw = to_monthly(daily, GROUP_COLS)
    monthly = complete_month_panel(monthly_raw, GROUP_COLS, start, current_month)

    closed = monthly[monthly["periode"] < current_month].copy()
    history_counts = closed.groupby(GROUP_COLS)["periode"].nunique()
    eligible = history_counts[history_counts >= FORECAST_CONFIG["min_history_months"]].index
    if GROUP_COLS == ["regioncode"]:
        monthly = monthly[monthly["regioncode"].isin(eligible)].copy()

    logger.info("Building leakage-safe historical features...")
    hist_features = build_historical_features(monthly, GROUP_COLS)
    working_days = historical_working_days_per_month(calendar)
    hist_features = hist_features.merge(working_days, on="periode", how="left", validate="many_to_one")
    if hist_features["total_working_days"].isna().any():
        raise ValueError("Calendar is missing total_working_days for one or more forecast months.")

    logger.info("Running WD4/7/10/15/20 rolling backtest for baseline/ETS/SARIMA/XGBoost...")
    backtest_results = rolling_backtest(
        monthly,
        GROUP_COLS,
        min_train_months=FORECAST_CONFIG["backtest_min_train_months"],
        daily_history=daily,
        calendar=calendar,
        checkpoints=FORECAST_CONFIG["backtest_checkpoints"],
        targets=targets,
    )
    best_models = select_best_model(backtest_results, GROUP_COLS)

    logger.info("Running strict leakage-safe P10/P50/P90 interval backtest...")
    interval_backtest_results = rolling_interval_backtest(
        monthly_history=monthly,
        daily_history=daily,
        calendar=calendar,
        group_cols=GROUP_COLS,
        checkpoints=FORECAST_CONFIG["backtest_checkpoints"],
        min_train_months=FORECAST_CONFIG["backtest_min_train_months"],
        targets=targets,
    )

    logger.info("Training pooled history-only XGBoost...")
    train_features = hist_features[hist_features["periode"] < current_month].copy()
    xgb_model = train_xgboost(train_features)

    logger.info("Training checkpoint-aware XGBoost models...")
    checkpoint_training = []
    for checkpoint in FORECAST_CONFIG["backtest_checkpoints"]:
        frame = _build_training_frame(
            monthly, daily, calendar, targets, GROUP_COLS,
            current_month, checkpoint, FORECAST_CONFIG["backtest_min_train_months"]
        )
        if not frame.empty:
            checkpoint_training.append(frame)
    checkpoint_training = pd.concat(checkpoint_training, ignore_index=True) if checkpoint_training else pd.DataFrame()
    xgb_checkpoint_models = train_xgboost_checkpoint_models(
        checkpoint_training,
        FORECAST_CONFIG["backtest_checkpoints"],
    )

    return {
        "daily": daily,
        "monthly": monthly,
        "hist_features": hist_features,
        "calendar": calendar,
        "targets": targets,
        "backtest_results": backtest_results,
        "interval_backtest_results": interval_backtest_results,
        "best_models": best_models,
        "xgb_model": xgb_model,
        "xgb_checkpoint_models": xgb_checkpoint_models,
    }
