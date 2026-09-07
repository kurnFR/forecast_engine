"""Training orchestration for the V2 region-month forecast engine."""
import logging

import pandas as pd

from data.extract import get_daily_sellin, get_region_monthly_targets, get_calendar
from data.validation import validate_daily_sellin, validate_targets
from data.aggregation import to_monthly, complete_month_panel
from features.historical import build_historical_features
from backtest.rolling import rolling_backtest
from backtest.model_selection import select_best_model
from models.xgboost_model import train_xgboost
from config import FORECAST_CONFIG

logger = logging.getLogger(__name__)
GROUP_COLS = FORECAST_CONFIG["grain"]


def run_training_pipeline() -> dict:
    """Extract, validate, panelize, backtest and train the V2 model set."""
    history_months = FORECAST_CONFIG["history_months"]

    logger.info("Extracting %d months of region-level Sell-In history...", history_months)
    daily = validate_daily_sellin(get_daily_sellin(history_months))
    targets = validate_targets(get_region_monthly_targets(history_months))
    calendar = get_calendar(history_months)

    if daily.empty:
        raise ValueError("No Sell-In history available for the configured period.")

    current_month = pd.Timestamp.today().normalize().replace(day=1)
    start = current_month - pd.DateOffset(months=history_months - 1)
    monthly_raw = to_monthly(daily, GROUP_COLS)
    monthly = complete_month_panel(monthly_raw, GROUP_COLS, start, current_month)

    # Model eligibility is based on closed months only.  The in-progress month
    # is never used as training history for the time-series models.
    closed = monthly[monthly["periode"] < current_month].copy()
    history_counts = closed.groupby(GROUP_COLS)["periode"].nunique()
    eligible = history_counts[history_counts >= FORECAST_CONFIG["min_history_months"]].index
    if GROUP_COLS == ["regioncode"]:
        monthly = monthly[monthly["regioncode"].isin(eligible)].copy()

    logger.info("Building leakage-safe historical features...")
    hist_features = build_historical_features(monthly, GROUP_COLS)

    logger.info("Running WD4/7/10/15/20 rolling backtest...")
    backtest_results = rolling_backtest(
        monthly,
        GROUP_COLS,
        min_train_months=FORECAST_CONFIG["backtest_min_train_months"],
        daily_history=daily,
        calendar=calendar,
        checkpoints=FORECAST_CONFIG["backtest_checkpoints"],
    )
    best_models = select_best_model(backtest_results, GROUP_COLS)

    logger.info("Training pooled XGBoost on historical region-month features...")
    train_features = hist_features[hist_features["periode"] < current_month].copy()
    xgb_model = train_xgboost(train_features)

    return {
        "daily": daily,
        "monthly": monthly,
        "hist_features": hist_features,
        "calendar": calendar,
        "targets": targets,
        "backtest_results": backtest_results,
        "best_models": best_models,
        "xgb_model": xgb_model,
    }
