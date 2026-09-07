"""Orchestrates: pull data -> engineer features -> backtest -> fit final XGBoost."""
import logging
import pandas as pd

from data.extract import get_daily_sellin, get_region_monthly_targets, get_calendar
from data.validation import validate_daily_sellin, validate_targets
from data.aggregation import to_monthly, allocate_branch_targets
from features.historical import build_historical_features
from backtest.rolling import rolling_backtest
from backtest.model_selection import select_best_model
from models.xgboost_model import train_xgboost
from config import FORECAST_CONFIG

logger = logging.getLogger(__name__)

GROUP_COLS = FORECAST_CONFIG["grain"]


def run_training_pipeline() -> dict:
    logger.info("Extracting raw data...")
    daily = validate_daily_sellin(get_daily_sellin(FORECAST_CONFIG["current_month_lookback_days"]))
    targets = validate_targets(get_region_monthly_targets())
    calendar = get_calendar(FORECAST_CONFIG["current_month_lookback_days"])

    monthly = to_monthly(daily, GROUP_COLS)
    branch_targets = allocate_branch_targets(targets, monthly)

    logger.info("Building historical features...")
    hist_features = build_historical_features(monthly, GROUP_COLS)

    logger.info("Running rolling backtest (evaluates ETS / SARIMA / naive per group)...")
    backtest_results = rolling_backtest(monthly, GROUP_COLS, FORECAST_CONFIG["min_history_months"])
    best_models = select_best_model(backtest_results, GROUP_COLS)

    logger.info("Training XGBoost on pooled history across all region/branch series...")
    xgb_model = train_xgboost(hist_features)

    return {
        "daily": daily,
        "monthly": monthly,
        "hist_features": hist_features,
        "calendar": calendar,
        "branch_targets": branch_targets,
        "backtest_results": backtest_results,
        "best_models": best_models,
        "xgb_model": xgb_model,
    }
