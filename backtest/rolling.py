"""Rolling-origin backtest: at each past closed month, pretend we're standing
right after that month closed and see how well each model would have
predicted it, using only data available before that point."""
import logging
import pandas as pd

from models.ets import forecast_ets
from models.sarima import forecast_sarima
from backtest.metrics import mape, mae, rmse, bias

logger = logging.getLogger(__name__)


def rolling_backtest(monthly_history: pd.DataFrame, group_cols: list, min_train_months: int = 6) -> pd.DataFrame:
    """
    monthly_history: one row per (group..., periode, monthly_value), any order.
    Returns per-group accuracy metrics for ets / sarima / naive (last-value) models,
    computed over every month that could be backtested for that group.
    """
    results = []
    for key, g in monthly_history.groupby(group_cols):
        g = g.sort_values("periode").reset_index(drop=True)
        if len(g) <= min_train_months:
            continue

        y_true, ets_preds, sarima_preds, naive_preds = [], [], [], []
        for cutoff in range(min_train_months, len(g)):
            train_series = g.loc[:cutoff - 1, "monthly_value"]
            actual = g.loc[cutoff, "monthly_value"]

            y_true.append(actual)
            ets_preds.append(forecast_ets(train_series) or train_series.iloc[-1])
            sarima_preds.append(forecast_sarima(train_series) or train_series.iloc[-1])
            naive_preds.append(train_series.iloc[-1])  # last-value naive baseline

        key_vals = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_vals))
        for model_name, preds in [("ets", ets_preds), ("sarima", sarima_preds), ("naive", naive_preds)]:
            row.update({
                f"{model_name}_mape": mape(y_true, preds),
                f"{model_name}_mae": mae(y_true, preds),
                f"{model_name}_rmse": rmse(y_true, preds),
                f"{model_name}_bias": bias(y_true, preds),
            })
        results.append(row)

    return pd.DataFrame(results)
