"""Trailing-history features: lags, growth, seasonality index, volatility."""
import numpy as np
import pandas as pd


def build_historical_features(monthly_df: pd.DataFrame, group_cols: list, n_lags: int = 12) -> pd.DataFrame:
    df = monthly_df.sort_values(group_cols + ["periode"]).copy()

    for lag in range(1, min(n_lags, 6) + 1):
        df[f"lag_{lag}"] = df.groupby(group_cols)["monthly_value"].shift(lag)

    df["mom_growth"] = df.groupby(group_cols)["monthly_value"].pct_change()
    df["rolling_mean_3"] = df.groupby(group_cols)["monthly_value"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["rolling_std_3"] = df.groupby(group_cols)["monthly_value"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).std()
    )

    df["calendar_month"] = df["periode"].dt.month
    seasonal_index = df.groupby(group_cols + ["calendar_month"])["monthly_value"].transform(
        lambda s: s.shift(1).mean()
    )
    overall_mean = df.groupby(group_cols)["monthly_value"].transform(lambda s: s.shift(1).mean())
    df["seasonal_index"] = (seasonal_index / overall_mean).replace([np.inf, -np.inf], np.nan)

    return df
