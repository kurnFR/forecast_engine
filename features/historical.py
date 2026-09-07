"""Trailing-history features for leakage-safe region-month forecasting."""
import numpy as np
import pandas as pd


def build_historical_features(monthly_df: pd.DataFrame, group_cols: list, n_lags: int = 12) -> pd.DataFrame:
    """Build features where every feature for month T uses data strictly before T."""
    required = set(group_cols + ["periode", "monthly_value"])
    missing = required - set(monthly_df.columns)
    if missing:
        raise ValueError(f"monthly_df missing columns: {missing}")

    df = monthly_df.sort_values(group_cols + ["periode"]).copy()
    grouped = df.groupby(group_cols)["monthly_value"]

    for lag in range(1, min(n_lags, 6) + 1):
        df[f"lag_{lag}"] = grouped.shift(lag)

    # IMPORTANT: growth for forecast month T must be based on T-1 versus T-2.
    # Using pct_change() directly would include T's actual value and leak the target.
    lag_1 = df.groupby(group_cols)["monthly_value"].shift(1)
    lag_2 = df.groupby(group_cols)["monthly_value"].shift(2)
    df["mom_growth"] = (lag_1 / lag_2 - 1.0).replace([np.inf, -np.inf], np.nan)

    df["rolling_mean_3"] = df.groupby(group_cols)["monthly_value"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["rolling_std_3"] = df.groupby(group_cols)["monthly_value"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).std()
    )

    df["calendar_month"] = df["periode"].dt.month
    prior = df.copy()
    prior["monthly_value"] = prior.groupby(group_cols)["monthly_value"].shift(1)
    seasonal_index = prior.groupby(group_cols + ["calendar_month"])["monthly_value"].transform("mean")
    overall_mean = prior.groupby(group_cols)["monthly_value"].transform("mean")
    df["seasonal_index"] = (seasonal_index / overall_mean).replace([np.inf, -np.inf], np.nan)

    return df
