"""Select the production candidate from leakage-safe backtest metrics."""
import numpy as np
import pandas as pd

CANDIDATE_MODELS = ["baseline", "ets", "sarima"]


def select_best_model(
    backtest_df: pd.DataFrame,
    group_cols: list,
    metric: str = "wape",
) -> pd.DataFrame:
    """Select the lowest-error available model per forecast grain.

    XGBoost is deliberately not selected here until its checkpoint backtest is
    implemented.  This prevents an unvalidated ML model from silently winning
    production model selection.
    """
    if backtest_df.empty:
        return pd.DataFrame(columns=group_cols + ["best_model", "best_model_score"])

    df = backtest_df.copy()
    metric_cols = [f"{m}_{metric}" for m in CANDIDATE_MODELS if f"{m}_{metric}" in df.columns]
    if not metric_cols:
        return pd.DataFrame(columns=group_cols + ["best_model", "best_model_score"])

    scores = df[metric_cols].replace([np.inf, -np.inf], np.nan)
    df["best_model"] = scores.idxmin(axis=1).str.replace(f"_{metric}", "", regex=False)
    df["best_model_score"] = scores.min(axis=1)
    return df[group_cols + ["best_model", "best_model_score"]]
