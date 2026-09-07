"""Pick the best-performing time-series model per (region, branch) from backtest results."""
import pandas as pd

CANDIDATE_MODELS = ["ets", "sarima", "naive"]


def select_best_model(backtest_df: pd.DataFrame, group_cols: list, metric: str = "mape") -> pd.DataFrame:
    if backtest_df.empty:
        return pd.DataFrame(columns=group_cols + ["best_model", "best_model_score"])

    df = backtest_df.copy()
    metric_cols = [f"{m}_{metric}" for m in CANDIDATE_MODELS]
    df["best_model"] = df[metric_cols].idxmin(axis=1).str.replace(f"_{metric}", "", regex=False)
    df["best_model_score"] = df[metric_cols].min(axis=1)
    return df[group_cols + ["best_model", "best_model_score"]]
