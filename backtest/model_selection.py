"""Select production candidates from leakage-safe checkpoint backtests."""
import numpy as np
import pandas as pd

from config import CANDIDATE_MODELS

CHECKPOINTS = [4, 7, 10, 15, 20]


def select_best_model(
    backtest_df: pd.DataFrame,
    group_cols: list,
    metric: str = "wape",
) -> pd.DataFrame:
    """Select the lowest WAPE model per grain after checkpoint validation.

    The primary score is the mean of available WD4/7/10/15/20 WAPE values.
    A model must have at least three valid checkpoints to be eligible. This
    avoids selecting a model because it happened to have one valid checkpoint.
    """
    if backtest_df.empty:
        return pd.DataFrame(columns=group_cols + ["best_model", "best_model_score"])

    df = backtest_df.copy()
    score_cols = {}
    valid_cols = {}
    for model in CANDIDATE_MODELS:
        cols = [f"{model}_wd{cp}_{metric}" for cp in CHECKPOINTS if f"{model}_wd{cp}_{metric}" in df.columns]
        if not cols:
            continue
        values = df[cols].replace([np.inf, -np.inf], np.nan)
        score_cols[model] = values.mean(axis=1, skipna=True)
        valid_cols[model] = values.notna().sum(axis=1)

    if not score_cols:
        return pd.DataFrame(columns=group_cols + ["best_model", "best_model_score"])

    score_frame = pd.DataFrame(score_cols, index=df.index)
    count_frame = pd.DataFrame(valid_cols, index=df.index)
    eligible_scores = score_frame.where(count_frame >= 3)

    df["best_model"] = eligible_scores.idxmin(axis=1)
    df["best_model_score"] = eligible_scores.min(axis=1)

    # Keep diagnostic checkpoint scores so monitoring can explain why a model
    # won, while retaining one production recommendation per grain.
    out_cols = group_cols + ["best_model", "best_model_score"]
    for model in CANDIDATE_MODELS:
        if model in score_cols:
            df[f"{model}_checkpoint_score"] = score_cols[model]
            out_cols.append(f"{model}_checkpoint_score")
    return df[out_cols]
