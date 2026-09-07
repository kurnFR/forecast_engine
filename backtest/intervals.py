"""Out-of-sample interval diagnostics for forecast P10/P50/P90."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.ensemble import interval_coverage_metrics


def evaluate_intervals(df: pd.DataFrame, group_cols: list[str],
                       actual_col: str = "actual",
                       p10_col: str = "forecast_p10",
                       p50_col: str = "forecast_p50",
                       p90_col: str = "forecast_p90",
                       checkpoint_col: str = "checkpoint") -> pd.DataFrame:
    """Return overall and checkpoint/region interval diagnostics."""
    required = set(group_cols + [actual_col, p10_col, p50_col, p90_col])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing interval columns: {missing}")

    rows = []
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["checkpoint", "observations",
            "p10_below_rate", "p90_above_rate", "coverage",
            "mean_interval_width", "mae_p50"])

    grouping = group_cols + ([checkpoint_col] if checkpoint_col in df.columns else [])
    for keys, group in df.groupby(grouping, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        metric = interval_coverage_metrics(
            group[actual_col], group[p10_col], group[p50_col], group[p90_col]
        )
        row = dict(zip(grouping, keys))
        row.update(metric)
        rows.append(row)

    return pd.DataFrame(rows)


def interval_quality_flags(metrics: pd.DataFrame,
                           target_coverage: float = 0.80,
                           tolerance: float = 0.10) -> pd.DataFrame:
    """Flag interval groups whose empirical coverage is materially off target."""
    out = metrics.copy()
    if out.empty:
        out["coverage_status"] = pd.Series(dtype=str)
        return out

    lower = target_coverage - tolerance
    upper = target_coverage + tolerance
    out["coverage_status"] = np.where(
        out["coverage"].between(lower, upper, inclusive="both"), "OK", "REVIEW"
    )
    return out
