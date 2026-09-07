"""Combine per-model forecasts into a single end-of-month number."""
from typing import Optional
import numpy as np
import pandas as pd

from config import MODEL_CONFIG


def combine_forecasts(row: pd.Series, weights: Optional[dict] = None) -> float:
    weights = weights or MODEL_CONFIG["ensemble_weights_default"]
    available = {k: row.get(f"forecast_{k}") for k in weights if pd.notna(row.get(f"forecast_{k}"))}
    if not available:
        return np.nan
    w_sum = sum(weights[k] for k in available)
    return sum(weights[k] * v for k, v in available.items()) / w_sum


def build_ensemble(df: pd.DataFrame, weights: Optional[dict] = None) -> pd.DataFrame:
    out = df.copy()
    out["forecast_ensemble"] = out.apply(lambda r: combine_forecasts(r, weights), axis=1)
    return out
