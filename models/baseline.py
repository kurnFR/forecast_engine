"""Simple, hard-to-beat run-rate baseline: MTD / elapsed_working_days * total_working_days."""
import pandas as pd


def forecast_baseline(current_month_features: pd.DataFrame) -> pd.DataFrame:
    out = current_month_features.copy()
    out["forecast_baseline"] = out["run_rate_forecast"]
    return out
