"""Transparent run-rate baseline using working-day progress."""
import pandas as pd


def forecast_from_mtd(mtd_value: float, elapsed_working_days: int, total_working_days: int) -> float:
    """Project month-end value from MTD value and working-day progress."""
    if total_working_days <= 0:
        return 0.0
    if elapsed_working_days <= 0:
        return 0.0
    return max(float(mtd_value), 0.0) / elapsed_working_days * total_working_days


def forecast_baseline(current_month_features: pd.DataFrame) -> pd.DataFrame:
    """Add the run-rate baseline to a current-month feature frame."""
    out = current_month_features.copy()
    out["forecast_baseline"] = out.apply(
        lambda r: forecast_from_mtd(
            r["mtd_value"],
            int(r["elapsed_working_days"]),
            int(r["total_working_days"]),
        ),
        axis=1,
    )
    return out
