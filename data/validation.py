from __future__ import annotations

import pandas as pd


class DataValidationError(ValueError):
    """Raised when forecast input/output violates the data contract."""


def validate_forecast_output(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the final region-month forecast contract before persistence."""
    if df.empty:
        raise DataValidationError("Forecast output is empty - nothing can be written.")

    required = {
        "regioncode", "periode", "forecast_p10", "forecast_p50", "forecast_p90",
        "target_sellin", "achievement_pct_forecast",
        "mtd_value", "elapsed_working_days", "remaining_working_days", "total_working_days",
    }
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Forecast output is missing expected columns: {sorted(missing)}")

    result = df.copy()
    result["periode"] = pd.to_datetime(result["periode"], errors="coerce")
    if result["periode"].isna().any():
        raise DataValidationError("Forecast output contains invalid periode values.")

    if result.duplicated(subset=["regioncode", "periode"]).any():
        raise DataValidationError("Forecast output contains duplicate (regioncode, periode) keys.")

    numeric_cols = [
        "forecast_p10", "forecast_p50", "forecast_p90", "target_sellin",
        "achievement_pct_forecast", "mtd_value",
        "elapsed_working_days", "remaining_working_days", "total_working_days",
    ]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")
        if result[col].isna().any():
            raise DataValidationError(f"Forecast output contains NULL/non-numeric values in {col}.")

    if not (result["forecast_p10"] <= result["forecast_p50"]).all():
        raise DataValidationError("Forecast interval violation: P10 must be <= P50 for every row.")
    if not (result["forecast_p50"] <= result["forecast_p90"]).all():
        raise DataValidationError("Forecast interval violation: P50 must be <= P90 for every row.")

    if (result[["forecast_p10", "forecast_p50", "forecast_p90"]] < 0).any().any():
        raise DataValidationError("Forecast interval contains negative values.")

    if (result["total_working_days"] <= 0).any():
        raise DataValidationError("Forecast output contains non-positive total_working_days.")
    if (result["elapsed_working_days"] < 0).any() or (result["remaining_working_days"] < 0).any():
        raise DataValidationError("Forecast output contains negative working-day counts.")
    if (result["elapsed_working_days"] + result["remaining_working_days"] != result["total_working_days"]).any():
        raise DataValidationError("Forecast working-day counts do not reconcile to total_working_days.")

    nonzero_target = result["target_sellin"] != 0
    expected_achievement = result["forecast_p50"] / result["target_sellin"].replace(0, pd.NA) * 100.0
    if not result.loc[nonzero_target, "achievement_pct_forecast"].sub(expected_achievement[nonzero_target]).abs().le(1e-6).all():
        raise DataValidationError(
            "achievement_pct_forecast must equal forecast_p50 / target_sellin * 100."
        )

    return result
