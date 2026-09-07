"""Sanity checks run before features/models see the data."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    pass


def validate_daily_sellin(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise DataValidationError("No sell-in rows returned - check the source query.")

    required = {"regioncode", "invoice_date", "sellin_value"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing expected columns: {missing}")

    df = df.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["sellin_value"] = pd.to_numeric(df["sellin_value"], errors="coerce")

    bad_dates = int(df["invoice_date"].isna().sum())
    if bad_dates:
        raise DataValidationError(f"{bad_dates} rows have invalid invoice_date values.")

    bad_values = int(df["sellin_value"].isna().sum())
    if bad_values:
        logger.warning("%d rows have NULL/non-numeric sellin_value; treating them as zero.", bad_values)
        df["sellin_value"] = df["sellin_value"].fillna(0.0)

    null_region = int(df["regioncode"].isna().sum())
    if null_region:
        logger.warning("%d rows have no regioncode; dropping them.", null_region)
        df = df[df["regioncode"].notna()].copy()

    dupes = int(df.duplicated(subset=["regioncode", "invoice_date"]).sum())
    if dupes:
        logger.warning("%d duplicate region/date rows found - aggregation will sum them.", dupes)

    n_negative = int((df["sellin_value"] < 0).sum())
    if n_negative:
        logger.warning("%d rows with negative sellin_value (returns/credit notes?) kept as-is.", n_negative)

    return df


def validate_targets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise DataValidationError("No rows returned from region_target_view.")

    required = {"regioncode", "periode", "target_sellin"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing expected columns: {missing}")

    df = df.copy()
    df["periode"] = pd.to_datetime(df["periode"], errors="coerce")
    df["target_sellin"] = pd.to_numeric(df["target_sellin"], errors="coerce")
    if df["periode"].isna().any():
        raise DataValidationError("Target data contains invalid periode values.")
    if df["target_sellin"].isna().all():
        raise DataValidationError("target_sellin is entirely NULL - cannot compute achievement %.")

    duplicate_keys = int(df.duplicated(subset=["regioncode", "periode"]).sum())
    if duplicate_keys:
        raise DataValidationError(
            f"Target source contains {duplicate_keys} duplicate (regioncode, periode) rows; "
            "the V2 target contract requires one row per region-month."
        )
    return df


def validate_region_alignment(daily: pd.DataFrame, targets: pd.DataFrame) -> None:
    """Reject region-code mapping drift between actuals and authoritative targets."""
    daily_regions = set(daily["regioncode"].dropna().astype(str).str.strip())
    target_regions = set(targets["regioncode"].dropna().astype(str).str.strip())
    missing_targets = sorted(daily_regions - target_regions)
    orphan_targets = sorted(target_regions - daily_regions)
    if missing_targets or orphan_targets:
        raise DataValidationError(
            "Region mapping mismatch between Sell-In and target source: "
            f"missing target regions={missing_targets[:10]}, "
            f"target-only regions={orphan_targets[:10]}"
        )


def validate_calendar(calendar: pd.DataFrame, required_checkpoints: list[int] | None = None) -> pd.DataFrame:
    """Validate the daily calendar and normalized networkeddays indicator."""
    if calendar.empty:
        raise DataValidationError("Calendar is empty - check the date dimension query.")
    required = {"date", "is_working_day"}
    missing = required - set(calendar.columns)
    if missing:
        raise DataValidationError(f"Calendar is missing expected columns: {missing}")

    c = calendar.copy()
    c["date"] = pd.to_datetime(c["date"], errors="coerce")
    if c["date"].isna().any():
        raise DataValidationError("Calendar contains invalid date values.")
    if c["date"].duplicated().any():
        raise DataValidationError("Calendar contains duplicate dates.")

    c["is_working_day"] = pd.to_numeric(c["is_working_day"], errors="coerce")
    if c["is_working_day"].isna().any() or not c["is_working_day"].isin([0, 1]).all():
        raise DataValidationError("Calendar is_working_day must contain only 0/1 values.")
    c = c.sort_values("date").reset_index(drop=True)
    if len(c) > 1:
        gaps = c["date"].diff().dropna().dt.days
        if not gaps.eq(1).all():
            raise DataValidationError("Calendar contains missing/non-consecutive dates.")

    checkpoints = required_checkpoints or []
    if checkpoints:
        month_wd = c.groupby(c["date"].dt.to_period("M"))["is_working_day"].sum()
        insufficient = month_wd[month_wd < max(checkpoints)]
        if not insufficient.empty:
            raise DataValidationError(
                "Calendar does not contain enough working days for required checkpoints: "
                f"{insufficient.to_dict()}"
            )
    return c
