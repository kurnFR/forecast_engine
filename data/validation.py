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
        raise DataValidationError(f"Missing target columns: {missing}")

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
