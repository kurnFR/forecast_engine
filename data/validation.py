"""Sanity checks run before features/models see the data."""
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    pass


def validate_daily_sellin(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise DataValidationError(
            "No sell-in rows returned - check the join in data/extract.get_daily_sellin()."
        )

    required = {"regioncode", "branchcode", "invoice_date", "sellin_value"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing expected columns: {missing}")

    n_negative = int((df["sellin_value"] < 0).sum())
    if n_negative:
        logger.warning("%d rows with negative sellin_value (returns/credit notes?) kept as-is.", n_negative)

    n_null_region = int(df["regioncode"].isna().sum())
    if n_null_region:
        logger.warning("%d rows have no regioncode after the branch-dim join; dropping them.", n_null_region)
        df = df[df["regioncode"].notna()]

    dupes = int(df.duplicated(subset=["regioncode", "branchcode", "invoice_date"]).sum())
    if dupes:
        logger.warning("%d duplicate (region, branch, date) rows found - summed during aggregation.", dupes)

    return df


def validate_targets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise DataValidationError("No rows returned from region_target_view.")
    if df["target_sellin"].isna().all():
        raise DataValidationError("target_sellin is entirely NULL - cannot compute achievement %.")
    return df
