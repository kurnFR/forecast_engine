"""V2 data extraction: region-level Sell-In history, targets and calendar."""
import pandas as pd

from config import SOURCE
from db import read_sql


def get_daily_sellin(history_months: int = 36) -> pd.DataFrame:
    """Return daily Sell-In aggregated at region grain.

    The extraction deliberately contains no branch dimension join.  Region is
    the locked V2 forecast grain, and avoiding an intermediate branch mapping
    removes a major source of double counting and mapping leakage.
    """
    months = max(int(history_months), 1)
    sql = f"""
        SELECT
            f.{SOURCE['region_col']} AS regioncode,
            f.{SOURCE['invoice_date_col']}::date AS invoice_date,
            SUM(f.{SOURCE['value_col']})::numeric AS sellin_value
        FROM {SOURCE['fact_table']} f
        WHERE f.{SOURCE['invoice_date_col']} >=
              date_trunc('month', CURRENT_DATE) - INTERVAL '{months - 1} months'
          AND f.{SOURCE['invoice_date_col']} <
              date_trunc('month', CURRENT_DATE) + INTERVAL '1 month'
          AND f.{SOURCE['region_col']} IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 2, 1
    """
    df = read_sql(sql)
    if not df.empty:
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
        df["sellin_value"] = pd.to_numeric(df["sellin_value"], errors="coerce").fillna(0.0)
    return df


def get_region_monthly_targets(history_months: int = 36) -> pd.DataFrame:
    """Read region/month targets directly from mv_ai_region_monthly.

    No pro-rata branch target allocation is performed in V2.  The target
    source is authoritative at the same grain as the forecast.
    """
    months = max(int(history_months), 1)
    sql = f"""
        SELECT
            periode::date AS periode,
            regioncode,
            target_sellin
        FROM {SOURCE['region_target_view']}
        WHERE regioncode IS NOT NULL
          AND periode >= date_trunc('month', CURRENT_DATE) - INTERVAL '{months - 1} months'
          AND periode < date_trunc('month', CURRENT_DATE) + INTERVAL '1 month'
        ORDER BY periode, regioncode
    """
    df = read_sql(sql)
    if not df.empty:
        df["periode"] = pd.to_datetime(df["periode"])
        df["target_sellin"] = pd.to_numeric(df["target_sellin"], errors="coerce")
    return df


def get_calendar(history_months: int = 36) -> pd.DataFrame:
    """Return the daily calendar and networkeddays working-day indicator."""
    months = max(int(history_months), 1)
    sql = f"""
        SELECT
            dates::date AS date,
            years,
            months,
            daysofmonths,
            monthsname,
            {SOURCE['networked_days_col']} AS networkeddays
        FROM {SOURCE['date_dim']}
        WHERE dates >= date_trunc('month', CURRENT_DATE) - INTERVAL '{months - 1} months'
          AND dates < date_trunc('month', CURRENT_DATE) + INTERVAL '1 month'
        ORDER BY dates
    """
    df = read_sql(sql)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        # networkeddays is normalized to a numeric working-day flag.  This
        # accepts common 0/1, boolean and Y/N source representations.
        raw = df["networkeddays"]
        if pd.api.types.is_bool_dtype(raw):
            df["is_working_day"] = raw.astype(int)
        else:
            normalized = raw.astype(str).str.strip().str.lower()
            df["is_working_day"] = normalized.isin(
                ["1", "1.0", "true", "t", "y", "yes"]
            ).astype(int)
    return df
