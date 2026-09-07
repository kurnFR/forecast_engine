"""Extraction of raw sell-in facts, region targets, and calendar data."""
import pandas as pd

from config import SOURCE
from db import read_sql


def get_daily_sellin(lookback_days: int = 400) -> pd.DataFrame:
    """
    Daily sell-in value per region + branch, joined via the branch dimension.
    NOTE: the join is fact.city = branch_dim.kota - verify this against your
    real customer/branch mapping (see config.SOURCE for how to change it).
    """
    sql = f"""
        SELECT
            d.regioncode,
            d.regionname,
            d.branchcode,
            d.branchname,
            f.{SOURCE['invoice_date_col']} AS invoice_date,
            SUM(f.{SOURCE['value_col']}) AS sellin_value
        FROM {SOURCE['fact_table']} f
        JOIN {SOURCE['branch_dim']} d
          ON f.{SOURCE['branch_join_col_fact']} = d.{SOURCE['branch_join_col_dim']}
        WHERE f.{SOURCE['invoice_date_col']} >= CURRENT_DATE - INTERVAL '{lookback_days} days'
        GROUP BY 1, 2, 3, 4, 5
        ORDER BY 5
    """
    df = read_sql(sql)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def get_region_monthly_targets() -> pd.DataFrame:
    """Region-level monthly target & actual sell-in, from the pre-built view."""
    sql = f"""
        SELECT
            periode,
            regioncode,
            regionname,
            totaltargetsellin AS target_sellin,
            totalrealsellin  AS actual_sellin
        FROM {SOURCE['region_target_view']}
        WHERE regioncode IS NOT NULL
        ORDER BY periode
    """
    df = read_sql(sql)
    df["periode"] = pd.to_datetime(df["periode"])
    return df


def get_calendar(lookback_days: int = 400) -> pd.DataFrame:
    """Calendar with a working-day flag, used to compute elapsed/remaining working days."""
    sql = f"""
        SELECT
            dates::date AS date,
            years, months, daysofmonths, monthsname,
            {SOURCE['working_day_col']} AS is_working_day
        FROM {SOURCE['date_dim']}
        WHERE dates >= CURRENT_DATE - INTERVAL '{lookback_days} days'
          AND dates <= (date_trunc('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day')
        ORDER BY dates
    """
    df = read_sql(sql)
    df["date"] = pd.to_datetime(df["date"])
    # normalize working-day flag to boolean regardless of source being text or numeric
    df["is_working_day"] = (
        df["is_working_day"].astype(str).str.strip().isin(["1", "1.0", "true", "True", "Y", "y"])
    )
    return df
