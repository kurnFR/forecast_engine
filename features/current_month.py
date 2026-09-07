"""Features describing how the current, in-progress month looks so far."""
from typing import Optional
import pandas as pd

from features.working_day import month_working_day_stats


def build_current_month_features(daily_df: pd.DataFrame, calendar: pd.DataFrame,
                                  group_cols: list, as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    as_of = as_of or pd.Timestamp.today().normalize()
    wd = month_working_day_stats(calendar, as_of)

    cur = daily_df[(daily_df["invoice_date"] >= wd["month_start"]) & (daily_df["invoice_date"] <= wd["as_of"])]
    mtd = (
        cur.groupby(group_cols, as_index=False)["sellin_value"]
        .sum()
        .rename(columns={"sellin_value": "mtd_value"})
    )

    mtd["elapsed_working_days"] = wd["elapsed_working_days"]
    mtd["remaining_working_days"] = wd["remaining_working_days"]
    mtd["total_working_days"] = wd["total_working_days"]
    mtd["avg_daily_rate"] = mtd["mtd_value"] / max(wd["elapsed_working_days"], 1)
    mtd["run_rate_forecast"] = mtd["avg_daily_rate"] * wd["total_working_days"]
    mtd["periode"] = wd["month_start"]
    return mtd
