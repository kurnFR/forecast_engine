"""Working-day arithmetic used by the run-rate baseline and as ML features."""
from typing import Optional
import pandas as pd


def month_working_day_stats(calendar: pd.DataFrame, as_of: Optional[pd.Timestamp] = None) -> dict:
    """Total working days in the current month, elapsed (through as_of), and remaining."""
    as_of = as_of or pd.Timestamp.today().normalize()
    month_start = as_of.replace(day=1)
    month_end = month_start + pd.offsets.MonthEnd(1)

    m = calendar[(calendar["date"] >= month_start) & (calendar["date"] <= month_end)]
    total_wd = int(m["is_working_day"].sum())
    elapsed_wd = int(m.loc[m["date"] <= as_of, "is_working_day"].sum())
    remaining_wd = max(total_wd - elapsed_wd, 0)

    return {
        "month_start": month_start,
        "month_end": month_end,
        "as_of": as_of,
        "total_working_days": total_wd,
        "elapsed_working_days": elapsed_wd,
        "remaining_working_days": remaining_wd,
    }


def historical_working_days_per_month(calendar_history: pd.DataFrame) -> pd.DataFrame:
    """Total working days for every month in a longer calendar pull (used as an XGBoost feature)."""
    c = calendar_history.copy()
    c["periode"] = c["date"].values.astype("datetime64[M]")
    return c.groupby("periode", as_index=False)["is_working_day"].sum().rename(
        columns={"is_working_day": "total_working_days"}
    )
