"""Transform daily region-level Sell-In into model-ready monthly series."""
import pandas as pd


def to_monthly(
    df: pd.DataFrame,
    group_cols: list,
    date_col: str = "invoice_date",
    value_col: str = "sellin_value",
) -> pd.DataFrame:
    """Aggregate daily Sell-In to one row per forecast grain and month."""
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["periode", "monthly_value"])

    d = df.copy()
    d["periode"] = pd.to_datetime(d[date_col]).dt.to_period("M").dt.to_timestamp()
    out = (
        d.groupby(group_cols + ["periode"], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "monthly_value"})
    )
    return out.sort_values(group_cols + ["periode"]).reset_index(drop=True)


def complete_month_panel(
    monthly: pd.DataFrame,
    group_cols: list,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Create a complete monthly panel and treat missing months as zero.

    Missing months must be explicit before lag/seasonality features are built;
    otherwise a missing sales month would be interpreted as a shorter time
    interval and distort both time-series and ML models.
    """
    if monthly.empty:
        return monthly.copy()

    months = pd.date_range(start=start, end=end, freq="MS")
    groups = monthly[group_cols].drop_duplicates().copy()
    groups["_key"] = 1
    calendar = pd.DataFrame({"periode": months, "_key": 1})
    panel = groups.merge(calendar, on="_key", how="inner").drop(columns="_key")
    panel = panel.merge(monthly, on=group_cols + ["periode"], how="left")
    panel["monthly_value"] = pd.to_numeric(panel["monthly_value"], errors="coerce").fillna(0.0)
    return panel.sort_values(group_cols + ["periode"]).reset_index(drop=True)
