"""Turn raw daily rows into the monthly / MTD structures the models need."""
import pandas as pd


def to_monthly(df: pd.DataFrame, group_cols: list, date_col: str = "invoice_date",
               value_col: str = "sellin_value") -> pd.DataFrame:
    d = df.copy()
    d["periode"] = d[date_col].values.astype("datetime64[M]")
    out = d.groupby(group_cols + ["periode"], as_index=False)[value_col].sum()
    return out.rename(columns={value_col: "monthly_value"})


def to_daily_by_grain(df: pd.DataFrame, group_cols: list, date_col: str = "invoice_date",
                       value_col: str = "sellin_value") -> pd.DataFrame:
    return df.groupby(group_cols + [date_col], as_index=False)[value_col].sum()


def allocate_branch_targets(region_targets: pd.DataFrame, branch_monthly: pd.DataFrame,
                             trailing_months: int = 3) -> pd.DataFrame:
    """
    Branch-level targets don't exist in the source schema you shared, so each
    region's monthly target is allocated to its branches pro-rata, using each
    branch's average share of regional sell-in over the trailing N *closed*
    months (the current, in-progress month is excluded so the share isn't
    skewed by a partial month).
    """
    months_sorted = sorted(branch_monthly["periode"].unique())
    closed_months = months_sorted[-(trailing_months + 1):-1] if len(months_sorted) > 1 else months_sorted
    recent = branch_monthly[branch_monthly["periode"].isin(closed_months)].copy()

    if recent.empty:
        branch_share = branch_monthly[["regioncode", "branchcode"]].drop_duplicates()
        branch_share["avg_share"] = 1.0
        branch_share["avg_share"] = branch_share.groupby("regioncode")["avg_share"].transform(
            lambda s: 1 / len(s)
        )
    else:
        region_totals = recent.groupby(["regioncode", "periode"])["monthly_value"].sum().rename("region_total")
        recent = recent.join(region_totals, on=["regioncode", "periode"])
        recent["share"] = recent["monthly_value"] / recent["region_total"].replace(0, pd.NA)

        branch_share = (
            recent.groupby(["regioncode", "branchcode"])["share"]
            .mean()
            .reset_index()
            .rename(columns={"share": "avg_share"})
        )
        branch_share["avg_share"] = branch_share.groupby("regioncode")["avg_share"].transform(
            lambda s: s / s.sum() if s.sum() else 1 / len(s)
        )

    out = region_targets.merge(branch_share, on="regioncode", how="left")
    out["avg_share"] = out["avg_share"].fillna(1.0)
    out["branch_target_sellin"] = out["target_sellin"] * out["avg_share"]
    return out
