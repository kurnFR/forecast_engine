"""Persist the V2 region-month forecast to PostgreSQL."""
import logging

import pandas as pd
from sqlalchemy import text

from config import OUTPUT_CONFIG
from db import get_engine

logger = logging.getLogger(__name__)
TABLE = OUTPUT_CONFIG["forecast_table"]
SCHEMA, NAME = TABLE.split(".")

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    regioncode text NOT NULL,
    periode date NOT NULL,
    mtd_value numeric(23,4),
    elapsed_working_days int,
    remaining_working_days int,
    total_working_days int,
    forecast_baseline numeric(23,4),
    forecast_ets numeric(23,4),
    forecast_sarima numeric(23,4),
    forecast_xgboost numeric(23,4),
    forecast_p10 numeric(23,4),
    forecast_p50 numeric(23,4),
    forecast_p90 numeric(23,4),
    target_sellin numeric(23,4),
    achievement_pct_forecast numeric(9,4),
    generated_at timestamp NOT NULL DEFAULT now(),
    CONSTRAINT {NAME}_pk PRIMARY KEY (regioncode, periode)
);
"""

NUMERIC_COLS = (
    "mtd_value",
    "forecast_baseline",
    "forecast_ets",
    "forecast_sarima",
    "forecast_xgboost",
    "forecast_p10",
    "forecast_p50",
    "forecast_p90",
    "target_sellin",
    "achievement_pct_forecast",
)
INT_COLS = (
    "elapsed_working_days",
    "remaining_working_days",
    "total_working_days",
)


def ensure_table() -> None:
    with get_engine().begin() as conn:
        conn.execute(text(DDL))


def _normalize_for_postgres(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize model output before staging so numeric columns stay numeric.

    Model fallbacks can return None/NaN or numeric-looking object values. If an
    object column is sent directly to pandas ``to_sql``, SQLAlchemy may infer a
    TEXT staging column, which then fails against the numeric production table.
    """
    out = df.copy()
    if "periode" in out.columns:
        out["periode"] = pd.to_datetime(out["periode"], errors="coerce").dt.date
    if "generated_at" in out.columns:
        out["generated_at"] = pd.to_datetime(out["generated_at"], errors="coerce")

    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in INT_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out



def verify_forecast_persistence(df: pd.DataFrame) -> None:
    """Verify the just-written forecast snapshot before AI insight generation.

    The AI layer must never run against a forecast row that was not actually
    persisted. This checks the exact region/period keys produced by the current
    run and, critically, the MTD value that feeds downstream AI facts.
    """
    if df.empty:
        return

    expected = _normalize_for_postgres(df)[["regioncode", "periode", "mtd_value"]].copy()
    expected["regioncode"] = expected["regioncode"].astype(str)
    expected = expected.drop_duplicates(["regioncode", "periode"])

    query = f"""
        SELECT regioncode, periode, mtd_value
        FROM {TABLE}
        WHERE periode = :periode
    """
    period = expected["periode"].iloc[0]
    actual = pd.read_sql(text(query), get_engine(), params={"periode": period})
    actual["regioncode"] = actual["regioncode"].astype(str)

    merged = expected.merge(
        actual,
        on=["regioncode", "periode"],
        how="left",
        suffixes=("_expected", "_actual"),
        indicator=True,
    )
    missing = merged[merged["_merge"] != "both"]
    if not missing.empty:
        keys = ", ".join(
            f"{row.regioncode}/{row.periode}" for row in missing.itertuples()
        )
        raise RuntimeError(
            f"Forecast persistence verification failed: missing rows {keys}"
        )

    for row in merged.itertuples():
        expected_value = row.mtd_value_expected
        actual_value = row.mtd_value_actual
        if pd.isna(expected_value) and pd.isna(actual_value):
            continue
        if pd.isna(expected_value) or pd.isna(actual_value):
            raise RuntimeError(
                f"Forecast persistence verification failed for "
                f"{row.regioncode}/{row.periode}: mtd_value mismatch"
            )
        if abs(float(expected_value) - float(actual_value)) > 0.0001:
            raise RuntimeError(
                f"Forecast persistence verification failed for "
                f"{row.regioncode}/{row.periode}: "
                f"expected mtd_value={expected_value}, actual={actual_value}"
            )

    logger.info(
        "Forecast persistence verified: %d region-month rows for %s.",
        len(expected),
        period,
    )

def write_forecast(df: pd.DataFrame) -> None:
    if df.empty:
        logger.warning("Forecast dataframe is empty - nothing written to %s.", TABLE)
        return

    ensure_table()
    df = _normalize_for_postgres(df)
    staging_name = f"{NAME}_staging"
    df.to_sql(staging_name, get_engine(), schema=SCHEMA, if_exists="replace", index=False)

    cols = list(df.columns)
    key_cols = ("regioncode", "periode")
    update_cols = [c for c in cols if c not in key_cols]
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

    upsert_sql = f"""
        INSERT INTO {TABLE} ({', '.join(cols)})
        SELECT {', '.join(cols)} FROM {SCHEMA}.{staging_name}
        ON CONFLICT (regioncode, periode)
        DO UPDATE SET {set_clause};
        DROP TABLE {SCHEMA}.{staging_name};
    """
    with get_engine().begin() as conn:
        conn.execute(text(upsert_sql))

    logger.info("Wrote %d region-month forecast rows to %s.", len(df), TABLE)
    verify_forecast_persistence(df)
