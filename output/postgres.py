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
