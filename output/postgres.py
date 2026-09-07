"""Write the forecast DataFrame to Postgres (creates table if missing, upserts by grain+periode)."""
import logging
import pandas as pd
from sqlalchemy import text

from db import get_engine
from config import OUTPUT_CONFIG

logger = logging.getLogger(__name__)

TABLE = OUTPUT_CONFIG["forecast_table"]
SCHEMA, NAME = TABLE.split(".")

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    regioncode text NOT NULL,
    branchcode text NOT NULL,
    periode date NOT NULL,
    mtd_value numeric(23,4),
    elapsed_working_days int,
    remaining_working_days int,
    total_working_days int,
    forecast_baseline numeric(23,4),
    forecast_ets numeric(23,4),
    forecast_sarima numeric(23,4),
    forecast_xgboost numeric(23,4),
    forecast_ensemble numeric(23,4),
    branch_target_sellin numeric(23,4),
    achievement_pct_forecast numeric(9,4),
    generated_at timestamp NOT NULL DEFAULT now(),
    CONSTRAINT {NAME}_pk PRIMARY KEY (regioncode, branchcode, periode)
);
"""


def ensure_table() -> None:
    with get_engine().begin() as conn:
        conn.execute(text(DDL))


def write_forecast(df: pd.DataFrame) -> None:
    if df.empty:
        logger.warning("Forecast dataframe is empty - nothing written to %s.", TABLE)
        return

    ensure_table()
    staging_name = f"{NAME}_staging"

    df.to_sql(staging_name, get_engine(), schema=SCHEMA, if_exists="replace", index=False)

    cols = list(df.columns)
    update_cols = [c for c in cols if c not in ("regioncode", "branchcode", "periode")]
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

    upsert_sql = f"""
        INSERT INTO {TABLE} ({', '.join(cols)})
        SELECT {', '.join(cols)} FROM {SCHEMA}.{staging_name}
        ON CONFLICT (regioncode, branchcode, periode)
        DO UPDATE SET {set_clause};
        DROP TABLE {SCHEMA}.{staging_name};
    """
    with get_engine().begin() as conn:
        conn.execute(text(upsert_sql))

    logger.info("Wrote %d forecast rows to %s.", len(df), TABLE)
