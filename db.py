"""Thin database access layer built on SQLAlchemy."""
import logging
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import SQLALCHEMY_URL

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(SQLALCHEMY_URL, pool_pre_ping=True)
    return _engine


def read_sql(query: str, params: Optional[dict] = None) -> pd.DataFrame:
    logger.debug("Running query: %s", query[:200].replace("\n", " "))
    with get_engine().connect() as conn:
        return pd.read_sql(text(query), conn, params=params or {})


def execute(sql: str, params: Optional[dict] = None) -> None:
    with get_engine().begin() as conn:
        conn.execute(text(sql), params or {})


def table_exists(schema: str, table: str) -> bool:
    q = """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = :schema AND table_name = :table
        )
    """
    df = read_sql(q, {"schema": schema, "table": table})
    return bool(df.iloc[0, 0])
