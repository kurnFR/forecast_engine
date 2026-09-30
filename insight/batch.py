"""Batch entry point for Region -> GM -> CEO V2 insight generation."""
from __future__ import annotations

import argparse
from datetime import date
import logging

from .engine import InsightAgent
from db import read_sql
from .persistence import deactivate_failed, persist

logger = logging.getLogger(__name__)



def verify_ai_input_freshness(period: str) -> None:
    """Ensure REGION AI inputs expose the same MTD values as the persisted forecast."""
    query = """
        SELECT
            f.regioncode,
            f.mtd_value AS forecast_mtd,
            r.mtd_actual AS ai_mtd
        FROM dwh_prod.forecast_sellin_eom f
        LEFT JOIN dwh_prod.v_ai_forecast_insight_input_v2 r
          ON r.entity_code = f.regioncode
         AND r.periode = f.periode
         AND r.hierarchy_level = 'REGION'
        WHERE f.periode = :period
        ORDER BY f.regioncode
    """
    rows = read_sql(query, {"period": period})
    if rows.empty:
        raise RuntimeError(
            f"AI input freshness verification failed: no forecast rows for {period}"
        )

    missing = rows[rows["ai_mtd"].isna() & rows["forecast_mtd"].notna()]
    if not missing.empty:
        codes = ", ".join(missing["regioncode"].astype(str))
        raise RuntimeError(
            f"AI input freshness verification failed: missing REGION input rows: {codes}"
        )

    mismatched = rows[
        rows["forecast_mtd"].notna()
        & rows["ai_mtd"].notna()
        & ((rows["forecast_mtd"].astype(float) - rows["ai_mtd"].astype(float)).abs() > 0.0001)
    ]
    if not mismatched.empty:
        details = ", ".join(
            f"{row.regioncode} forecast={row.forecast_mtd} ai={row.ai_mtd}"
            for row in mismatched.itertuples()
        )
        raise RuntimeError(
            f"AI input freshness verification failed: canonical MTD mismatch: {details}"
        )

    logger.info(
        "AI input freshness verified: %d REGION rows for %s.",
        len(rows),
        period,
    )

def current_month_period() -> str:
    """Return the first day of the current calendar month as YYYY-MM-DD."""
    today = date.today()
    return today.replace(day=1).isoformat()


def run(period: str | None = None) -> list[dict]:
    period = period or current_month_period()
    logger.info("Running V2 insight batch for period %s.", period)
    verify_ai_input_freshness(period)

    agent = InsightAgent(period)
    # Generate is ordered REGION, GM, CEO by the input view; CEO receives the
    # full lower-level context while deterministic values remain unchanged.
    results = agent.generate(continue_on_error=True)
    if results:
        persist(results)
    if agent.failures:
        # Never leave an older active snapshot looking current when the current
        # authoritative row failed AI generation/validation.
        deactivate_failed(agent.failures)
        failed = ", ".join(
            f"{x['input'].get('hierarchy_level')}:{x['input'].get('entity_code', x['input'].get('gm_code', 'CEO'))}"
            for x in agent.failures
        )
        raise RuntimeError(
            f"V2 insight batch completed with {len(agent.failures)} failed row(s): {failed}"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hermes V2 Sell-In insights")
    parser.add_argument(
        "--period",
        help="YYYY-MM-DD period; default is the first day of the current month",
    )
    args = parser.parse_args()
    results = run(args.period)
    print(f"Generated and persisted {len(results)} V2 insights.")


if __name__ == "__main__":
    main()
