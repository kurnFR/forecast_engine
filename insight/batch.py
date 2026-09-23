"""Batch entry point for Region -> GM -> CEO V2 insight generation."""
from __future__ import annotations
import argparse
import logging

from .engine import InsightAgent
from .persistence import deactivate_failed, persist

logger = logging.getLogger(__name__)


def run(period: str | None = None) -> list[dict]:
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
    parser.add_argument("--period", help="YYYY-MM-DD period; default is latest available")
    args = parser.parse_args()
    results = run(args.period)
    print(f"Generated and persisted {len(results)} V2 insights.")


if __name__ == "__main__":
    main()
