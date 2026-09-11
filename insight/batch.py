"""Batch entry point for Region -> GM -> CEO V2 insight generation."""
import argparse
import logging

from .engine import InsightAgent
from .persistence import persist

logger = logging.getLogger(__name__)


def run(period: str | None = None) -> list[dict]:
    agent = InsightAgent(period)
    # Generate is ordered REGION, GM, CEO by the input view; CEO receives the
    # full lower-level context while deterministic values remain unchanged.
    results = agent.generate()
    persist(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hermes V2 Sell-In insights")
    parser.add_argument("--period", help="YYYY-MM-DD period; default is latest available")
    args = parser.parse_args()
    results = run(args.period)
    print(f"Generated and persisted {len(results)} V2 insights.")


if __name__ == "__main__":
    main()
