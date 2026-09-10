"""CLI entry point for the V2 region-month Sell-In forecast engine.

Usage:
    python main.py
    python main.py --dry-run
"""
import argparse
import logging

from data.validation import validate_forecast_output
from forecast.predict import run_prediction_pipeline
from forecast.train import run_training_pipeline
from output.postgres import write_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run the V2 Sell-In forecast engine.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and generate forecasts but do not write results to PostgreSQL.",
    )
    args = parser.parse_args()

    logger.info("=== Forecast engine V2: training ===")
    trained = run_training_pipeline()

    logger.info("=== Forecast engine V2: prediction ===")
    result = run_prediction_pipeline(trained)
    result = validate_forecast_output(result)

    month_label = result["periode"].iloc[0].strftime("%Y-%m") if not result.empty else "N/A"
    logger.info("Generated %d validated region-month forecasts for %s.", len(result), month_label)

    if args.dry_run:
        logger.info("=== Dry run: PostgreSQL write skipped ===")
        if not result.empty:
            logger.info("Forecast columns: %s", ", ".join(result.columns))
            logger.info("Forecast preview:\n%s", result.to_string(index=False))
        return

    logger.info("=== Forecast engine V2: writing to Postgres ===")
    write_forecast(result)
    logger.info("Done. %d region-month forecasts written for %s.", len(result), month_label)


if __name__ == "__main__":
    main()
