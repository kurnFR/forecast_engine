"""CLI entry point.

Usage:
    python main.py
"""
import logging

from forecast.train import run_training_pipeline
from forecast.predict import run_prediction_pipeline
from output.postgres import write_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Forecast engine: training ===")
    trained = run_training_pipeline()

    logger.info("=== Forecast engine: prediction ===")
    result = run_prediction_pipeline(trained)

    logger.info("=== Forecast engine: writing to Postgres ===")
    write_forecast(result)

    if not result.empty:
        month_label = result["periode"].iloc[0].strftime("%Y-%m")
    else:
        month_label = "N/A"
    logger.info("Done. %d (region, branch) forecasts generated for %s.", len(result), month_label)


if __name__ == "__main__":
    main()
