"""CLI entry point for the V2 region-month Sell-In forecast engine.

Usage:
    python main.py
"""
import logging

from forecast.predict import run_prediction_pipeline
from forecast.train import run_training_pipeline
from output.postgres import write_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Forecast engine V2: training ===")
    trained = run_training_pipeline()

    logger.info("=== Forecast engine V2: prediction ===")
    result = run_prediction_pipeline(trained)

    logger.info("=== Forecast engine V2: writing to Postgres ===")
    write_forecast(result)

    month_label = result["periode"].iloc[0].strftime("%Y-%m") if not result.empty else "N/A"
    logger.info("Done. %d region-month forecasts generated for %s.", len(result), month_label)


if __name__ == "__main__":
    main()
