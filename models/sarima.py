"""SARIMA forecast for a single series - needs more history than ETS, so
falls back to None gracefully when a series is too short."""
import logging
import warnings
from typing import Optional
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import MODEL_CONFIG

logger = logging.getLogger(__name__)


def forecast_sarima(series: pd.Series, steps: int = 1) -> Optional[float]:
    cfg = MODEL_CONFIG["sarima"]
    min_needed = max(cfg["seasonal_order"][3] * 2, 12)
    if len(series) < min_needed or series.isna().any():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                series.values,
                order=cfg["order"],
                seasonal_order=cfg["seasonal_order"],
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)
            pred = float(fit.forecast(steps)[-1])
            # Guard against unstable fits on short/noisy series producing an
            # implausible (e.g. negative or wildly out-of-range) forecast.
            if pred < 0 or pred > series.max() * 5:
                logger.warning("SARIMA forecast %.0f implausible for series (n=%d); discarding.", pred, len(series))
                return None
            return pred
    except Exception as e:
        logger.warning("SARIMA failed for series (n=%d): %s", len(series), e)
        return None
