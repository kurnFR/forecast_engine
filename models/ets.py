"""Exponential smoothing (Holt / Holt-Winters) forecast for a single series."""
import logging
from typing import Optional
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config import MODEL_CONFIG

logger = logging.getLogger(__name__)


def forecast_ets(series: pd.Series, steps: int = 1) -> Optional[float]:
    """series: monthly_value in chronological order, for one region/branch."""
    if len(series) < 4 or series.isna().any():
        return None
    try:
        cfg = MODEL_CONFIG["ets"]
        model = ExponentialSmoothing(series.values, trend=cfg["trend"], seasonal=cfg["seasonal"])
        fit = model.fit(optimized=True)
        pred = float(fit.forecast(steps)[-1])
        if pred < 0 or pred > series.max() * 5:
            logger.warning("ETS forecast %.0f implausible for series (n=%d); discarding.", pred, len(series))
            return None
        return pred
    except Exception as e:
        logger.warning("ETS failed for series (n=%d): %s", len(series), e)
        return None
