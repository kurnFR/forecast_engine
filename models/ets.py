"""Exponential smoothing (Holt / Holt-Winters) forecast for a single series."""
import logging
from typing import Optional

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from config import MODEL_CONFIG

logger = logging.getLogger(__name__)


def forecast_ets(series: pd.Series, steps: int = 1) -> Optional[float]:
    """Forecast a monthly series with configured additive Holt-Winters ETS.

    The forecast engine operates on monthly observations, so the configured
    seasonal period is passed explicitly instead of relying on a pandas
    DatetimeIndex frequency. This keeps ETS deterministic even when callers
    provide a plain Series or an index without a known frequency.
    """
    if len(series) < 4 or series.isna().any():
        return None
    try:
        cfg = MODEL_CONFIG["ets"]
        seasonal_periods = cfg.get("seasonal_periods")
        model = ExponentialSmoothing(
            series.astype(float).values,
            trend=cfg["trend"],
            seasonal=cfg["seasonal"],
            seasonal_periods=seasonal_periods,
        )
        fit = model.fit(optimized=True)
        pred = float(fit.forecast(steps)[-1])
        if pred < 0 or pred > series.max() * 5:
            logger.warning("ETS forecast %.0f implausible for series (n=%d); discarding.", pred, len(series))
            return None
        return pred
    except Exception as e:
        logger.warning("ETS failed for series (n=%d): %s", len(series), e)
        return None
