"""SARIMA forecast for a single series - needs more history than ETS, so
falls back to None gracefully when a series is too short."""
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import MODEL_CONFIG

logger = logging.getLogger(__name__)


def forecast_sarima(series: pd.Series, steps: int = 1) -> Optional[float]:
    cfg = MODEL_CONFIG["sarima"]
    min_needed = max(cfg["seasonal_order"][3] * 2, 12)
    if len(series) < min_needed or series.isna().any():
        return None

    values = series.astype(float).to_numpy()
    if not np.isfinite(values).all() or np.all(values == 0):
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                values,
                order=cfg["order"],
                seasonal_order=cfg["seasonal_order"],
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)

            # statsmodels can return an apparently valid numeric forecast from
            # a poorly converged optimization. Treat that as unavailable rather
            # than allowing an unstable candidate into model selection/ensemble.
            mle_retvals = getattr(fit, "mle_retvals", {}) or {}
            if mle_retvals.get("converged") is False:
                logger.warning(
                    "SARIMA fit did not converge for series (n=%d); discarding.",
                    len(series),
                )
                return None

            pred = float(fit.forecast(steps)[-1])

            # Use a robust level bound rather than only max(series): a single
            # historical spike should not make an unstable forecast acceptable.
            median = float(np.median(values))
            recent = values[-min(12, len(values)):]
            recent_median = float(np.median(recent))
            level = max(median, recent_median, 1.0)
            upper_bound = max(float(np.max(values)) * 2.5, level * 4.0)

            if not np.isfinite(pred) or pred < 0 or pred > upper_bound:
                logger.warning(
                    "SARIMA forecast %.0f implausible for series (n=%d, bound=%.0f); discarding.",
                    pred,
                    len(series),
                    upper_bound,
                )
                return None
            return pred
    except Exception as e:
        logger.warning("SARIMA failed for series (n=%d): %s", len(series), e)
        return None
