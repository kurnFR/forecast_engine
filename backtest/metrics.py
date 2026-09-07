"""Forecast accuracy metrics used for leakage-safe model selection."""
import numpy as np


def _arrays(y_true, y_pred):
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def mape(y_true, y_pred) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error; stable for uneven region volumes."""
    y_true, y_pred = _arrays(y_true, y_pred)
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denominator * 100)


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true, y_pred) -> float:
    """Mean signed error; positive means over-forecast."""
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.mean(y_pred - y_true))


def bias_pct(y_true, y_pred) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.nan
    return float(np.sum(y_pred - y_true) / denominator * 100)
