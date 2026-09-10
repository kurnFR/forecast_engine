"""Leakage-safe out-of-sample audit for the production ensemble."""
import logging

import numpy as np
import pandas as pd

from backtest.metrics import bias, bias_pct, mae, rmse, wape
from config import CANDIDATE_MODELS

logger = logging.getLogger(__name__)


def _weights_for_row(row: pd.Series) -> dict:
    scores = {}
    for model in CANDIDATE_MODELS:
        value = row.get(f"{model}_checkpoint_score")
        if pd.notna(value) and np.isfinite(float(value)) and float(value) > 0:
            scores[model] = float(value)
    if not scores:
        return {"baseline": 1.0}
    inverse = {model: 1.0 / score for model, score in scores.items()}
    total = sum(inverse.values())
    return {model: value / total for model, value in inverse.items()}


def _metrics(pairs):
    if not pairs:
        return None
    actual, predicted = zip(*pairs)
    return {
        "wape": wape(actual, predicted),
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "bias": bias(actual, predicted),
        "bias_pct": bias_pct(actual, predicted),
        "observations": len(pairs),
    }


def audit_ensemble(backtest_results: pd.DataFrame, best_models: pd.DataFrame, base_pairs: dict, xgb_pairs: dict, checkpoints, group_cols) -> pd.DataFrame:
    """Evaluate the same inverse-WAPE ensemble used in production on OOS pairs.

    Model predictions are aligned by target month and checkpoint, never by
    list position or actual-value matching.
    """
    rows = []
    for _, selection_row in best_models.iterrows():
        key = tuple(selection_row[col] for col in group_cols)
        weights = _weights_for_row(selection_row)
        ensemble_pairs = []

        for cp in checkpoints:
            model_maps = {}
            for model in ("baseline", "ets", "sarima"):
                entries = base_pairs.get(key, {}).get(model, {}).get(cp, [])
                model_maps[model] = {pd.Timestamp(month): (float(actual), float(pred)) for month, actual, pred in entries}
            entries = xgb_pairs.get(key, {}).get(cp, [])
            model_maps["xgboost"] = {pd.Timestamp(month): (float(actual), float(pred)) for month, actual, pred in entries}

            available_maps = {model: values for model, values in model_maps.items() if values}
            if not available_maps:
                continue
            common_months = set.intersection(*(set(values) for values in available_maps.values()))
            for month in sorted(common_months):
                numerator = 0.0
                denominator = 0.0
                actual_value = None
                for model, weight in weights.items():
                    if model not in available_maps or month not in available_maps[model]:
                        continue
                    actual, predicted = available_maps[model][month]
                    if actual_value is None:
                        actual_value = actual
                    elif not np.isclose(actual_value, actual, rtol=1e-9, atol=1.0):
                        raise ValueError(f"OOS actual mismatch for {key} {month} WD{cp}")
                    numerator += weight * predicted
                    denominator += weight
                if denominator > 0 and actual_value is not None:
                    ensemble_pairs.append((actual_value, numerator / denominator))

        metrics = _metrics(ensemble_pairs)
        if metrics is None:
            continue
        row = {col: selection_row[col] for col in group_cols}
        row.update({f"ensemble_{metric}": value for metric, value in metrics.items()})
        row["selected_model"] = str(selection_row.get("best_model", "unknown"))
        selected_wape = selection_row.get("best_model_score")
        row["selected_wape"] = float(selected_wape) if pd.notna(selected_wape) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def log_ensemble_audit(audit_results: pd.DataFrame) -> None:
    """Log ensemble OOS metrics for production-quality review."""
    logger.info("=== Ensemble OOS quality audit ===")
    if audit_results.empty:
        logger.warning("Ensemble OOS quality audit produced no valid pairs.")
        return
    for _, row in audit_results.iterrows():
        key = row.get("regioncode", "unknown")
        selected = row.get("selected_model", "unknown")
        selected_wape = row.get("selected_wape")
        selected_text = f"{float(selected_wape):.2f}%" if pd.notna(selected_wape) else "NA"
        logger.info(
            "regioncode=%s | selected=%s (WAPE=%s) | ensemble: WAPE=%.2f%%, MAE=%.2f, RMSE=%.2f, bias=%.2f%%, n=%d",
            key, selected, selected_text,
            float(row["ensemble_wape"]), float(row["ensemble_mae"]),
            float(row["ensemble_rmse"]), float(row["ensemble_bias_pct"]),
            int(row["ensemble_observations"]),
        )
