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


def _month_maps(base_pairs, xgb_pairs, key, checkpoint):
    """Return month -> (actual, prediction) maps for every candidate model."""
    maps = {}
    for model in ("baseline", "ets", "sarima"):
        entries = base_pairs.get(key, {}).get(model, {}).get(checkpoint, [])
        maps[model] = {
            pd.Timestamp(month): (float(actual), float(predicted))
            for month, actual, predicted in entries
        }
    entries = xgb_pairs.get(key, {}).get(checkpoint, [])
    maps["xgboost"] = {
        pd.Timestamp(month): (float(actual), float(predicted))
        for month, actual, predicted in entries
    }
    return maps


def audit_ensemble(backtest_results: pd.DataFrame, best_models: pd.DataFrame, base_pairs: dict, xgb_pairs: dict, checkpoints, group_cols) -> pd.DataFrame:
    """Evaluate production ensemble and candidate models on identical OOS pairs.

    The comparison is deliberately restricted to target-month/checkpoint pairs
    where every candidate model has a valid prediction. This makes ensemble vs
    selected-model WAPE directly comparable and exposes any prediction
    availability gap separately via each model's full observation count.
    """
    rows = []
    for _, selection_row in best_models.iterrows():
        key = tuple(selection_row[col] for col in group_cols)
        weights = _weights_for_row(selection_row)
        common_pairs = {model: [] for model in CANDIDATE_MODELS}
        ensemble_pairs = []
        all_model_counts = {model: 0 for model in CANDIDATE_MODELS}

        for cp in checkpoints:
            model_maps = _month_maps(base_pairs, xgb_pairs, key, cp)
            for model in CANDIDATE_MODELS:
                all_model_counts[model] += len(model_maps[model])

            non_empty = [values for values in model_maps.values() if values]
            if len(non_empty) != len(CANDIDATE_MODELS):
                continue
            common_months = set.intersection(*(set(values) for values in model_maps.values()))

            for month in sorted(common_months):
                actual_value = None
                predictions = {}
                for model in CANDIDATE_MODELS:
                    actual, predicted = model_maps[model][month]
                    if actual_value is None:
                        actual_value = actual
                    elif not np.isclose(actual_value, actual, rtol=1e-9, atol=1.0):
                        raise ValueError(f"OOS actual mismatch for {key} {month} WD{cp}")
                    predictions[model] = predicted
                    common_pairs[model].append((actual, predicted))

                numerator = sum(weights.get(model, 0.0) * predictions[model] for model in CANDIDATE_MODELS)
                denominator = sum(weights.get(model, 0.0) for model in CANDIDATE_MODELS)
                if denominator > 0 and actual_value is not None:
                    ensemble_pairs.append((actual_value, numerator / denominator))

        ensemble_metrics = _metrics(ensemble_pairs)
        if ensemble_metrics is None:
            continue

        row = {col: selection_row[col] for col in group_cols}
        row["selected_model"] = str(selection_row.get("best_model", "unknown"))
        row["selected_wape_full"] = float(selection_row.get("best_model_score")) if pd.notna(selection_row.get("best_model_score")) else np.nan
        row["common_observations"] = len(ensemble_pairs)
        for model in CANDIDATE_MODELS:
            metrics = _metrics(common_pairs[model])
            if metrics is None:
                continue
            for metric, value in metrics.items():
                row[f"{model}_common_{metric}"] = value
            row[f"{model}_full_observations"] = all_model_counts[model]
        for metric, value in ensemble_metrics.items():
            row[f"ensemble_{metric}"] = value
        selected_common = _metrics(common_pairs.get(row["selected_model"], []))
        row["selected_wape_common"] = selected_common["wape"] if selected_common else np.nan
        row["ensemble_delta_wape_vs_selected"] = (
            ensemble_metrics["wape"] - selected_common["wape"]
            if selected_common else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def log_ensemble_audit(audit_results: pd.DataFrame) -> None:
    """Log fair common-sample model vs ensemble OOS metrics."""
    logger.info("=== Ensemble OOS quality audit ===")
    if audit_results.empty:
        logger.warning("Ensemble OOS quality audit produced no valid common pairs.")
        return
    for _, row in audit_results.iterrows():
        key = row.get("regioncode", "unknown")
        selected = row.get("selected_model", "unknown")
        common_n = int(row["common_observations"])
        parts = []
        for model in CANDIDATE_MODELS:
            common_wape = row.get(f"{model}_common_wape")
            full_n = row.get(f"{model}_full_observations")
            common_bias = row.get(f"{model}_common_bias_pct")
            if pd.notna(common_wape):
                parts.append(
                    f"{model}: WAPE={float(common_wape):.2f}%, bias={float(common_bias):.2f}%, "
                    f"n={common_n}, full_n={int(full_n)}"
                )
        delta = row.get("ensemble_delta_wape_vs_selected")
        delta_text = f"{float(delta):+.2f}pp" if pd.notna(delta) else "NA"
        logger.info(
            "regioncode=%s | common_n=%d | %s | ensemble: WAPE=%.2f%%, bias=%.2f%%, n=%d | "
            "delta_vs_selected=%s",
            key,
            common_n,
            " | ".join(parts),
            float(row["ensemble_wape"]),
            float(row["ensemble_bias_pct"]),
            int(row["ensemble_observations"]),
            delta_text,
        )
