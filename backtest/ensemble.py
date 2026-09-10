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


def audit_ensemble(backtest_results: pd.DataFrame, best_models: pd.DataFrame, base_pairs: dict, xgb_pairs: dict, checkpoints) -> pd.DataFrame:
    """Evaluate the same inverse-WAPE ensemble used in production on OOS pairs."""
    rows = []
    for _, selection_row in best_models.iterrows():
        key = tuple(selection_row[col] for col in best_models.columns if col in [])
        # Group keys are taken from the configured grain columns below.
        group_cols = [col for col in ("regioncode", "periode") if col in selection_row.index]
        if not group_cols:
            group_cols = ["regioncode"] if "regioncode" in selection_row.index else []
        if not group_cols:
            continue
        key = tuple(selection_row[col] for col in group_cols)
        weights = _weights_for_row(selection_row)
        base_key = key
        if base_key not in base_pairs and len(base_key) == 1 and (base_key[0],) in base_pairs:
            base_key = (base_key[0],)
        if base_key not in xgb_pairs and len(base_key) == 1 and (base_key[0],) in xgb_pairs:
            pass

        ensemble_pairs = []
        for cp in checkpoints:
            model_pairs = {}
            for model in ("baseline", "ets", "sarima"):
                pairs = base_pairs.get(base_key, {}).get(model, {}).get(cp, [])
                if pairs:
                    model_pairs[model] = dict(pairs)
            xpairs = xgb_pairs.get(base_key, {}).get(cp, [])
            if xpairs:
                model_pairs["xgboost"] = dict(xpairs)
            if not model_pairs:
                continue
            common = set.intersection(*(set(pairs) for pairs in model_pairs.values()))
            for actual in sorted(common):
                # Each model-pair dict is keyed by actual value; duplicate actual
                # values in a checkpoint are not expected at this region/month grain.
                numerator = 0.0
                denominator = 0.0
                for model, weight in weights.items():
                    if model in model_pairs:
                        predicted = model_pairs[model][actual]
                        numerator += weight * predicted
                        denominator += weight
                if denominator > 0:
                    ensemble_pairs.append((actual, numerator / denominator))

        metrics = _metrics(ensemble_pairs)
        if metrics is None:
            continue
        row = {col: selection_row[col] for col in group_cols}
        row.update({f"ensemble_{metric}": value for metric, value in metrics.items()})
        selected = str(selection_row.get("best_model", "unknown"))
        row["selected_model"] = selected
        row["selected_wape"] = float(selection_row.get("best_model_score")) if pd.notna(selection_row.get("best_model_score")) else np.nan
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
            key,
            selected,
            selected_text,
            float(row["ensemble_wape"]),
            float(row["ensemble_mae"]),
            float(row["ensemble_rmse"]),
            float(row["ensemble_bias_pct"]),
            int(row["ensemble_observations"]),
        )
