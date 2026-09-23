"""Persistence adapter for the existing AI Insight tables.

V2 forecast fields are authoritative. Legacy Region columns are retained only
for backward-compatible identity/status fields and are populated from the same
V2 scenario/priority values where semantics remain compatible. Obsolete V1
momentum and daily-rate fields are intentionally not written.
"""
from __future__ import annotations
import json
import math
from typing import Any

from db import get_engine
from sqlalchemy import text


def _json_safe(value: Any) -> Any:
    """Convert values that PostgreSQL JSON cannot represent to JSON-safe values."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_dumps(value: Any) -> str:
    """Serialize JSON using standards-compliant values only."""
    return json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False, default=str)


def persist(results: list[dict[str, Any]]) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        for item in results:
            row = item["input"]
            insight = item["insight"]
            level = row["hierarchy_level"]
            period = row["periode"]

            if level == "REGION":
                conn.execute(text("""
                    UPDATE dwh_prod.ai_region_insight
                    SET is_active = 0
                    WHERE periode = :p AND regioncode = :c AND is_active = 1
                """), {"p": period, "c": row["entity_code"]})
                conn.execute(text("""
                    INSERT INTO dwh_prod.ai_region_insight
                    (
                        periode, regioncode, regionname, ai_scenario_code,
                        performance_scenario, projected_month_end_sellin,
                        projected_gap_to_target, projected_achievement_status, priority,
                        target_sellin, total_sellin, ai_diagnosis, triggered_action_plan,
                        model_name, prompt_version, insight_status, is_active,
                        v2_target_sellin, v2_mtd_actual, v2_mtd_achievement_pct,
                        v2_forecast_p10, v2_forecast_p50, v2_forecast_p90,
                        v2_achievement_pct_forecast, v2_forecast_gap_to_target,
                        v2_forecast_uncertainty_pct, v2_performance_scenario,
                        v2_performance_scenario_name, v2_forecast_scenario,
                        v2_model_spread, v2_model_spread_pct_p50, v2_forecast_shortfall,
                        v2_shortfall_contribution_pct, v2_focus_required, v2_ai_insight_category
                    )
                    VALUES
                    (
                        :p, :c, :n, :scenario, :scenario, :p50, :gap, :scenario, :pri,
                        :target, :actual, :diag, :action, :model, :prompt, 'GENERATED', 1,
                        :target, :actual, :mtdach, :p10, :p50, :p90, :ach, :gap, :unc,
                        :scenario, :psn, :fs, :spread, :spreadpct, :shortfall, :contrib,
                        :focus, :category
                    )
                """), {
                    "p": period, "c": row["entity_code"], "n": row["entity_name"],
                    "scenario": row["performance_scenario"], "pri": row["priority"],
                    "target": row["target_sellin"], "actual": row["mtd_actual"],
                    "mtdach": row["mtd_achievement_pct"], "p50": row["forecast_p50"],
                    "gap": row["forecast_gap_to_target"], "diag": insight["ai_diagnosis"],
                    "action": insight["triggered_action_plan"], "model": "hermes-bi-insight",
                    "prompt": "v2-forecast", "p10": row["forecast_p10"], "p90": row["forecast_p90"],
                    "ach": row["achievement_pct_forecast"], "unc": row["forecast_uncertainty_pct"],
                    "psn": row["performance_scenario_name"], "fs": row["forecast_scenario"],
                    "spread": row["model_spread"], "spreadpct": row["model_spread_pct_p50"],
                    "shortfall": row["forecast_shortfall"], "contrib": row["shortfall_contribution_pct"],
                    "focus": row.get("focus_required"), "category": insight["ai_insight_category"],
                })

            elif level == "GM":
                conn.execute(text("""
                    UPDATE dwh_prod.ai_gm_insight
                    SET is_active = 0
                    WHERE periode = :p AND gm_code = :c AND is_active = 1
                """), {"p": period, "c": row["entity_code"]})
                conn.execute(text("""
                    INSERT INTO dwh_prod.ai_gm_insight
                    (
                        periode, gm_code, ai_insight_category, ai_diagnosis,
                        triggered_action_plan, priority, model_name, prompt_version,
                        insight_status, is_active, v2_target_sellin, v2_mtd_actual,
                        v2_mtd_achievement_pct, v2_forecast_p10, v2_forecast_p50, v2_forecast_p90,
                        v2_achievement_pct_forecast, v2_forecast_gap_to_target,
                        v2_forecast_uncertainty_pct, v2_performance_scenario, v2_forecast_scenario,
                        v2_model_spread, v2_model_spread_pct_p50, v2_forecast_shortfall,
                        v2_shortfall_contribution_pct, v2_largest_shortfall_regioncode,
                        v2_largest_shortfall_regionname, v2_largest_region_shortfall,
                        v2_largest_region_shortfall_pct, v2_focus_required, v2_ai_insight_category
                    )
                    VALUES
                    (
                        :p, :c, :category, :diag, :action, :pri, :model, :prompt, 'GENERATED', 1,
                        :target, :actual, :mtdach, :p10, :p50, :p90, :ach, :gap, :unc,
                        :scenario, :fs, :spread, :spreadpct, :shortfall, :contrib,
                        :lr, :lrn, :ls, :lsp, :focus, :category
                    )
                """), {
                    "p": period, "c": row["entity_code"], "category": insight["ai_insight_category"],
                    "diag": insight["ai_diagnosis"], "action": insight["triggered_action_plan"],
                    "pri": row["priority"], "model": "hermes-bi-insight", "prompt": "v2-forecast",
                    "target": row["target_sellin"], "actual": row["mtd_actual"],
                    "mtdach": row["mtd_achievement_pct"], "p10": row["forecast_p10"],
                    "p50": row["forecast_p50"], "p90": row["forecast_p90"],
                    "ach": row["achievement_pct_forecast"], "gap": row["forecast_gap_to_target"],
                    "unc": row["forecast_uncertainty_pct"], "scenario": row["performance_scenario"],
                    "fs": row["forecast_scenario"], "spread": row["model_spread"],
                    "spreadpct": row["model_spread_pct_p50"], "shortfall": row["forecast_shortfall"],
                    "contrib": row["shortfall_contribution_pct"], "lr": row["largest_shortfall_regioncode"],
                    "lrn": row["largest_shortfall_regionname"], "ls": row["largest_region_shortfall"],
                    "lsp": row["largest_region_shortfall_pct"], "focus": row.get("focus_required"),
                })

            else:
                conn.execute(text("""
                    UPDATE dwh_prod.ai_insight
                    SET is_active = 0
                    WHERE insight_type = 'CEO_SALES_SUMMARY' AND is_active = 1
                """))
                conn.execute(text("""
                    INSERT INTO dwh_prod.ai_insight
                    ("period", insight_type, description, ai_summary, is_active,
                     v2_snapshot, model_name, prompt_version)
                    VALUES
                    (:p, 'CEO_SALES_SUMMARY', :description, :summary, 1,
                     :snapshot, :model, :prompt)
                """), {
                    "p": period,
                    "description": _json_dumps(row),
                    "summary": f"{insight['ai_diagnosis']} {insight['triggered_action_plan']}",
                    "snapshot": _json_dumps(insight),
                    "model": "hermes-bi-insight", "prompt": "v2-forecast",
                })


def deactivate_failed(failures: list[dict[str, Any]]) -> None:
    """Remove stale active snapshots for entities whose current generation failed.

    A failed generation must never leave an older snapshot marked active, because
    that row contains facts from an earlier forecast refresh and can be mistaken
    for the current authoritative AI snapshot.
    """
    if not failures:
        return

    engine = get_engine()
    with engine.begin() as conn:
        for item in failures:
            row = item["input"]
            level = row["hierarchy_level"]
            period = row["periode"]

            if level == "REGION":
                conn.execute(text("""
                    UPDATE dwh_prod.ai_region_insight
                    SET is_active = 0
                    WHERE periode = :p AND regioncode = :c AND is_active = 1
                """), {"p": period, "c": row["entity_code"]})

            elif level == "GM":
                conn.execute(text("""
                    UPDATE dwh_prod.ai_gm_insight
                    SET is_active = 0
                    WHERE periode = :p AND gm_code = :c AND is_active = 1
                """), {"p": period, "c": row["entity_code"]})

            elif level == "CEO":
                conn.execute(text("""
                    UPDATE dwh_prod.ai_insight
                    SET is_active = 0
                    WHERE insight_type = 'CEO_SALES_SUMMARY' AND is_active = 1
                """))
