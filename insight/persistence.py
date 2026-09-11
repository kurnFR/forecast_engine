"""Persistence adapter for the existing AI Insight tables.

V2 snapshot columns are additive; legacy columns remain populated where their
semantics are still compatible, while obsolete V1 daily-rate/momentum fields
are deliberately left NULL.
"""
import json
from typing import Any

from db import get_engine
from sqlalchemy import text


def persist(results: list[dict[str, Any]]) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        for item in results:
            row = item["input"]
            insight = item["insight"]
            level = row["hierarchy_level"]
            period = row["periode"]
            if level == "REGION":
                conn.execute(text("UPDATE dwh_prod.ai_region_insight SET is_active=0 WHERE periode=:p AND regioncode=:c AND is_active=1"), {"p": period, "c": row["entity_code"]})
                conn.execute(text("""
                    INSERT INTO dwh_prod.ai_region_insight
                    (periode, regioncode, regionname, ai_scenario_code, performance_scenario,
                     projected_achievement_status, priority, target_sellin, total_sellin,
                     projected_month_end_sellin, projected_gap_to_target, ai_diagnosis,
                     triggered_action_plan, model_name, prompt_version, insight_status, is_active,
                     v2_target_sellin, v2_mtd_actual, v2_forecast_p10, v2_forecast_p50, v2_forecast_p90,
                     v2_achievement_pct_forecast, v2_forecast_gap_to_target, v2_forecast_uncertainty_pct,
                     v2_performance_scenario, v2_performance_scenario_name, v2_forecast_scenario,
                     v2_model_spread, v2_model_spread_pct_p50, v2_forecast_shortfall,
                     v2_shortfall_contribution_pct)
                    VALUES (:p,:c,:n,:cat,:ps,:ps,:pri,:target,:actual,:p50,:gap,:diag,:action,
                            :model,:prompt,'GENERATED',1,:target,:actual,:p10,:p50,:p90,:ach,:gap,:unc,
                            :ps,:psn,:fs,:spread,:spreadpct,:shortfall,:contrib)
                """), {"p":period,"c":row["entity_code"],"n":row["entity_name"],"cat":insight["ai_insight_category"],"ps":row["performance_scenario"],"pri":row["priority"],"target":row["target_sellin"],"actual":row["mtd_actual"],"p50":row["forecast_p50"],"gap":row["forecast_gap_to_target"],"diag":insight["ai_diagnosis"],"action":insight["triggered_action_plan"],"model": "hermes-bi-insight","prompt": "v2-forecast","p10":row["forecast_p10"],"p90":row["forecast_p90"],"ach":row["achievement_pct_forecast"],"unc":row["forecast_uncertainty_pct"],"psn":row["performance_scenario_name"],"fs":row["forecast_scenario"],"spread":row["model_spread"],"spreadpct":row["model_spread_pct_p50"],"shortfall":row["forecast_shortfall"],"contrib":row["shortfall_contribution_pct"]})
            elif level == "GM":
                conn.execute(text("UPDATE dwh_prod.ai_gm_insight SET is_active=0 WHERE periode=:p AND gm_code=:c AND is_active=1"), {"p": period, "c": row["entity_code"]})
                conn.execute(text("""
                    INSERT INTO dwh_prod.ai_gm_insight
                    (periode, gm_code, ai_insight_category, ai_diagnosis, triggered_action_plan,
                     priority, model_name, prompt_version, insight_status, is_active,
                     v2_target_sellin, v2_mtd_actual, v2_forecast_p10, v2_forecast_p50, v2_forecast_p90,
                     v2_achievement_pct_forecast, v2_forecast_gap_to_target, v2_forecast_uncertainty_pct,
                     v2_performance_scenario, v2_forecast_scenario, v2_model_spread,
                     v2_model_spread_pct_p50, v2_forecast_shortfall, v2_shortfall_contribution_pct,
                     v2_largest_shortfall_regioncode, v2_largest_shortfall_regionname,
                     v2_largest_region_shortfall, v2_largest_region_shortfall_pct)
                    VALUES (:p,:c,:cat,:diag,:action,:pri,:model,:prompt,'GENERATED',1,
                            :target,:actual,:p10,:p50,:p90,:ach,:gap,:unc,:ps,:fs,:spread,:spreadpct,
                            :shortfall,:contrib,:lr,:lrn,:ls,:lsp)
                """), {"p":period,"c":row["entity_code"],"cat":insight["ai_insight_category"],"diag":insight["ai_diagnosis"],"action":insight["triggered_action_plan"],"pri":row["priority"],"model":"hermes-bi-insight","prompt":"v2-forecast","target":row["target_sellin"],"actual":row["mtd_actual"],"p10":row["forecast_p10"],"p50":row["forecast_p50"],"p90":row["forecast_p90"],"ach":row["achievement_pct_forecast"],"gap":row["forecast_gap_to_target"],"unc":row["forecast_uncertainty_pct"],"ps":row["performance_scenario"],"fs":row["forecast_scenario"],"spread":row["model_spread"],"spreadpct":row["model_spread_pct_p50"],"shortfall":row["forecast_shortfall"],"contrib":row["shortfall_contribution_pct"],"lr":row["largest_shortfall_regioncode"],"lrn":row["largest_shortfall_regionname"],"ls":row["largest_region_shortfall"],"lsp":row["largest_region_shortfall_pct"]})
            else:
                conn.execute(text("UPDATE dwh_prod.ai_insight SET is_active=0 WHERE insight_type='CEO_SALES_SUMMARY' AND is_active=1"))
                conn.execute(text("""
                    INSERT INTO dwh_prod.ai_insight
                    ("period", insight_type, description, ai_summary, is_active, v2_snapshot, model_name, prompt_version)
                    VALUES (:p,'CEO_SALES_SUMMARY',:description,:summary,1,:snapshot,:model,:prompt)
                """), {"p":period,"description":json.dumps(insight,ensure_ascii=False),"summary":f"{insight['ai_diagnosis']} {insight['triggered_action_plan']}","snapshot":json.dumps(row,ensure_ascii=False,default=str),"model":"hermes-bi-insight","prompt":"v2-forecast"})
