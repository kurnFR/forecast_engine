-- Unified V2 AI insight input contract
-- One deterministic interface for Region, GM and CEO AI insight generation.
-- AI consumes these facts and may explain/prioritize them, but must not alter
-- target, actual, forecast, achievement, gap, scenario, or priority values.
-- Working-calendar fields are sourced from the existing region insight view;
-- no calendar logic is duplicated in Hermes.

CREATE OR REPLACE VIEW dwh_prod.v_ai_forecast_insight_input_v2 AS
WITH working_calendar AS (
    SELECT
        periode,
        MAX(total_working_days) AS total_working_days,
        MAX(mtd_working_days) AS mtd_working_days,
        MAX(remaining_working_days) AS remaining_working_days
    FROM dwh_prod.v_ai_region_monthly_insight_v2
    GROUP BY periode
)
SELECT
    r.periode,
    'REGION'::text AS hierarchy_level,
    r.regioncode AS entity_code,
    r.regionname AS entity_name,
    r.target_sellin,
    r.total_sellin AS mtd_actual,
    r.forecast_p10,
    r.forecast_p50,
    r.forecast_p90,
    r.achievement_pct_forecast,
    r.forecast_gap_to_target,
    r.forecast_uncertainty_pct,
    r.performance_scenario,
    r.performance_scenario_name,
    r.forecast_scenario,
    r.priority,
    r.model_spread,
    r.model_spread_pct_p50,
    r.forecast_shortfall,
    r.shortfall_contribution_pct,
    NULL::text AS largest_shortfall_regioncode,
    NULL::text AS largest_shortfall_regionname,
    NULL::numeric AS largest_region_shortfall,
    NULL::numeric AS largest_region_shortfall_pct,
    NULL::text AS largest_shortfall_gm_code,
    NULL::text AS largest_shortfall_gm_name,
    NULL::numeric AS largest_gm_shortfall,
    NULL::numeric AS largest_gm_shortfall_pct,
    wc.total_working_days,
    wc.mtd_working_days,
    wc.remaining_working_days,
    ROUND(r.total_sellin / NULLIF(r.target_sellin, 0) * 100, 2) AS mtd_achievement_pct,
    CASE
        WHEN r.performance_scenario IN ('AT_RISK', 'HIGH_RISK', 'CRITICAL') THEN TRUE
        WHEN r.performance_scenario IN ('TARGET_ACHIEVED', 'NEAR_TARGET', 'NO_FORECAST_DATA') THEN FALSE
        ELSE NULL
    END AS focus_required
FROM dwh_prod.v_ai_region_monthly_diagnostics_v2 r
LEFT JOIN working_calendar wc ON wc.periode = r.periode

UNION ALL

SELECT
    g.periode,
    'GM'::text AS hierarchy_level,
    g.gm_code AS entity_code,
    g.gm_name AS entity_name,
    g.target_sellin,
    g.total_sellin AS mtd_actual,
    g.forecast_p10,
    g.forecast_p50,
    g.forecast_p90,
    g.achievement_pct_forecast,
    g.forecast_gap_to_target,
    g.forecast_uncertainty_pct,
    g.performance_scenario,
    NULL::text AS performance_scenario_name,
    g.forecast_scenario,
    g.priority,
    g.max_region_model_spread AS model_spread,
    g.max_region_model_spread_pct_p50 AS model_spread_pct_p50,
    g.forecast_shortfall,
    g.shortfall_contribution_pct,
    g.largest_shortfall_regioncode,
    g.largest_shortfall_regionname,
    g.largest_region_shortfall,
    g.largest_region_shortfall_pct_gm,
    NULL::text AS largest_shortfall_gm_code,
    NULL::text AS largest_shortfall_gm_name,
    NULL::numeric AS largest_gm_shortfall,
    NULL::numeric AS largest_gm_shortfall_pct,
    wc.total_working_days,
    wc.mtd_working_days,
    wc.remaining_working_days,
    ROUND(g.total_sellin / NULLIF(g.target_sellin, 0) * 100, 2) AS mtd_achievement_pct,
    CASE
        WHEN g.performance_scenario IN ('AT_RISK', 'HIGH_RISK', 'CRITICAL') THEN TRUE
        WHEN g.performance_scenario IN ('TARGET_ACHIEVED', 'NEAR_TARGET', 'NO_FORECAST_DATA') THEN FALSE
        ELSE NULL
    END AS focus_required
FROM dwh_prod.v_ai_gm_monthly_diagnostics_v2 g
LEFT JOIN working_calendar wc ON wc.periode = g.periode

UNION ALL

SELECT
    c.periode,
    'CEO'::text AS hierarchy_level,
    'CEO'::text AS entity_code,
    'CEO'::text AS entity_name,
    c.target_sellin,
    c.total_sellin AS mtd_actual,
    c.forecast_p10,
    c.forecast_p50,
    c.forecast_p90,
    c.achievement_pct_forecast,
    c.forecast_gap_to_target,
    c.forecast_uncertainty_pct,
    c.performance_scenario,
    NULL::text AS performance_scenario_name,
    c.forecast_scenario,
    c.priority,
    c.max_region_model_spread AS model_spread,
    c.max_region_model_spread_pct_p50 AS model_spread_pct_p50,
    c.forecast_shortfall,
    c.shortfall_contribution_pct,
    c.largest_shortfall_regioncode,
    c.largest_shortfall_regionname,
    c.largest_region_shortfall,
    c.largest_region_shortfall_pct_ceo,
    c.largest_shortfall_gm_code,
    c.largest_shortfall_gm_name,
    c.largest_gm_shortfall,
    c.largest_gm_shortfall_pct_ceo,
    wc.total_working_days,
    wc.mtd_working_days,
    wc.remaining_working_days,
    ROUND(c.total_sellin / NULLIF(c.target_sellin, 0) * 100, 2) AS mtd_achievement_pct,
    CASE
        WHEN c.performance_scenario IN ('AT_RISK', 'HIGH_RISK', 'CRITICAL') THEN TRUE
        WHEN c.performance_scenario IN ('TARGET_ACHIEVED', 'NEAR_TARGET', 'NO_FORECAST_DATA') THEN FALSE
        ELSE NULL
    END AS focus_required
FROM dwh_prod.v_ai_ceo_monthly_diagnostics_v2 c
LEFT JOIN working_calendar wc ON wc.periode = c.periode;