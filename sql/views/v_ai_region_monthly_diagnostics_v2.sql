-- V2 region diagnostics compatibility view.
-- Canonical regional facts now live in v_ai_region_monthly_insight_v2.
-- Keep this name for existing downstream compatibility.

CREATE OR REPLACE VIEW dwh_prod.v_ai_region_monthly_diagnostics_v2 AS
SELECT
    periode,
    regioncode,
    regionname,
    target_sellin,
    total_sellin,
    mtd_working_days,
    remaining_working_days,
    forecast_baseline,
    forecast_ets,
    forecast_sarima,
    forecast_xgboost,
    forecast_p10,
    forecast_p50,
    forecast_p90,
    achievement_pct_forecast,
    forecast_gap_to_target,
    forecast_uncertainty_pct,
    performance_scenario,
    performance_scenario_name,
    forecast_scenario,
    priority,
    model_spread,
    model_spread_pct_p50,
    forecast_shortfall,
    shortfall_contribution_pct
FROM dwh_prod.v_ai_region_monthly_insight_v2;
