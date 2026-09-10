-- V2 CEO monthly business diagnostics
-- Deterministic enterprise-level facts derived from regional V2 diagnostics.
-- No independent CEO forecasting model and no daily-rate forecast.

CREATE OR REPLACE VIEW dwh_prod.v_ai_ceo_monthly_diagnostics_v2 AS
WITH base AS (
    SELECT
        r.periode,
        r.regioncode,
        r.regionname,
        r.target_sellin,
        r.total_sellin,
        r.forecast_baseline,
        r.forecast_ets,
        r.forecast_sarima,
        r.forecast_xgboost,
        r.forecast_p10,
        r.forecast_p50,
        r.forecast_p90,
        r.achievement_pct_forecast,
        r.forecast_gap_to_target,
        r.forecast_uncertainty_pct,
        r.performance_scenario,
        r.forecast_scenario,
        r.priority,
        r.model_spread,
        r.model_spread_pct_p50,
        r.forecast_shortfall
    FROM dwh_prod.v_ai_region_monthly_diagnostics_v2 r
), aggregated AS (
    SELECT
        periode,
        COUNT(DISTINCT regioncode) AS region_count,
        SUM(target_sellin) AS target_sellin,
        SUM(total_sellin) AS total_sellin,
        SUM(forecast_baseline) AS forecast_baseline,
        SUM(forecast_ets) AS forecast_ets,
        SUM(forecast_sarima) AS forecast_sarima,
        SUM(forecast_xgboost) AS forecast_xgboost,
        SUM(forecast_p10) AS forecast_p10,
        SUM(forecast_p50) AS forecast_p50,
        SUM(forecast_p90) AS forecast_p90,
        SUM(forecast_shortfall) AS forecast_shortfall,
        MAX(model_spread) AS max_region_model_spread,
        MAX(model_spread_pct_p50) AS max_region_model_spread_pct_p50
    FROM base
    GROUP BY periode
), ranked_regions AS (
    SELECT
        b.*,
        ROW_NUMBER() OVER (
            PARTITION BY b.periode
            ORDER BY b.forecast_shortfall DESC, b.regioncode
        ) AS shortfall_rank
    FROM base b
), ranked_gms AS (
    SELECT
        g.*,
        ROW_NUMBER() OVER (
            PARTITION BY g.periode
            ORDER BY g.forecast_gap_to_target ASC, g.gm_code
        ) AS gm_rank
    FROM dwh_prod.v_ai_gm_monthly_diagnostics_v2 g
)
SELECT
    a.periode,
    a.region_count,
    a.target_sellin,
    a.total_sellin,
    a.forecast_baseline,
    a.forecast_ets,
    a.forecast_sarima,
    a.forecast_xgboost,
    a.forecast_p10,
    a.forecast_p50,
    a.forecast_p90,
    ROUND(a.total_sellin / NULLIF(a.target_sellin, 0) * 100, 2)
        AS current_achievement_pct,
    ROUND(a.forecast_p50 / NULLIF(a.target_sellin, 0) * 100, 2)
        AS achievement_pct_forecast,
    ROUND(a.forecast_p50 - a.target_sellin, 0)
        AS forecast_gap_to_target,
    ROUND(
        (a.forecast_p90 - a.forecast_p10)
        / NULLIF(a.forecast_p50, 0) * 100,
        2
    ) AS forecast_uncertainty_pct,
    CASE
        WHEN a.forecast_p50 / NULLIF(a.target_sellin, 0) * 100 >= 100 THEN 'TARGET_ACHIEVED'
        WHEN a.forecast_p50 / NULLIF(a.target_sellin, 0) * 100 >= 90 THEN 'NEAR_TARGET'
        WHEN a.forecast_p50 / NULLIF(a.target_sellin, 0) * 100 >= 70 THEN 'AT_RISK'
        WHEN a.forecast_p50 / NULLIF(a.target_sellin, 0) * 100 >= 50 THEN 'HIGH_RISK'
        ELSE 'CRITICAL'
    END AS performance_scenario,
    CASE
        WHEN a.forecast_p10 >= a.target_sellin THEN 'HIGH_CONFIDENCE_ABOVE_TARGET'
        WHEN a.forecast_p90 < a.target_sellin THEN 'HIGH_CONFIDENCE_BELOW_TARGET'
        WHEN a.forecast_p50 >= a.target_sellin THEN 'P50_ABOVE_TARGET_BUT_UNCERTAIN'
        ELSE 'P50_BELOW_TARGET_BUT_UNCERTAIN'
    END AS forecast_scenario,
    CASE
        WHEN a.forecast_p50 / NULLIF(a.target_sellin, 0) * 100 < 50 THEN 'CRITICAL'
        WHEN a.forecast_p50 / NULLIF(a.target_sellin, 0) * 100 < 70 THEN 'HIGH'
        WHEN a.forecast_p50 / NULLIF(a.target_sellin, 0) * 100 < 90 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS priority,
    ROUND(a.forecast_shortfall, 0) AS forecast_shortfall,
    ROUND(a.forecast_shortfall / NULLIF(a.forecast_shortfall, 0) * 100, 2)
        AS shortfall_contribution_pct,
    ROUND(a.max_region_model_spread, 0) AS max_region_model_spread,
    ROUND(a.max_region_model_spread_pct_p50, 2) AS max_region_model_spread_pct_p50,
    rr.regioncode AS largest_shortfall_regioncode,
    rr.regionname AS largest_shortfall_regionname,
    ROUND(rr.forecast_shortfall, 0) AS largest_region_shortfall,
    ROUND(
        rr.forecast_shortfall / NULLIF(a.forecast_shortfall, 0) * 100,
        2
    ) AS largest_region_shortfall_pct_ceo,
    rg.gm_code AS largest_shortfall_gm_code,
    rg.gm_name AS largest_shortfall_gm_name,
    ROUND(GREATEST(-rg.forecast_gap_to_target, 0), 0) AS largest_gm_shortfall,
    ROUND(
        GREATEST(-rg.forecast_gap_to_target, 0)
        / NULLIF(a.forecast_shortfall, 0) * 100,
        2
    ) AS largest_gm_shortfall_pct_ceo
FROM aggregated a
LEFT JOIN ranked_regions rr
  ON a.periode = rr.periode
 AND rr.shortfall_rank = 1
LEFT JOIN ranked_gms rg
  ON a.periode = rg.periode
 AND rg.gm_rank = 1;
