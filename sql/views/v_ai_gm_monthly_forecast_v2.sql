-- V2 GM monthly forecast aggregation
-- Aggregates authoritative regional targets and V2 regional forecasts.
-- Forecasts are never re-estimated at GM level; they are the sum of
-- region-level P10/P50/P90 outputs.

CREATE OR REPLACE VIEW dwh_prod.v_ai_gm_monthly_forecast_v2 AS
WITH aggregated AS (
    SELECT
        r.periode,
        m.gm_code,
        m.gm_name,
        COUNT(DISTINCT r.regioncode) AS region_count,
        SUM(r.target_sellin) AS target_sellin,
        SUM(r.total_sellin) AS total_sellin,
        MAX(r.total_working_days) AS total_working_days,
        MAX(r.mtd_working_days) AS mtd_working_days,
        MAX(r.remaining_working_days) AS remaining_working_days,
        SUM(r.forecast_baseline) AS forecast_baseline,
        SUM(r.forecast_ets) AS forecast_ets,
        SUM(r.forecast_sarima) AS forecast_sarima,
        SUM(r.forecast_xgboost) AS forecast_xgboost,
        SUM(r.forecast_p10) AS forecast_p10,
        SUM(r.forecast_p50) AS forecast_p50,
        SUM(r.forecast_p90) AS forecast_p90
    FROM dwh_prod.v_ai_region_monthly_insight_v2 r
    JOIN dwh_prod.m_sales_org_hierarchy m
      ON r.regioncode = m.regioncode
     AND m.is_active = TRUE
    GROUP BY r.periode, m.gm_code, m.gm_name
), derived AS (
    SELECT
        a.*,
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
        ) AS forecast_uncertainty_pct
    FROM aggregated a
)
SELECT
    d.periode,
    d.gm_code,
    d.gm_name,
    d.region_count,
    d.target_sellin,
    d.total_sellin,
    d.total_working_days,
    d.mtd_working_days,
    d.remaining_working_days,
    d.forecast_baseline,
    d.forecast_ets,
    d.forecast_sarima,
    d.forecast_xgboost,
    d.forecast_p10,
    d.forecast_p50,
    d.forecast_p90,
    d.current_achievement_pct,
    d.achievement_pct_forecast,
    d.forecast_gap_to_target,
    d.forecast_uncertainty_pct,
    CASE
        WHEN d.achievement_pct_forecast >= 100 THEN 'TARGET_ACHIEVED'
        WHEN d.achievement_pct_forecast >= 90 THEN 'NEAR_TARGET'
        WHEN d.achievement_pct_forecast >= 70 THEN 'AT_RISK'
        WHEN d.achievement_pct_forecast >= 50 THEN 'HIGH_RISK'
        ELSE 'CRITICAL'
    END AS performance_scenario,
    CASE
        WHEN d.forecast_p10 >= d.target_sellin
            THEN 'HIGH_CONFIDENCE_ABOVE_TARGET'
        WHEN d.forecast_p90 < d.target_sellin
            THEN 'HIGH_CONFIDENCE_BELOW_TARGET'
        WHEN d.forecast_p50 >= d.target_sellin
            THEN 'P50_ABOVE_TARGET_BUT_UNCERTAIN'
        ELSE 'P50_BELOW_TARGET_BUT_UNCERTAIN'
    END AS forecast_scenario,
    CASE
        WHEN d.achievement_pct_forecast < 50 THEN 'CRITICAL'
        WHEN d.achievement_pct_forecast < 70 THEN 'HIGH'
        WHEN d.achievement_pct_forecast < 90 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS priority
FROM derived d;
