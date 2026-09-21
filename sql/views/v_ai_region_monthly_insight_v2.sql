-- V2 canonical region monthly AI facts
-- Persisted forecast snapshot boundary: dwh_prod.forecast_sellin_eom.
-- This view is the single regional fact layer for Region, GM and CEO.
-- No forecasting logic is introduced here.

CREATE OR REPLACE VIEW dwh_prod.v_ai_region_monthly_insight_v2 AS
WITH active_hierarchy AS (
    SELECT
        regioncode,
        MAX(regionname) AS regionname
    FROM dwh_prod.m_sales_org_hierarchy
    WHERE is_active = TRUE
    GROUP BY regioncode
    HAVING COUNT(*) = 1
), base AS (
    SELECT
        f.periode,
        f.regioncode,
        h.regionname,
        f.target_sellin,
        f.mtd_value AS total_sellin,
        f.total_working_days,
        f.elapsed_working_days AS mtd_working_days,
        f.remaining_working_days,
        f.forecast_baseline,
        f.forecast_ets,
        f.forecast_sarima,
        f.forecast_xgboost,
        f.forecast_p10,
        f.forecast_p50,
        f.forecast_p90,
        f.achievement_pct_forecast,
        ROUND(f.forecast_p50 - f.target_sellin, 0) AS forecast_gap_to_target,
        ROUND(
            (f.forecast_p90 - f.forecast_p10)
            / NULLIF(f.forecast_p50, 0) * 100,
            2
        ) AS forecast_uncertainty_pct,
        ROUND(
            f.mtd_value / NULLIF(f.target_sellin, 0) * 100,
            2
        ) AS current_achievement_pct,
        GREATEST(f.target_sellin - f.forecast_p50, 0) AS forecast_shortfall,
        GREATEST(f.forecast_p50 - f.target_sellin, 0) AS forecast_surplus,
        GREATEST(
            f.forecast_baseline,
            f.forecast_ets,
            f.forecast_sarima,
            f.forecast_xgboost
        ) - LEAST(
            f.forecast_baseline,
            f.forecast_ets,
            f.forecast_sarima,
            f.forecast_xgboost
        ) AS model_spread
    FROM dwh_prod.forecast_sellin_eom f
    LEFT JOIN active_hierarchy h
        ON f.regioncode = h.regioncode
), classified AS (
    SELECT
        b.*,
        CASE
            WHEN b.achievement_pct_forecast IS NULL THEN 'NO_FORECAST_DATA'
            WHEN b.achievement_pct_forecast >= 100 THEN 'TARGET_ACHIEVED'
            WHEN b.achievement_pct_forecast >= 90 THEN 'NEAR_TARGET'
            WHEN b.achievement_pct_forecast >= 70 THEN 'AT_RISK'
            WHEN b.achievement_pct_forecast >= 50 THEN 'HIGH_RISK'
            ELSE 'CRITICAL'
        END AS performance_scenario,
        CASE
            WHEN b.forecast_p10 IS NULL
              OR b.forecast_p50 IS NULL
              OR b.forecast_p90 IS NULL
              OR b.target_sellin IS NULL THEN 'NO_FORECAST_DATA'
            WHEN b.forecast_p10 >= b.target_sellin
                THEN 'HIGH_CONFIDENCE_ABOVE_TARGET'
            WHEN b.forecast_p90 < b.target_sellin
                THEN 'HIGH_CONFIDENCE_BELOW_TARGET'
            WHEN b.forecast_p50 >= b.target_sellin
                THEN 'P50_ABOVE_TARGET_BUT_UNCERTAIN'
            ELSE 'P50_BELOW_TARGET_BUT_UNCERTAIN'
        END AS forecast_scenario
    FROM base b
), totals AS (
    SELECT
        periode,
        SUM(forecast_shortfall) AS total_forecast_shortfall
    FROM classified
    GROUP BY periode
)
SELECT
    c.periode,
    c.regioncode,
    c.regionname,
    c.target_sellin,
    c.total_sellin,
    c.total_working_days,
    c.mtd_working_days,
    c.remaining_working_days,
    c.forecast_baseline,
    c.forecast_ets,
    c.forecast_sarima,
    c.forecast_xgboost,
    c.forecast_p10,
    c.forecast_p50,
    c.forecast_p90,
    c.achievement_pct_forecast,
    c.forecast_gap_to_target,
    c.forecast_uncertainty_pct,
    c.current_achievement_pct,
    c.performance_scenario,
    CASE
        WHEN c.performance_scenario = 'TARGET_ACHIEVED' THEN 'Target Achieved'
        WHEN c.performance_scenario = 'NEAR_TARGET' THEN 'Near Target'
        WHEN c.performance_scenario = 'AT_RISK' THEN 'At Risk'
        WHEN c.performance_scenario = 'HIGH_RISK' THEN 'High Risk'
        WHEN c.performance_scenario = 'NO_FORECAST_DATA' THEN 'No Forecast Data'
        ELSE 'Critical'
    END AS performance_scenario_name,
    c.forecast_scenario,
    CASE
        WHEN c.performance_scenario = 'NO_FORECAST_DATA' THEN 'REVIEW'
        WHEN c.achievement_pct_forecast < 50 THEN 'CRITICAL'
        WHEN c.achievement_pct_forecast < 70 THEN 'HIGH'
        WHEN c.achievement_pct_forecast < 90 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS priority,
    ROUND(c.model_spread, 0) AS model_spread,
    ROUND(c.model_spread / NULLIF(c.forecast_p50, 0) * 100, 2)
        AS model_spread_pct_p50,
    ROUND(c.forecast_shortfall, 0) AS forecast_shortfall,
    ROUND(
        c.forecast_shortfall
        / NULLIF(t.total_forecast_shortfall, 0) * 100,
        2
    ) AS shortfall_contribution_pct
FROM classified c
JOIN totals t
  ON t.periode = c.periode;
