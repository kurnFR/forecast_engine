-- V2 GM monthly business diagnostics
-- Deterministic management facts derived from regional V2 diagnostics.
-- No second forecasting model and no daily-rate forecast.
--
-- GM P50 is the sum of regional P50 forecasts.
-- GM P10/P90 are NOT sums of regional quantiles. They are reconstructed
-- from regional 10-90 widths using a normal/independence approximation:
--   sigma_i ~= (P90_i - P10_i) / (2 * z90)
--   sigma_GM = sqrt(sum(sigma_i^2))
--   GM P10/P90 = GM P50 +/- z90 * sigma_GM
-- This is an interim aggregate uncertainty method and must be validated
-- with GM-level historical backtesting before production sign-off.

CREATE OR REPLACE VIEW dwh_prod.v_ai_gm_monthly_diagnostics_v2 AS
WITH active_hierarchy AS (
    SELECT
        regioncode,
        MAX(gm_code) AS gm_code,
        MAX(gm_name) AS gm_name
    FROM dwh_prod.m_sales_org_hierarchy
    WHERE is_active = TRUE
    GROUP BY regioncode
    HAVING COUNT(*) = 1
), base AS (
    SELECT
        r.periode,
        h.gm_code,
        h.gm_name,
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
        r.forecast_shortfall,
        CASE
            WHEN r.forecast_p10 IS NOT NULL
             AND r.forecast_p90 IS NOT NULL
             AND r.forecast_p90 >= r.forecast_p10
            THEN (r.forecast_p90 - r.forecast_p10) / (2.0 * 1.2815515655446)
            ELSE NULL
        END AS regional_sigma
    FROM dwh_prod.v_ai_region_monthly_insight_v2 r
    JOIN active_hierarchy h
      ON r.regioncode = h.regioncode
), aggregated AS (
    SELECT
        periode,
        gm_code,
        gm_name,
        COUNT(DISTINCT regioncode) AS region_count,
        SUM(target_sellin) AS target_sellin,
        SUM(total_sellin) AS total_sellin,
        SUM(forecast_baseline) AS forecast_baseline,
        SUM(forecast_ets) AS forecast_ets,
        SUM(forecast_sarima) AS forecast_sarima,
        SUM(forecast_xgboost) AS forecast_xgboost,
        SUM(forecast_p50) AS forecast_p50,
        GREATEST(
            SUM(forecast_p50)
            - 1.2815515655446 * SQRT(SUM(POWER(regional_sigma, 2))),
            0
        ) AS forecast_p10,
        SUM(forecast_p50)
            + 1.2815515655446 * SQRT(SUM(POWER(regional_sigma, 2)))
            AS forecast_p90,
        SUM(forecast_shortfall) AS forecast_shortfall,
        MAX(model_spread) AS max_region_model_spread,
        MAX(model_spread_pct_p50) AS max_region_model_spread_pct_p50
    FROM base
    GROUP BY periode, gm_code, gm_name
), ranked_regions AS (
    SELECT
        b.*,
        ROW_NUMBER() OVER (
            PARTITION BY b.periode, b.gm_code
            ORDER BY b.forecast_shortfall DESC, b.regioncode
        ) AS shortfall_rank
    FROM base b
)
SELECT
    a.periode,
    a.gm_code,
    a.gm_name,
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
        WHEN a.forecast_p50 IS NULL OR a.target_sellin IS NULL THEN 'NO_FORECAST_DATA'
        WHEN a.forecast_p50 / NULLIF(a.target_sellin, 0) * 100 >= 100 THEN 'TARGET_ACHIEVED'
        WHEN a.forecast_p50 / NULLIF(a.target_sellin, 0) * 100 >= 90 THEN 'NEAR_TARGET'
        WHEN a.forecast_p50 / NULLIF(a.target_sellin, 0) * 100 >= 70 THEN 'AT_RISK'
        WHEN a.forecast_p50 / NULLIF(a.target_sellin, 0) * 100 >= 50 THEN 'HIGH_RISK'
        ELSE 'CRITICAL'
    END AS performance_scenario,
    CASE
        WHEN a.forecast_p10 IS NULL OR a.forecast_p50 IS NULL OR a.forecast_p90 IS NULL
            OR a.target_sellin IS NULL THEN 'NO_FORECAST_DATA'
        WHEN a.forecast_p10 >= a.target_sellin THEN 'HIGH_CONFIDENCE_ABOVE_TARGET'
        WHEN a.forecast_p90 < a.target_sellin THEN 'HIGH_CONFIDENCE_BELOW_TARGET'
        WHEN a.forecast_p50 >= a.target_sellin THEN 'P50_ABOVE_TARGET_BUT_UNCERTAIN'
        ELSE 'P50_BELOW_TARGET_BUT_UNCERTAIN'
    END AS forecast_scenario,
    CASE
        WHEN a.forecast_p50 IS NULL OR a.target_sellin IS NULL THEN 'REVIEW'
        WHEN a.forecast_p50 / NULLIF(a.target_sellin, 0) * 100 < 50 THEN 'CRITICAL'
        WHEN a.forecast_p50 / NULLIF(a.target_sellin, 0) * 100 < 70 THEN 'HIGH'
        WHEN a.forecast_p50 / NULLIF(a.target_sellin, 0) * 100 < 90 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS priority,
    ROUND(a.forecast_shortfall, 0) AS forecast_shortfall,
    ROUND(
        a.forecast_shortfall
        / NULLIF(SUM(a.forecast_shortfall) OVER (PARTITION BY a.periode), 0)
        * 100,
        2
    ) AS shortfall_contribution_pct,
    ROUND(a.max_region_model_spread, 0) AS max_region_model_spread,
    ROUND(a.max_region_model_spread_pct_p50, 2) AS max_region_model_spread_pct_p50,
    rr.regioncode AS largest_shortfall_regioncode,
    rr.regionname AS largest_shortfall_regionname,
    ROUND(rr.forecast_shortfall, 0) AS largest_region_shortfall,
    ROUND(
        rr.forecast_shortfall / NULLIF(a.forecast_shortfall, 0) * 100,
        2
    ) AS largest_region_shortfall_pct_gm
FROM aggregated a
LEFT JOIN ranked_regions rr
  ON a.periode = rr.periode
 AND a.gm_code = rr.gm_code
 AND rr.shortfall_rank = 1;
