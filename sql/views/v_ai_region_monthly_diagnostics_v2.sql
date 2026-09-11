-- V2 region monthly business diagnostics
-- Deterministic facts for the AI insight layer. No narrative and no daily-rate forecast.
-- Includes candidate model forecasts so GM/CEO diagnostics can aggregate them.

CREATE OR REPLACE VIEW dwh_prod.v_ai_region_monthly_diagnostics_v2 AS
WITH base AS (
    SELECT
        r.*,
        GREATEST(r.target_sellin - r.forecast_p50, 0) AS forecast_shortfall,
        GREATEST(r.forecast_p50 - r.target_sellin, 0) AS forecast_surplus,
        GREATEST(
            r.forecast_baseline,
            r.forecast_ets,
            r.forecast_sarima,
            r.forecast_xgboost
        )
        - LEAST(
            r.forecast_baseline,
            r.forecast_ets,
            r.forecast_sarima,
            r.forecast_xgboost
        ) AS model_spread
    FROM dwh_prod.v_ai_region_monthly_insight_v2 r
), totals AS (
    SELECT
        periode,
        SUM(forecast_shortfall) AS total_forecast_shortfall
    FROM base
    GROUP BY periode
)
SELECT
    b.periode,
    b.regioncode,
    b.regionname,
    b.target_sellin,
    b.total_sellin,
    b.mtd_working_days,
    b.remaining_working_days,
    b.forecast_baseline,
    b.forecast_ets,
    b.forecast_sarima,
    b.forecast_xgboost,
    b.forecast_p10,
    b.forecast_p50,
    b.forecast_p90,
    b.achievement_pct_forecast,
    b.forecast_gap_to_target,
    b.forecast_uncertainty_pct,
    b.performance_scenario,
    b.performance_scenario_name,
    b.forecast_scenario,
    b.priority,
    ROUND(b.model_spread, 0) AS model_spread,
    ROUND(b.model_spread / NULLIF(b.forecast_p50, 0) * 100, 2)
        AS model_spread_pct_p50,
    ROUND(b.forecast_shortfall, 0) AS forecast_shortfall,
    ROUND(
        b.forecast_shortfall
        / NULLIF(t.total_forecast_shortfall, 0) * 100,
        2
    ) AS shortfall_contribution_pct
FROM base b
JOIN totals t
  ON b.periode = t.periode;
