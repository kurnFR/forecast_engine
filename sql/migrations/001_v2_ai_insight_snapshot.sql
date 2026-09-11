-- Additive V2 snapshot contract for the existing AI Insight tables.
-- Safe to run repeatedly; no legacy columns are removed.

ALTER TABLE dwh_prod.ai_region_insight
    ADD COLUMN IF NOT EXISTS v2_target_sellin numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_mtd_actual numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_forecast_p10 numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_forecast_p50 numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_forecast_p90 numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_achievement_pct_forecast numeric(12,4),
    ADD COLUMN IF NOT EXISTS v2_forecast_gap_to_target numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_forecast_uncertainty_pct numeric(12,4),
    ADD COLUMN IF NOT EXISTS v2_performance_scenario varchar(100),
    ADD COLUMN IF NOT EXISTS v2_performance_scenario_name varchar(150),
    ADD COLUMN IF NOT EXISTS v2_forecast_scenario varchar(100),
    ADD COLUMN IF NOT EXISTS v2_model_spread numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_model_spread_pct_p50 numeric(12,4),
    ADD COLUMN IF NOT EXISTS v2_forecast_shortfall numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_shortfall_contribution_pct numeric(12,4),
    ADD COLUMN IF NOT EXISTS v2_ai_insight_category varchar(150);

ALTER TABLE dwh_prod.ai_gm_insight
    ADD COLUMN IF NOT EXISTS v2_target_sellin numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_mtd_actual numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_forecast_p10 numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_forecast_p50 numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_forecast_p90 numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_achievement_pct_forecast numeric(12,4),
    ADD COLUMN IF NOT EXISTS v2_forecast_gap_to_target numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_forecast_uncertainty_pct numeric(12,4),
    ADD COLUMN IF NOT EXISTS v2_performance_scenario varchar(100),
    ADD COLUMN IF NOT EXISTS v2_forecast_scenario varchar(100),
    ADD COLUMN IF NOT EXISTS v2_model_spread numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_model_spread_pct_p50 numeric(12,4),
    ADD COLUMN IF NOT EXISTS v2_forecast_shortfall numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_shortfall_contribution_pct numeric(12,4),
    ADD COLUMN IF NOT EXISTS v2_largest_shortfall_regioncode varchar(50),
    ADD COLUMN IF NOT EXISTS v2_largest_shortfall_regionname varchar(150),
    ADD COLUMN IF NOT EXISTS v2_largest_region_shortfall numeric(23,4),
    ADD COLUMN IF NOT EXISTS v2_largest_region_shortfall_pct numeric(12,4),
    ADD COLUMN IF NOT EXISTS v2_ai_insight_category varchar(150);

ALTER TABLE dwh_prod.ai_insight
    ADD COLUMN IF NOT EXISTS v2_snapshot jsonb,
    ADD COLUMN IF NOT EXISTS model_name varchar(150),
    ADD COLUMN IF NOT EXISTS prompt_version varchar(50);
