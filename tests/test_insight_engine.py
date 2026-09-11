from insight.engine import build_prompt, validate


def base(level="REGION"):
    return {
        "periode": "2026-09-01",
        "hierarchy_level": level,
        "entity_code": "ASWJWA1" if level == "REGION" else "GM-COMJAWA" if level == "GM" else "CEO",
        "entity_name": "ASW JAWA 1" if level == "REGION" else "GM COMMERCIAL JAWA PULAU" if level == "GM" else "CEO",
        "target_sellin": 24700000000,
        "mtd_actual": 7481451991,
        "forecast_p10": 13528200000,
        "forecast_p50": 21666500000,
        "forecast_p90": 25917000000,
        "achievement_pct_forecast": 87.7186,
        "forecast_gap_to_target": -3033500000,
        "forecast_uncertainty_pct": 54.0,
        "performance_scenario": "AT_RISK",
        "performance_scenario_name": "At Risk",
        "forecast_scenario": "P50_BELOW_TARGET_BUT_UNCERTAIN",
        "priority": "MEDIUM",
        "model_spread": 1962200000,
        "model_spread_pct_p50": 9.06,
        "forecast_shortfall": 3033500000,
        "shortfall_contribution_pct": 5.58,
        "largest_shortfall_regioncode": None,
        "largest_shortfall_regionname": None,
        "largest_region_shortfall": None,
        "largest_region_shortfall_pct": None,
        "largest_shortfall_gm_code": None,
        "largest_shortfall_gm_name": None,
        "largest_gm_shortfall": None,
        "largest_gm_shortfall_pct": None,
    }


def test_region_prompt_contains_v2_contract_and_no_v1_momentum():
    prompt = build_prompt(base())
    assert "v_ai_forecast_insight_input_v2" in prompt
    assert "Never use daily-rate or momentum forecasting logic" in prompt


def test_region_validation_preserves_identity_and_priority():
    row = base()
    insight = {
        "entity_code": "ASWJWA1",
        "ai_insight_category": "FORECAST_RISK",
        "ai_diagnosis": "Forecast P50 berada di bawah target dan masih berisiko.",
        "triggered_action_plan": "Pantau realisasi sell-in dan fokus pada percepatan eksekusi.",
        "priority": "MEDIUM",
    }
    assert validate(row, insight) == insight


def test_validation_rejects_changed_priority():
    row = base()
    insight = {
        "entity_code": "ASWJWA1",
        "ai_insight_category": "FORECAST_RISK",
        "ai_diagnosis": "Diagnosis.",
        "triggered_action_plan": "Action.",
        "priority": "HIGH",
    }
    try:
        validate(row, insight)
        assert False, "expected validation failure"
    except RuntimeError as exc:
        assert "priority" in str(exc)
