import json

import pytest

from insight.engine import _extract_json, build_prompt, validate


def base(level="REGION"):
    row = {
        "periode": "2026-09-01", "hierarchy_level": level,
        "entity_code": "ASWJWA1" if level == "REGION" else "GM-COMJAWA" if level == "GM" else "CEO",
        "entity_name": "ASW JAWA 1" if level == "REGION" else "GM COMMERCIAL JAWA PULAU" if level == "GM" else "CEO",
        "target_sellin": 24700000000, "mtd_actual": 7481451991,
        "forecast_p10": 13528200000, "forecast_p50": 21666500000, "forecast_p90": 25917000000,
        "achievement_pct_forecast": 87.7186, "forecast_gap_to_target": -3033500000,
        "forecast_uncertainty_pct": 54.0, "performance_scenario": "AT_RISK",
        "performance_scenario_name": "At Risk", "forecast_scenario": "P50_BELOW_TARGET_BUT_UNCERTAIN",
        "priority": "MEDIUM", "model_spread": 1962200000, "model_spread_pct_p50": 9.06,
        "forecast_shortfall": 3033500000, "shortfall_contribution_pct": 5.58,
        "largest_shortfall_regioncode": None, "largest_shortfall_regionname": None,
        "largest_region_shortfall": None, "largest_region_shortfall_pct": None,
        "largest_shortfall_gm_code": None, "largest_shortfall_gm_name": None,
        "largest_gm_shortfall": None, "largest_gm_shortfall_pct": None,
    }
    if level == "GM": row["gm_code"] = "GM-COMJAWA"
    return row


def valid_insight(row, identity=None, priority=None):
    level = row["hierarchy_level"]
    key = "entity_code" if level == "REGION" else "gm_code" if level == "GM" else "insight_level"
    expected = "CEO" if level == "CEO" else row[key]
    return {key: identity if identity is not None else expected, "ai_insight_category": "FORECAST_RISK",
            "ai_diagnosis": "Forecast P50 berada di bawah target dan masih berisiko.",
            "triggered_action_plan": "Pantau realisasi sell-in dan fokus pada percepatan eksekusi.",
            "priority": priority if priority is not None else row["priority"]}


def test_region_prompt_contains_v2_contract_and_no_v1_momentum():
    prompt = build_prompt(base())
    assert "v_ai_forecast_insight_input_v2" in prompt
    assert "Never use daily-rate or momentum forecasting logic" in prompt


def test_gm_prompt_uses_gm_identity_and_preserves_priority():
    prompt = build_prompt(base("GM"))
    assert "LEVEL: GM" in prompt
    assert '"gm_code":"GM-COMJAWA"' in prompt
    assert '"priority":"MEDIUM"' in prompt


def test_ceo_prompt_uses_ceo_identity():
    prompt = build_prompt(base("CEO"))
    assert "LEVEL: CEO" in prompt
    assert '"insight_level": "CEO"' in prompt


def test_validation_preserves_region_identity_and_priority():
    row = base("REGION"); insight = valid_insight(row); assert validate(row, insight) == insight


def test_validation_preserves_gm_identity_and_priority():
    row = base("GM"); insight = valid_insight(row); assert validate(row, insight) == insight


def test_validation_preserves_ceo_identity_and_priority():
    row = base("CEO"); insight = valid_insight(row); assert validate(row, insight) == insight


def test_validation_accepts_review_priority():
    row = base(); row["priority"] = "REVIEW"; insight = valid_insight(row, priority="REVIEW")
    assert validate(row, insight)["priority"] == "REVIEW"


def test_validation_rejects_changed_priority():
    row = base()
    with pytest.raises(RuntimeError, match="priority"): validate(row, valid_insight(row, priority="HIGH"))


def test_validation_rejects_invalid_priority():
    row = base(); row["priority"] = "URGENT"
    with pytest.raises(RuntimeError, match="invalid priority"): validate(row, valid_insight(row, priority="URGENT"))


def test_validation_rejects_changed_identity_for_all_levels():
    for level in ("REGION", "GM", "CEO"):
        row = base(level)
        with pytest.raises(RuntimeError, match="changed"): validate(row, valid_insight(row, identity="OTHER"))


def test_validation_rejects_extra_or_missing_fields():
    row = base(); insight = valid_insight(row); insight["unexpected"] = "x"
    with pytest.raises(RuntimeError, match="extra"): validate(row, insight)
    insight = valid_insight(row); del insight["ai_diagnosis"]
    with pytest.raises(RuntimeError, match="missing"): validate(row, insight)


def test_no_forecast_data_prompt_forbids_inventing_forecast_or_risk():
    row = base(); row["performance_scenario"] = "NO_FORECAST_DATA"; row["priority"] = "REVIEW"
    prompt = build_prompt(row)
    assert "Forecast belum tersedia" in prompt
    assert "do not manufacture a business risk" in prompt


def test_parser_accepts_hermes_extra_output_before_json():
    output = 'Berikut hasilnya:\n\n' + json.dumps(valid_insight(base()), ensure_ascii=False) + '\nselesai.'
    assert _extract_json(output)["entity_code"] == "ASWJWA1"


def test_parser_rejects_non_json_output():
    with pytest.raises(RuntimeError, match="did not return JSON"): _extract_json("Hermes gagal menghasilkan output")


def test_prompt_contains_no_v1_daily_rate_input_fields():
    prompt = build_prompt(base())
    assert "mtd_daily_run_rate" not in prompt
    assert "avg_7_working_days" not in prompt
    assert "momentum_factor" not in prompt


def test_review_priority_is_normalized_to_uppercase():
    row = base(); row["priority"] = "review"; insight = valid_insight(row, priority="review")
    assert validate(row, insight)["priority"] == "REVIEW"
