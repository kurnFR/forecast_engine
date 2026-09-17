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


def valid_insight(row, identity=None, priority=None, category=None):
    level = row["hierarchy_level"]
    key = "entity_code" if level == "REGION" else "gm_code" if level == "GM" else "insight_level"
    expected = "CEO" if level == "CEO" else row[key]
    return {key: identity if identity is not None else expected,
            "ai_insight_category": category if category is not None else row["performance_scenario"],
            "ai_diagnosis": "Forecast masih di bawah target dan risiko pencapaian perlu diperhatikan.",
            "triggered_action_plan": "Fokuskan percepatan eksekusi sell-in pada area prioritas.",
            "priority": priority if priority is not None else row["priority"]}


def test_region_prompt_contains_v2_contract_and_no_v1_momentum():
    prompt = build_prompt(base())
    assert "v_ai_forecast_insight_input_v2" in prompt
    assert "Never use daily-rate or momentum forecasting logic" in prompt


def test_prompt_requires_controlled_category():
    prompt = build_prompt(base())
    assert "ai_insight_category MUST exactly equal performance_scenario" in prompt
    assert "TARGET_ACHIEVED, NEAR_TARGET, AT_RISK, HIGH_RISK, CRITICAL, NO_FORECAST_DATA" in prompt


def test_prompt_requires_indonesian_single_action_and_fact_based_narrative():
    prompt = build_prompt(base())
    assert "Diagnosis and action MUST be Indonesian" in prompt
    assert "Diagnosis must explain the supplied forecast situation" in prompt
    assert "Use at most one management response/action" in prompt
    assert "Narrative fields are QUALITATIVE ONLY" in prompt


def test_prompt_exposes_no_raw_numeric_metric_fields():
    prompt = build_prompt(base())
    for field in (
        "target_sellin", "mtd_actual", "forecast_p10", "forecast_p50", "forecast_p90",
        "achievement_pct_forecast", "forecast_gap_to_target", "forecast_uncertainty_pct",
        "model_spread", "model_spread_pct_p50", "forecast_shortfall",
    ):
        assert field not in prompt


def test_gm_prompt_uses_gm_identity_and_preserves_priority():
    prompt = build_prompt(base("GM"))
    assert "LEVEL: GM" in prompt
    assert '"identity":"GM-COMJAWA"' in prompt
    assert '"priority":"MEDIUM"' in prompt


def test_gm_prompt_uses_entity_code_when_gm_code_column_is_absent():
    row = base("GM")
    del row["gm_code"]
    prompt = build_prompt(row)
    assert "LEVEL: GM" in prompt
    assert "EXPECTED IDENTITY: GM-COMJAWA" in prompt
    assert '"identity":"GM-COMJAWA"' in prompt


def test_ceo_prompt_uses_ceo_identity():
    prompt = build_prompt(base("CEO"))
    assert "LEVEL: CEO" in prompt
    assert '"insight_level": "CEO"' in prompt


def test_validation_preserves_region_identity_category_and_priority():
    row = base("REGION"); insight = valid_insight(row); assert validate(row, insight) == insight


def test_validation_preserves_gm_identity_category_and_priority():
    row = base("GM"); insight = valid_insight(row); assert validate(row, insight) == insight


def test_validation_accepts_gm_identity_from_entity_code_when_gm_code_column_is_absent():
    row = base("GM")
    del row["gm_code"]
    insight = {
        "gm_code": "GM-COMJAWA",
        "ai_insight_category": "AT_RISK",
        "ai_diagnosis": "Forecast masih di bawah target dan risiko pencapaian perlu diperhatikan.",
        "triggered_action_plan": "Fokuskan percepatan eksekusi sell-in pada area prioritas.",
        "priority": "MEDIUM",
    }
    assert validate(row, insight) == insight


def test_validation_preserves_ceo_identity_category_and_priority():
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


def test_validation_rejects_invalid_category():
    row = base()
    with pytest.raises(RuntimeError, match="invalid ai_insight_category"):
        validate(row, valid_insight(row, category="FORECAST_RISK"))


def test_validation_rejects_category_that_does_not_match_source_scenario():
    row = base()
    with pytest.raises(RuntimeError, match="changed ai_insight_category"):
        validate(row, valid_insight(row, category="HIGH_RISK"))


def test_validation_accepts_no_forecast_data_category():
    row = base(); row["performance_scenario"] = "NO_FORECAST_DATA"; row["priority"] = "REVIEW"
    insight = valid_insight(row, priority="REVIEW", category="NO_FORECAST_DATA")
    assert validate(row, insight)["ai_insight_category"] == "NO_FORECAST_DATA"


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


def test_validation_rejects_unsupported_numeric_value():
    row = base()
    insight = valid_insight(row)
    insight["ai_diagnosis"] = "Forecast diperkirakan 99% dari target dan masih berisiko."
    with pytest.raises(RuntimeError, match="unsupported number"):
        validate(row, insight)


def test_validation_rejects_numeric_value_even_when_authoritative():
    row = base()
    insight = valid_insight(row)
    insight["ai_diagnosis"] = "Achievement forecast tercatat 87.7186% dan posisi masih berisiko."
    with pytest.raises(RuntimeError, match="unsupported number"):
        validate(row, insight)


def test_validation_rejects_unsupported_cause():
    row = base()
    insight = valid_insight(row)
    insight["ai_diagnosis"] = "Forecast berada di bawah target karena stok distributor kurang."
    with pytest.raises(RuntimeError, match="unsupported cause"):
        validate(row, insight)


def test_validation_accepts_non_causal_diagnosis():
    row = base()
    insight = valid_insight(row)
    insight["ai_diagnosis"] = "Forecast berada di bawah target dengan ketidakpastian model yang material."
    assert validate(row, insight)["ai_diagnosis"] == insight["ai_diagnosis"]


def test_validation_rejects_multiple_management_actions():
    row = base()
    insight = valid_insight(row)
    insight["triggered_action_plan"] = "Percepat eksekusi sell-in dan kemudian evaluasi distributor."
    with pytest.raises(RuntimeError, match="multiple management actions"):
        validate(row, insight)


def test_validation_accepts_one_management_action_with_compound_objective():
    row = base()
    insight = valid_insight(row)
    insight["triggered_action_plan"] = "Fokuskan percepatan eksekusi sell-in pada area dengan gap terbesar."
    assert validate(row, insight)["triggered_action_plan"] == insight["triggered_action_plan"]


def test_validation_rejects_english_narrative():
    row = base()
    insight = valid_insight(row)
    insight["ai_diagnosis"] = "The forecast is below target and risk remains high."
    with pytest.raises(RuntimeError, match="non-Indonesian"):
        validate(row, insight)


def test_validation_rejects_more_than_two_diagnosis_sentences():
    row = base()
    insight = valid_insight(row)
    insight["ai_diagnosis"] = "Forecast berada di bawah target. Risiko masih material. Fokus pada eksekusi."
    with pytest.raises(RuntimeError, match="too many sentences"):
        validate(row, insight)


def test_validation_rejects_multi_sentence_action():
    row = base()
    insight = valid_insight(row)
    insight["triggered_action_plan"] = "Percepat eksekusi sell-in. Evaluasi kembali hasil harian."
    with pytest.raises(RuntimeError, match="too many sentences"):
        validate(row, insight)
