import json

import pytest

from insight.engine import _extract_json, build_prompt, validate


def base(level="REGION"):
    row = {
        "periode": "2026-09-01", "hierarchy_level": level,
        "entity_code": "ASWJWA1" if level == "REGION" else "GM-COMJAWA" if level == "GM" else "CEO",
        "entity_name": "ASW JAWA 1" if level == "REGION" else "GM COMMERCIAL JAWA PULAU" if level == "GM" else "CEO",
        "target_sellin": 24700000000, "mtd_actual": 7481451991, "mtd_achievement_pct": 30.29,
        "forecast_p10": 13528200000, "forecast_p50": 21666500000, "forecast_p90": 25917000000,
        "achievement_pct_forecast": 87.7186, "forecast_gap_to_target": -3033500000,
        "forecast_uncertainty_pct": 54.0, "performance_scenario": "AT_RISK",
        "performance_scenario_name": "At Risk", "forecast_scenario": "P50_BELOW_TARGET_BUT_UNCERTAIN",
        "priority": "MEDIUM", "model_spread": 1962200000, "model_spread_pct_p50": 9.06,
        "forecast_shortfall": 3033500000, "shortfall_contribution_pct": 5.58,
        "focus_required": True,
        "total_working_days": 24, "mtd_working_days": 13, "remaining_working_days": 11,
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
            "ai_diagnosis": "Forecast P50 berada di bawah target dan masih berisiko.",
            "triggered_action_plan": "Pantau realisasi sell-in dan fokus pada percepatan eksekusi.",
            "priority": priority if priority is not None else row["priority"]}


def test_region_prompt_contains_v2_contract_and_no_v1_momentum():
    prompt = build_prompt(base())
    assert "v_ai_forecast_insight_input_v2" in prompt
    assert "Never use daily-rate or momentum forecasting logic" in prompt


def test_prompt_requires_controlled_category():
    prompt = build_prompt(base())
    assert "ai_insight_category MUST exactly equal performance_scenario" in prompt
    assert "TARGET_ACHIEVED, NEAR_TARGET, AT_RISK, HIGH_RISK, CRITICAL, NO_FORECAST_DATA" in prompt


def test_prompt_requires_numeric_legacy_style_narrative():
    prompt = build_prompt(base())
    assert "You MAY reproduce supplied numeric facts in executive Indonesian prose" in prompt
    assert "Pencapaian saat ini sebesar X% dengan proyeksi akhir bulan di Y% dari target" in prompt
    assert "Terdapat gap proyeksi sebesar Rp Z" in prompt
    assert "remaining working days" in prompt
    assert "include supplied MTD achievement %, EOM forecast achievement %, and forecast gap" in prompt


def test_prompt_requires_focus_proportional_language():
    prompt = build_prompt(base())
    assert "Focus required = TRUE" in prompt
    assert "focus or intervensi manajemen" in prompt


def test_near_target_without_focus_requires_monitoring_language():
    row = base()
    row["performance_scenario"] = "NEAR_TARGET"
    row["focus_required"] = False
    prompt = build_prompt(row)
    assert "Focus required = FALSE" in prompt
    assert "jangan menyebut entitas ini sebagai area prioritas/fokus" in prompt
    assert "bahasa pemantauan, pemeliharaan, atau pengawalan eksekusi" in prompt
    assert "For NEAR_TARGET with focus_required=false" in prompt
    assert "perkembangan realisasi perlu dipantau secara rutin" in prompt


def test_prompt_forbids_unsupported_urgency_wording_and_model_confidence():
    prompt = build_prompt(base())
    assert "do not use unsupported wording such as \"mendadak\"" in prompt
    assert "Do not claim or imply model confidence unless an explicit confidence field is supplied" in prompt


def test_gm_prompt_uses_gm_identity_and_preserves_priority():
    prompt = build_prompt(base("GM"))
    assert "LEVEL: GM" in prompt
    assert '"gm_code":"GM-COMJAWA"' in prompt
    assert '"priority":"MEDIUM"' in prompt


def test_ceo_prompt_uses_ceo_identity():
    prompt = build_prompt(base("CEO"))
    assert "LEVEL: CEO" in prompt
    assert '"insight_level": "CEO"' in prompt


def test_validation_preserves_region_identity_category_and_priority():
    row = base("REGION"); insight = valid_insight(row); assert validate(row, insight) == insight


def test_validation_preserves_gm_identity_category_and_priority():
    row = base("GM"); insight = valid_insight(row); assert validate(row, insight) == insight


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


def test_review_priority_is_normalized_to_uppercase():
    row = base(); row["priority"] = "review"; insight = valid_insight(row, priority="review")
    assert validate(row, insight)["priority"] == "REVIEW"

def test_validation_rejects_forbidden_lexical_claim():
    row = base()
    insight = valid_insight(row)
    insight["triggered_action_plan"] = "Lakukan pemantauan untuk memastikan penutupan gap."
    with pytest.raises(RuntimeError, match="forbidden narrative phrase"):
        validate(row, insight)


def test_validation_rejects_unsupported_numeric_claim():
    row = base()
    insight = valid_insight(row)
    insight["ai_diagnosis"] = "Pencapaian saat ini 30,29% dengan proyeksi 87,72%."
    insight["triggered_action_plan"] = "Pantau gap sebesar Rp 9,99 miliar."
    with pytest.raises(RuntimeError, match="unsupported numeric value"):
        validate(row, insight)


def test_validation_accepts_supported_numeric_claims():
    row = base()
    insight = valid_insight(row)
    insight["ai_diagnosis"] = (
        "Pencapaian saat ini 30,29% dengan proyeksi akhir bulan 87,72% dari target."
    )
    insight["triggered_action_plan"] = "Pantau gap proyeksi sebesar Rp 3,03 miliar."
    assert validate(row, insight) == insight


def test_validation_rejects_invented_operational_cause():
    row = base()
    insight = valid_insight(row)
    insight["ai_diagnosis"] = "Pencapaian tertinggal karena distribusi belum optimal."
    with pytest.raises(RuntimeError, match="unsupported causal"):
        validate(row, insight)


def test_validation_accepts_fact_without_invented_cause():
    row = base()
    insight = valid_insight(row)
    insight["ai_diagnosis"] = (
        "Pencapaian saat ini 30,29% dengan proyeksi akhir bulan 87,72% dari target."
    )
    insight["triggered_action_plan"] = "Pantau gap proyeksi sebesar Rp 3,03 miliar."
    assert validate(row, insight) == insight


def test_validation_rejects_unanchored_generic_action():
    row = base()
    insight = valid_insight(row)
    insight["triggered_action_plan"] = "Tingkatkan kinerja penjualan."
    with pytest.raises(RuntimeError, match="factual anchor|forbidden narrative phrase"):
        validate(row, insight)


def test_validation_accepts_single_fact_anchored_action():
    row = base()
    insight = valid_insight(row)
    insight["triggered_action_plan"] = "Pantau gap proyeksi sebesar Rp 3,03 miliar selama sisa hari kerja."
    assert validate(row, insight) == insight


def test_prompt_hardens_gm_identity_and_monetary_magnitude():
    row = base("GM")
    prompt = build_prompt(row)
    assert "Copy it exactly as supplied: GM-COMJAWA" in prompt
    assert "Never replace a GM identity with CEO" in prompt
    assert "Never scale a value by 1,000 or 1,000,000" in prompt
    assert "a source value of Rp7,246,690,000 must not become Rp7,246.69 miliar" in prompt
