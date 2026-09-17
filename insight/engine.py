"""Hermes V2 insight engine.

The SQL view is the sole source of deterministic facts. Hermes only writes
narrative fields; validation rejects identity, category, priority, unsupported
numbers/causes, non-Indonesian output, or multi-action plans.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from typing import Any

from db import read_sql

VIEW = "dwh_prod.v_ai_forecast_insight_input_v2"
MODEL_NAME = os.getenv("INSIGHT_MODEL_NAME", "hermes-bi-insight")
PROMPT_VERSION = "v2-forecast"
ALLOWED_PRIORITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL", "REVIEW"}
ALLOWED_CATEGORIES = {"TARGET_ACHIEVED", "NEAR_TARGET", "AT_RISK", "HIGH_RISK", "CRITICAL", "NO_FORECAST_DATA"}
REQUIRED = {
    "REGION": ("entity_code", "ai_insight_category", "ai_diagnosis", "triggered_action_plan", "priority"),
    "GM": ("gm_code", "ai_insight_category", "ai_diagnosis", "triggered_action_plan", "priority"),
    "CEO": ("insight_level", "ai_insight_category", "ai_diagnosis", "triggered_action_plan", "priority"),
}

CAUSE_PATTERNS = re.compile(r"\b(?:karena|disebabkan|penyebab(?:nya)?|akibat|dipicu|terkendala|kendala)\b", re.IGNORECASE)
MULTI_ACTION_PATTERNS = re.compile(r"\b(?:dan kemudian|kemudian|selanjutnya|lalu)\b|;|\b(?:serta|dan)\s+(?:pastikan|lakukan|tingkatkan|evaluasi|koordinasikan|percepat|fokuskan)\b", re.IGNORECASE)
ENGLISH_MARKERS = re.compile(r"\b(?:the|actual|action|monitor|focus|ensure|increase|decrease|below|above|because|due|shortfall)\b", re.IGNORECASE)


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    return {k: (None if v is None else str(v) if hasattr(v, "isoformat") else v) for k, v in row.items()}


def load_input(period: str | None = None):
    sql = f"SELECT * FROM {VIEW}"
    params = {}
    if period:
        sql += " WHERE periode = :periode"
        params["periode"] = period
    sql += " ORDER BY CASE hierarchy_level WHEN 'REGION' THEN 1 WHEN 'GM' THEN 2 ELSE 3 END, entity_code"
    return read_sql(sql, params).to_dict("records")


def _period_context(row: dict[str, Any]) -> str:
    if row.get("target_sellin") is None:
        return "Target belum tersedia; jangan membuat atau mengestimasi target."
    if row.get("performance_scenario") == "NO_FORECAST_DATA":
        return "Forecast belum tersedia; jangan membuat atau mengestimasi forecast atau risiko bisnis."
    return "Gunakan status forecast dan sinyal ketidakpastian yang sudah disediakan oleh sistem."


def _qualitative_facts(row: dict[str, Any]) -> list[str]:
    """Build a numeric-free interpretation layer for Hermes."""
    facts: list[str] = []
    scenario = str(row.get("performance_scenario") or "NO_FORECAST_DATA").upper()
    forecast_scenario = str(row.get("forecast_scenario") or "").upper()

    status_map = {
        "TARGET_ACHIEVED": "Posisi forecast berada pada kondisi pencapaian target.",
        "NEAR_TARGET": "Posisi forecast mendekati target.",
        "AT_RISK": "Posisi forecast masih di bawah target dan berada dalam kondisi berisiko.",
        "HIGH_RISK": "Posisi forecast menunjukkan risiko tinggi terhadap pencapaian target.",
        "CRITICAL": "Posisi forecast menunjukkan kondisi kritis terhadap pencapaian target.",
        "NO_FORECAST_DATA": "Forecast belum tersedia untuk penilaian kinerja.",
    }
    facts.append(status_map.get(scenario, "Status forecast mengikuti klasifikasi sistem."))

    if scenario != "NO_FORECAST_DATA":
        scenario_map = {
            "P50_BELOW_TARGET_BUT_UNCERTAIN": "Forecast berada di bawah target dengan ketidakpastian yang material.",
            "HIGH_CONFIDENCE_BELOW_TARGET": "Forecast berada di bawah target dengan keyakinan model yang relatif tinggi.",
            "HIGH_CONFIDENCE_ABOVE_TARGET": "Forecast berada di atas target dengan keyakinan model yang relatif tinggi.",
        }
        matched = scenario_map.get(forecast_scenario)
        if matched:
            facts.append(matched)
        elif "UNCERTAIN" in forecast_scenario:
            facts.append("Forecast memiliki sinyal ketidakpastian yang perlu diperhatikan.")
        elif "BELOW_TARGET" in forecast_scenario:
            facts.append("Forecast berada di bawah target berdasarkan klasifikasi sistem.")
        elif "ABOVE_TARGET" in forecast_scenario:
            facts.append("Forecast berada di atas target berdasarkan klasifikasi sistem.")

        if row.get("model_spread") not in (None, 0, 0.0):
            facts.append("Terdapat sinyal perbedaan antar-model yang relevan untuk perhatian manajemen.")

    priority = str(row.get("priority") or "").upper()
    priority_map = {
        "CRITICAL": "Prioritas penanganan bersifat kritis.",
        "HIGH": "Prioritas penanganan tinggi.",
        "MEDIUM": "Prioritas penanganan memerlukan perhatian manajemen.",
        "LOW": "Prioritas penanganan dapat dipantau secara rutin.",
        "REVIEW": "Prioritas penanganan berfokus pada kesiapan data dan forecast.",
    }
    if priority in priority_map:
        facts.append(priority_map[priority])

    return facts


def _qualitative_support(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return supporting context without exposing numeric metric fields to Hermes."""
    result = []
    for row in rows:
        identity = row.get("entity_code") or row.get("gm_code") or row.get("insight_level")
        result.append({
            "identity": identity,
            "category": row.get("performance_scenario", "NO_FORECAST_DATA"),
            "priority": row.get("priority"),
            "qualitative_facts": _qualitative_facts(row),
        })
    return result


def _expected_identity(row: dict[str, Any]) -> str:
    """Return the authoritative identity used by the V2 input view.

    REGION and GM rows are keyed by entity_code in the view. GM output uses
    the gm_code field, but the source row does not necessarily expose a
    separate gm_code column. CEO is represented by the literal CEO identity.
    """
    level = row["hierarchy_level"]
    if level == "CEO":
        return "CEO"
    return str(row.get("entity_code") or row.get("gm_code") or "").strip()


def build_prompt(row: dict[str, Any], supporting: list[dict[str, Any]] | None = None) -> str:
    level = row["hierarchy_level"]
    identity = "entity_code" if level == "REGION" else "gm_code" if level == "GM" else "insight_level"
    expected_identity = _expected_identity(row)
    qualitative_input = {
        "identity": expected_identity,
        "category": row.get("performance_scenario", "NO_FORECAST_DATA"),
        "priority": row.get("priority"),
        "qualitative_facts": _qualitative_facts(row),
    }
    support = _qualitative_support(supporting or [])
    output_template = {
        identity: expected_identity,
        "ai_insight_category": row.get("performance_scenario", "NO_FORECAST_DATA"),
        "ai_diagnosis": "<DIAGNOSIS>",
        "triggered_action_plan": "<ONE ACTION>",
        "priority": row.get("priority"),
    }
    identity_instruction = (
        f'- The ONLY valid value for {identity} is exactly "{expected_identity}". Copy it character-for-character. '
        f'Do not output "CEO" unless the LEVEL is CEO.'
    )
    return f"""You are the V2 Sell-In executive BI Insight Agent.

PostgreSQL view dwh_prod.v_ai_forecast_insight_input_v2 is authoritative for deterministic facts.
You are an interpreter, NOT a calculator.

HARD RULES:
- Use only supplied qualitative facts and classifications.
- Numeric source facts are intentionally NOT provided to you. Do not ask for them and do not reconstruct them.
- Never calculate, estimate, infer, or invent any numeric value.
- Never invent a business/root cause. Model disagreement is a signal only; do not explain why.
- Never use daily-rate or momentum forecasting logic.
- {_period_context(row)}
- Diagnosis and action MUST be Indonesian and executive-ready.
- Diagnosis must explain the supplied forecast situation, not merely say to monitor it.
- Use at most one management response/action; do not combine multiple actions.
- Narrative fields are QUALITATIVE ONLY. Never copy, calculate, transform, abbreviate, or mention
  any numeric value, percentage, amount, ratio, period number, model value, or numeric token.
- Do not write number-bearing metrics or model labels such as P50, P10, or P90 in narrative fields.
- Use qualitative wording such as "masih di bawah target", "mendekati target",
  "risiko tinggi", "ketidakpastian material", or "memerlukan perhatian manajemen".
- Return ONLY one JSON object, with exactly five fields.

LEVEL: {level}
EXPECTED IDENTITY: {expected_identity}
QUALITATIVE INPUT ONLY:
{json.dumps(qualitative_input, ensure_ascii=False, separators=(",", ":"))}

SUPPORTING CONTEXT (qualitative only; use for CEO/GM attention prioritization):
{json.dumps(support, ensure_ascii=False, separators=(",", ":"))}

OUTPUT CONTRACT:
{json.dumps(output_template, ensure_ascii=False)}

FIELD RULES:
{identity_instruction}
- ai_insight_category MUST exactly equal performance_scenario from the supplied qualitative input.
- Allowed ai_insight_category values: TARGET_ACHIEVED, NEAR_TARGET, AT_RISK, HIGH_RISK, CRITICAL, NO_FORECAST_DATA.
- priority must exactly equal the supplied priority.
- ai_diagnosis: max 2 short Indonesian sentences; describe forecast status, gap/risk,
  uncertainty/model-spread signal when material, and management implication using qualitative wording only.
- triggered_action_plan: exactly ONE short Indonesian management action supported by the qualitative facts.
- Do not state a causal explanation unless the authoritative input explicitly supplies that cause.
- Do not introduce numeric values in diagnosis or action; authoritative numbers remain in SQL/persistence, not narrative.
- If target is missing, say target is not established rather than estimating it.
- If priority is REVIEW because forecast data is unavailable, focus on data/forecast readiness and do not manufacture a business risk.
- No Markdown, no code fence, no extra fields, no commentary.
""".strip()


def _extract_json(output: str) -> dict[str, Any]:
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", output or "")
    start = cleaned.find("{")
    if start < 0:
        raise RuntimeError("Hermes did not return JSON")
    obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(obj, dict):
        raise RuntimeError("Hermes output is not an object")
    return obj


def run_hermes(prompt: str, attempts: int = 2) -> dict[str, Any]:
    last = None
    command = ["hermes", "chat", "--ignore-rules", "-q", prompt, "-Q"]
    for attempt in range(attempts):
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=180)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or "Hermes failed")
            return _extract_json(result.stdout)
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(5)
    raise RuntimeError(f"Hermes failed after {attempts} attempts: {last}")


def _sentence_count(value: str) -> int:
    return len([part for part in re.split(r"[.!?]+(?=\s|$)", value.strip()) if part.strip()])


def _validate_narrative_text(row: dict[str, Any], field: str, value: str) -> None:
    if not value.strip():
        raise RuntimeError(f"Hermes returned invalid field: {field}")
    if ENGLISH_MARKERS.search(value):
        raise RuntimeError(f"Hermes returned non-Indonesian narrative: {field}")
    max_sentences = 2 if field == "ai_diagnosis" else 1
    if _sentence_count(value) > max_sentences:
        raise RuntimeError(f"Hermes returned too many sentences in {field}")
    numbers = re.findall(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)?%?", value)
    if numbers:
        raise RuntimeError(f"Hermes introduced unsupported number in {field}: {numbers[0]}")
    if field == "ai_diagnosis" and CAUSE_PATTERNS.search(value):
        raise RuntimeError("Hermes introduced an unsupported cause in ai_diagnosis")
    if field == "triggered_action_plan" and MULTI_ACTION_PATTERNS.search(value):
        raise RuntimeError("Hermes returned multiple management actions")


def validate(row: dict[str, Any], insight: dict[str, Any]) -> dict[str, Any]:
    level = row["hierarchy_level"]
    fields = REQUIRED[level]
    missing = [f for f in fields if f not in insight]
    extra = [f for f in insight if f not in fields]
    if missing or extra:
        raise RuntimeError(f"Invalid Hermes shape; missing={missing}, extra={extra}")
    identity = fields[0]
    expected_identity = _expected_identity(row)
    if insight[identity] != expected_identity:
        raise RuntimeError(f"Hermes changed {identity}: expected {expected_identity}, got {insight[identity]}")
    category = str(insight["ai_insight_category"]).strip().upper()
    expected_category = str(row.get("performance_scenario") or "NO_FORECAST_DATA").strip().upper()
    if category not in ALLOWED_CATEGORIES:
        raise RuntimeError(f"Hermes returned invalid ai_insight_category: {insight['ai_insight_category']}")
    if category != expected_category:
        raise RuntimeError(f"Hermes changed ai_insight_category for {expected_identity}: expected {expected_category}, got {category}")
    insight["ai_insight_category"] = category
    returned_priority = str(insight["priority"]).strip().upper()
    expected_priority = str(row["priority"]).strip().upper()
    if returned_priority != expected_priority:
        raise RuntimeError(f"Hermes changed priority for {expected_identity}")
    if returned_priority not in ALLOWED_PRIORITIES:
        raise RuntimeError(f"Hermes returned invalid priority: {insight['priority']}")
    insight["priority"] = returned_priority
    placeholders = {"...", "CODE", "DIAGNOSIS", "ACTION", "PLACEHOLDER", "N/A", "UNKNOWN", "TBD"}
    for field in fields:
        value = str(insight.get(field, "")).strip()
        if not value or value.upper() in placeholders or (value.startswith("<") and value.endswith(">")):
            raise RuntimeError(f"Hermes returned invalid field: {field}")
        if field in {"ai_diagnosis", "triggered_action_plan"}:
            _validate_narrative_text(row, field, value)
    return insight


class InsightAgent:
    def __init__(self, period: str | None = None):
        self.rows = load_input(period)
        if not self.rows:
            raise RuntimeError("No rows found in V2 insight input view")
        levels = {r["hierarchy_level"] for r in self.rows}
        if levels != {"REGION", "GM", "CEO"}:
            raise RuntimeError(f"Expected REGION/GM/CEO hierarchy, got {levels}")

    def generate(self) -> list[dict[str, Any]]:
        results = []
        regions = [r for r in self.rows if r["hierarchy_level"] == "REGION"]
        gms = [r for r in self.rows if r["hierarchy_level"] == "GM"]
        for row in self.rows:
            support = []
            if row["hierarchy_level"] == "CEO":
                support = regions + gms
            elif row["hierarchy_level"] == "GM":
                support = [r for r in regions if r.get("entity_code") == row.get("largest_shortfall_regioncode")]
            prompt = build_prompt(row, support)
            insight = None
            last_err = None
            for attempt in range(3):
                try:
                    raw = run_hermes(prompt)
                    insight = validate(row, raw)
                    break
                except RuntimeError as exc:
                    last_err = exc
                    prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {exc}\nFix the invalid field(s). The expected identity is EXACTLY '{_expected_identity(row)}'. Return the exact same JSON shape with real values. Do not output CEO for a REGION or GM row. No placeholders, invented causes, numbers, or multiple actions. Respect sentence limits."
                    time.sleep(3)
            if insight is None:
                raise RuntimeError(f"Row {row.get('entity_code', row.get('gm_code', '?'))} failed after 3 attempts: {last_err}")
            results.append({"input": row, "insight": insight})
        return results
