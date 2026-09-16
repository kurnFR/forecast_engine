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
# Common English connective/action terms only. Standard Indonesian BI terms such as
# forecast, target, risk, performance, and uncertainty are intentionally allowed.
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
    return "Gunakan forecast P50/P10/P90 dan achievement forecast persis seperti diberikan."


def build_prompt(row: dict[str, Any], supporting: list[dict[str, Any]] | None = None) -> str:
    payload = json.dumps(_jsonable(row), ensure_ascii=False, separators=(",", ":"), default=str)
    support = json.dumps([_jsonable(x) for x in (supporting or [])], ensure_ascii=False, separators=(",", ":"), default=str)
    level = row["hierarchy_level"]
    identity = "entity_code" if level == "REGION" else "gm_code" if level == "GM" else "insight_level"
    return f"""You are the V2 Sell-In executive BI Insight Agent.

PostgreSQL view dwh_prod.v_ai_forecast_insight_input_v2 is authoritative.
You are an interpreter, NOT a calculator.

HARD RULES:
- Use only supplied facts.
- Never recalculate, modify, round, estimate, or invent numeric values.
- Never change target_sellin, mtd_actual, forecast P10/P50/P90, achievement_pct_forecast,
  forecast_gap_to_target, uncertainty, performance_scenario, forecast_scenario, priority,
  shortfall, contribution, or model spread.
- Never invent a business/root cause. Model disagreement is a signal only; do not explain why.
- Never use daily-rate or momentum forecasting logic.
- {_period_context(row)}
- Diagnosis and action MUST be Indonesian and executive-ready.
- Diagnosis must explain the supplied forecast situation, not merely say to monitor it.
- Use at most one management response/action; do not combine multiple actions.
- Return ONLY one JSON object, with exactly five fields.

LEVEL: {level}
AUTHORITATIVE INPUT:
{payload}

SUPPORTING CONTEXT (use only for CEO/GM attention prioritization; never recompute company metrics):
{support}

OUTPUT:
{json.dumps({identity: row.get(identity, "CEO"), "ai_insight_category": row.get("performance_scenario", "NO_FORECAST_DATA"), "ai_diagnosis": "<DIAGNOSIS>", "triggered_action_plan": "<ONE ACTION>", "priority": row.get("priority")}, ensure_ascii=False)}

FIELD RULES:
- {identity} must exactly equal the supplied identity.
- ai_insight_category MUST exactly equal performance_scenario from the authoritative input.
- priority must exactly equal the supplied priority.
- ai_diagnosis: max 2 short Indonesian sentences; describe supplied forecast status, gap/risk,
  uncertainty/model-spread signal when material, and management implication. Do not invent causes.
- triggered_action_plan: exactly ONE short Indonesian management action supported by the facts.
- Do not state a causal explanation unless the authoritative input explicitly supplies that cause.
- Do not introduce numeric values unless they appear in the authoritative input.
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


def _authoritative_numbers(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key, value in row.items():
        if value is None or key == "periode":
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.add(str(value)); values.add(str(round(float(value), 2)))
            values.add(str(int(value)) if float(value).is_integer() else str(value))
    return values


def _validate_narrative_text(row: dict[str, Any], field: str, value: str) -> None:
    if not value.strip():
        raise RuntimeError(f"Hermes returned invalid field: {field}")
    if ENGLISH_MARKERS.search(value):
        raise RuntimeError(f"Hermes returned non-Indonesian narrative: {field}")
    numbers = re.findall(r"(?<![A-Za-z])\d+(?:[.,]\d+)?%?", value)
    allowed = _authoritative_numbers(row)
    for number in numbers:
        normalized = number.replace(",", ".").rstrip("%")
        if normalized not in allowed and number.rstrip("%") not in allowed:
            raise RuntimeError(f"Hermes introduced unsupported number in {field}: {number}")
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
    expected_identity = row["entity_code"] if level != "CEO" else "CEO"
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
                    prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {exc}\nFix the invalid field(s) and return the exact same JSON shape with real values. No placeholders. No invented causes, numbers, or multiple actions."
                    time.sleep(3)
            if insight is None:
                raise RuntimeError(f"Row {row.get('entity_code', row.get('gm_code', '?'))} failed after 3 attempts: {last_err}")
            results.append({"input": row, "insight": insight})
        return results
