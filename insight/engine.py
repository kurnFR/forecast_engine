"""Hermes V2 insight engine.

The SQL view is the sole source of deterministic facts. Hermes only writes
narrative fields; validation rejects identity, scenario, priority, or shape
changes. The implementation intentionally reuses the old bi-insight-agent
Hermes subprocess/retry/JSON pattern without its V1 forecasting inputs.
"""
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
REQUIRED = {
    "REGION": ("entity_code", "ai_insight_category", "ai_diagnosis", "triggered_action_plan", "priority"),
    "GM": ("gm_code", "ai_insight_category", "ai_diagnosis", "triggered_action_plan", "priority"),
    "CEO": ("insight_level", "ai_insight_category", "ai_diagnosis", "triggered_action_plan", "priority"),
}


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
- Return ONLY one JSON object, with exactly five fields.

LEVEL: {level}
AUTHORITATIVE INPUT:
{payload}

SUPPORTING CONTEXT (use only for CEO/GM attention prioritization; never recompute company metrics):
{support}

OUTPUT:
{json.dumps({identity: row.get(identity, "CEO"), "ai_insight_category": "CODE", "ai_diagnosis": "DIAGNOSIS", "triggered_action_plan": "ACTION", "priority": row.get("priority")}, ensure_ascii=False)}

FIELD RULES:
- {identity} must exactly equal the supplied identity.
- ai_insight_category is a concise classification, not a replacement for performance_scenario or forecast_scenario.
- priority must exactly equal the supplied priority. Allowed values are LOW, MEDIUM, HIGH, CRITICAL, or REVIEW.
- ai_diagnosis: max 2 short Indonesian sentences; describe supplied forecast status, gap/risk,
  uncertainty/model-spread signal when material, and management implication. Do not invent causes.
- triggered_action_plan: exactly one short Indonesian management action supported by the facts.
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
            insight = validate(row, run_hermes(prompt))
            results.append({"input": row, "insight": insight})
        return results
