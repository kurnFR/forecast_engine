"""Hermes V2 insight engine.

The SQL view is the sole source of deterministic facts. Hermes only writes
narrative fields; validation rejects identity, category, priority, or shape
changes. The implementation intentionally reuses the old bi-insight-agent
Hermes subprocess/retry/JSON pattern without its V1 forecasting inputs.
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
ALLOWED_CATEGORIES = {
    "TARGET_ACHIEVED",
    "NEAR_TARGET",
    "AT_RISK",
    "HIGH_RISK",
    "CRITICAL",
    "NO_FORECAST_DATA",
}
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


def _focus_context(row: dict[str, Any]) -> str:
    focus = row.get("focus_required")
    if focus is True:
        return (
            "Focus required = TRUE. Bahasa diagnosis dan aksi boleh menyatakan kebutuhan "
            "focus or intervensi manajemen bila didukung oleh kategori dan fakta yang diberikan."
        )
    if focus is False:
        return (
            "Focus required = FALSE. "
            "jangan menyebut entitas ini sebagai area prioritas/fokus, jangan menyarankan "
            "intervensi atau perhatian khusus terhadap entitas ini, dan gunakan bahasa "
            "pemantauan, pemeliharaan, atau pengawalan eksekusi yang proporsional dengan statusnya."
        )
    return "Focus requirement tidak tersedia; jangan mengarang status fokus manajemen."


def build_prompt(row: dict[str, Any], supporting: list[dict[str, Any]] | None = None) -> str:
    payload = json.dumps(_jsonable(row), ensure_ascii=False, separators=(",", ":"), default=str)
    support = json.dumps([_jsonable(x) for x in (supporting or [])], ensure_ascii=False, separators=(",", ":"), default=str)
    level = row["hierarchy_level"]
    identity = "entity_code" if level == "REGION" else "gm_code" if level == "GM" else "insight_level"
    return f"""ROLE: V2 Sell-In Executive BI Insight Agent.
AUTHORITATIVE SOURCE: PostgreSQL view dwh_prod.v_ai_forecast_insight_input_v2.

═══ PERIOD CONTEXT ═══
{_period_context(row)}

═══ FOCUS CONTEXT ═══
{_focus_context(row)}

═══ HARD CONSTRAINTS (NON-NEGOTIABLE) ═══
1. Use ONLY supplied facts. Zero invention, zero derivation, zero estimation.
2. Numeric formatting allowed: Indonesian decimal commas, % sign, Rp juta/miliar.
   Arithmetic FORBIDDEN: never recalculate, change, or invent any number.
3. Protected fields (copy exactly, never modify):
   target_sellin, mtd_actual, mtd_achievement_pct, forecast P10/P50/P90,
   achievement_pct_forecast, forecast_gap_to_target, uncertainty,
   performance_scenario, forecast_scenario, priority, shortfall,
   contribution, model_spread, working-day fields.
4. No root-cause invention. Model disagreement = signal only, no explanation.
5. No daily-rate or momentum forecasting logic.
6. ai_insight_category MUST equal performance_scenario (exact value, case-sensitive).
7. priority MUST equal supplied priority (exact value, case-sensitive).
8. {identity} MUST equal supplied identity (exact value).
9. Output: ONE JSON object, exactly 5 fields, no Markdown, no commentary.

═══ NARRATIVE STYLE ═══
- Lead with MTD achievement % and EOM forecast achievement %.
- When supplied: state forecast gap (Rp) and remaining working days.
- Structure: "Pencapaian saat ini X% dengan proyeksi akhir bulan Y% dari target. Gap proyeksi Rp Z [implikasi] [sisa hari kerja jika ada]."
- Second sentence: uncertainty/model spread ONLY when material AND supplied.
- Avoid generic phrases when concrete gap/working-day facts exist.

═══ STYLE INTENT ═══
- Softened & realistic: early-period = context, not crisis.
- No judgment/editorializing beyond data.
- No hallucination of causes, facts, values.
- Factual, measured, executive-appropriate.

═══ FIELD SPECIFICATIONS ═══
ai_diagnosis (max 2 kalimat pendek, Bahasa Indonesia):
  • Wajib: MTD achievement %, EOM forecast achievement %
  • Bila ada: forecast gap, sisa hari kerja
  • Early period: sebut "hari kerja pertama" eksplisit
  • LARANGAN: jangan sebut penyebab/root cause

triggered_action_plan (max 1 kalimat pendek, Bahasa Indonesia):
  • Berbasis: forecast gap, working-day, focus/category, priority, named shortfall contributor
  • Pola: "monitor realisasi sell-in harian dan mengejar run rate minimal Rp... sesuai target bulanan"
  • LARANGAN: "memastikan", "menjamin", kata pasti/garansi
  • LARANGAN: inventarisasi levers (pipeline, distribusi, alokasi, revisi target) kecuali eksplisit di data

GM/CEO rows: jika largest_shortfall_regioncode/supplier disediakan → sebut eksplisit, jangan generik.

NEAR_TARGET + focus_required=false: bahasa pemantauan proporsional ("perkembangan realisasi perlu dipantau secara rutin"), bukan prioritas khusus.

Urgency wording: "segera" hanya bila kategori/focus mendukung; jangan "mendadak"/tidak terdukung.

Confidence: jangan klaim kecuali field confidence eksplisit disediakan.

Target missing: "target belum ditetapkan", jangan estimasi.

Priority REVIEW (no forecast): fokus kesiapan data/forecast, jangan buat risiko bisnis.

═══ LEVEL ═══
{level}

═══ AUTHORITATIVE INPUT ═══
{payload}

═══ SUPPORTING CONTEXT ═══
(untuk prioritisasi attention GM/CEO saja; JANGAN hitung ulang metrik company)
{support}

═══ REQUIRED OUTPUT (exact keys, exact values) ═══
{json.dumps({identity: row.get(identity, "CEO"), "ai_insight_category": row.get("performance_scenario", "NO_FORECAST_DATA"), "ai_diagnosis": "<DIAGNOSIS>", "triggered_action_plan": "<ACTION>", "priority": row.get("priority")}, ensure_ascii=False)}
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
                    prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {exc}\nFix the invalid field(s) and return the exact same JSON shape with real values. No placeholders."
                    time.sleep(3)
            if insight is None:
                raise RuntimeError(f"Row {row.get('entity_code', row.get('gm_code', '?'))} failed after 3 attempts: {last_err}")
            results.append({"input": row, "insight": insight})
        return results