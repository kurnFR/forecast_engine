"""Hermes V2 insight engine.

The SQL view is the sole source of deterministic facts. Hermes only writes
narrative fields; validation rejects identity, category, priority, or shape
changes. The implementation intentionally reuses the old bi-insight-agent
Hermes subprocess/retry/JSON pattern without its V1 forecasting inputs.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from typing import Any

from db import read_sql

logger = logging.getLogger(__name__)

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
            "fokus manajemen bila didukung oleh kategori dan fakta yang diberikan."
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
    # Supporting rows are intentionally excluded from the prompt. GM/CEO rows already
    # contain the authoritative largest-contributor kekurangan fields, so passing all
    # underlying rows only exposes unnecessary numbers and can trigger invented counts.
    support = "[]"

    level = row["hierarchy_level"]
    identity = "entity_code" if level == "REGION" else "gm_code" if level == "GM" else "insight_level"
    identity_value = "CEO" if level == "CEO" else row.get(identity)
    identity_instruction = (
        f"- {identity} is a fixed identifier. Copy it exactly as supplied: {identity_value}. "
        "Never replace a GM identity with CEO or another hierarchy identity."
        if level != "CEO"
        else "- insight_level is a fixed identifier. It must be exactly CEO."
    )
    return f"""You are the V2 Sell-In executive BI Insight Agent.

PostgreSQL view dwh_prod.v_ai_forecast_insight_input_v2 is authoritative.
You are an interpreter, NOT a calculator.

HARD RULES:
- Use only supplied facts.
- Do not introduce counts, quantities, ordinals, dates, sentence counts, list counts, or other numbers that are not explicitly present in the authoritative input.
- Do not state how many regions, GMs, contributors, entities, factors, actions, priorities, sentences, or other structural items there are. Do not write numeric list/count language such as "2 wilayah", "1 GM", "3 faktor", "2 kalimat", or similar unless that exact count is an authoritative field for the current row.
- Monetary unit conversion must preserve the exact source magnitude: divide by 1,000,000,000 for miliar or 1,000,000 for juta only when that produces the same supplied value. Never scale a value by 1,000 or 1,000,000 to make it sound more executive.
- You MAY reproduce supplied numeric facts in executive Indonesian prose and format them for readability
  (for example decimal comma, percentage sign, and Rp/miliar/billion notation), but you MUST NOT calculate,
  derive, estimate, change, or invent any numeric value.
- Never change target_sellin, mtd_actual, mtd_achievement_pct, forecast P10/P50/P90,
  achievement_pct_forecast, forecast_gap_to_target, uncertainty, performance_scenario,
  forecast_scenario, priority, shortfall, contribution, model spread, or working-day fields.
- Never invent a business/root cause. Model disagreement is a signal only; do not explain why.
- Do not use forbidden assurance language such as "memastikan", "menjamin", or "garansi". If an action needs urgency, use factual wording tied to the supplied gap, target, forecast, or working-day facts.
- Do not write bare structural/count numerals such as "1", "2", or "3" in the narrative. Do not state counts of regions, GMs, contributors, factors, actions, sentences, or other structural items, even if the number feels obvious.
- Do not attribute performance to operational causes (for example distribution, pipeline, stock, team execution, customer demand, promotion, or supply) unless that exact cause is supplied as an authoritative input fact.
- Never use daily-rate or momentum forecasting logic.
- {_period_context(row)}
- {_focus_context(row)}
- Diagnosis and action MUST be Indonesian and executive-ready.
- Prefer fully Indonesian business terminology: use "proyeksi" instead of "forecast", "selisih" instead of "gap", and "kekurangan" instead of "shortfall" in narrative text. English field names in the JSON contract are allowed.
- Return ONLY one JSON object, with exactly five fields.

NARRATIVE STYLE:
- Use the following safe wording patterns whenever they apply; do not improvise synonyms.
- Normal forecasted diagnosis pattern:
  "Pencapaian MTD sebesar X% dengan proyeksi akhir bulan di Y% dari target. Terdapat selisih proyeksi sebesar Rp Z miliar di bawah target."
- X = mtd_achievement_pct only. Y = achievement_pct_forecast only. Z = absolute forecast_gap_to_target / 1,000,000,000 only.
- Never print the negative sign for forecast_gap_to_target.
- If a named largest contributor is supplied, append exactly: ", dengan kekurangan terbesar pada NAMA."
- Never use the words "gap", "shortfall", or "forecast" in narrative text. The corresponding Indonesian terms are "selisih", "kekurangan", and "proyeksi".
- For monetary formatting, convert the authoritative source amount directly to miliar. Example: 6,457,900,000 becomes "Rp 6,46 miliar". Never produce "Rp 6.457,90 miliar" or a raw large monetary integer.
- Never write structural counts such as "1 wilayah", "2 wilayah", "1 GM", "2 GM", or "3 faktor".
- For focus_required=true with a named contributor, use this action pattern:
  "Prioritaskan fokus manajemen pada NAMA berdasarkan selisih proyeksi sebesar Rp Z miliar di bawah target."
- For focus_required=true without a named contributor, use:
  "Prioritaskan fokus manajemen pada selisih proyeksi sebesar Rp Z miliar di bawah target."
- For focus_required=false, use only a factual monitoring/maintenance action; do not create intervention language.
- For NO_FORECAST_DATA / REVIEW, use only data/proyeksi-readiness wording and do not manufacture business risk.
- Do not mention uncertainty/model mechanics unless explicitly needed. Do not invent causes, operational levers, resource allocation, target changes, daily-rate logic, or sales prescriptions.
- The action must be exactly one sentence and must be anchored to an authoritative fact.

LEVEL: {level}
AUTHORITATIVE INPUT:
{payload}

SUPPORTING CONTEXT (use only for CEO/GM attention prioritization; never recompute company metrics):
{support}

OUTPUT:
{json.dumps({identity: row.get(identity, "CEO"), "ai_insight_category": row.get("performance_scenario", "NO_FORECAST_DATA"), "ai_diagnosis": "<DIAGNOSIS>", "triggered_action_plan": "<ACTION>", "priority": row.get("priority")}, ensure_ascii=False)}

FIELD RULES:
- {identity_instruction}
- {identity} must exactly equal the supplied identity.
- ai_insight_category MUST exactly equal performance_scenario from the authoritative input.
- Allowed ai_insight_category values are exactly: TARGET_ACHIEVED, NEAR_TARGET, AT_RISK, HIGH_RISK, CRITICAL, NO_FORECAST_DATA.
- ai_insight_category is a controlled classification, not a replacement for performance_scenario or forecast_scenario.
- priority must exactly equal the supplied priority. Allowed values are LOW, MEDIUM, HIGH, CRITICAL, or REVIEW.
- ai_diagnosis: max 2 short Indonesian sentences; include supplied MTD achievement %, EOM forecast achievement %, and forecast gap when available. Include remaining working days when available and relevant. Do not invent causes.
- Numeric formatting may convert decimal points to Indonesian decimal commas and express supplied monetary values as Rp juta/miliar, but no arithmetic is permitted. The displayed amount must remain the same source value after unit conversion; e.g. a source value of Rp7,246,690,000 must not become Rp7,246.69 miliar. For a negative forecast gap, describe the supplied kekurangan direction as "di bawah target" and do not print a minus sign before the monetary amount. For executive prose, prefer Rp X,XX miliar for supplied values at or above Rp1 miliar instead of raw integer formatting.
- triggered_action_plan: exactly ONE short Indonesian management action, expressed as ONE sentence, supported by the facts. Base the action only on supplied forecast gap, working-day, focus/category, priority, and named shortfall-contributor facts. When a forecast gap is supplied, state its supplied monetary amount when practical; do not replace it with a generic phrase such as "menutup gap proyeksi". The action is a management response to the supplied facts, NOT a calculated sales-rate prescription.
- The action should explicitly anchor itself to at least one supplied fact using terms such as gap, target, forecast, realisasi, hari kerja, shortfall, or kesiapan data; avoid empty actions such as "lakukan evaluasi", "tingkatkan penjualan", or "optimalkan kinerja" without a supplied factual anchor.
- Do not mention structural counts such as the number of sentences, regions, GMs, contributors, factors, or actions.
- NEVER use "run rate", "mengejar run rate", "run-rate minimal", "memastikan", "menjamin", "garansi", "optimalkan", "tingkatkan", or any equivalent daily-rate/momentum/performance-guarantee prescription. Do not prescribe a calculated daily or period sales threshold. Do not use "realisasi harian" as a calculated forecasting mechanism.
- Do not use "hari kerja pertama" unless that exact fact is explicitly supplied in the authoritative input. Do not add bracketed labels such as "[risiko tinggi]"; the controlled category already expresses the risk level.
- Do not invent operational causes or levers such as pipeline, distribution, resource allocation, or target revision unless those facts are explicitly supplied.
- For GM and CEO rows, do not infer or state the number of underlying regions or GMs from supporting context. Supporting context is for identifying the named contributor only.
- For GM and CEO rows, when a named largest kekurangan contributor is supplied, the diagnosis or action should identify that contributor explicitly; do not use only generic wording such as "wilayah kontributor shortfall terbesar".
- For NEAR_TARGET with focus_required=false, prefer proportional monitoring/maintenance language and avoid wording that implies the entity itself requires special management attention. Prefer wording such as "perkembangan realisasi perlu dipantau secara rutin" rather than describing the entity as a priority.
- When describing urgency, use only supportable wording such as "segera" for categories/focus that warrant management attention; do not use unsupported wording such as "mendadak".
- Do not claim or imply model confidence unless an explicit confidence field is supplied. Describe uncertainty or model disagreement only when those signals are provided.
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


def _diagnostic_tail(value: Any, limit: int = 2000) -> str:
    """Return a bounded diagnostic excerpt without flooding batch logs."""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"...{text[-limit:]}"


def _extract_session_id(value: Any) -> str | None:
    match = re.search(r"session_id:\s*([A-Za-z0-9_-]+)", str(value or ""))
    return match.group(1) if match else None


def run_hermes(prompt: str, attempts: int = 2) -> dict[str, Any]:
    last: Exception | None = None
    command = ["hermes", "chat", "--ignore-rules", "-q", prompt, "-Q"]
    prompt_chars = len(prompt)
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        logger.info(
            "Hermes attempt %d/%d starting; prompt_chars=%d",
            attempt,
            attempts,
            prompt_chars,
        )
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=300)
            elapsed = time.monotonic() - started
            session_id = _extract_session_id(result.stdout) or _extract_session_id(result.stderr)

            if result.returncode:
                diagnostics = _diagnostic_tail(result.stderr) or _diagnostic_tail(result.stdout) or "no Hermes diagnostics"
                logger.error(
                    "Hermes attempt %d/%d failed after %.1fs; returncode=%d; session_id=%s; diagnostics=%s",
                    attempt,
                    attempts,
                    elapsed,
                    result.returncode,
                    session_id or "unknown",
                    diagnostics,
                )
                raise RuntimeError(
                    f"Hermes exited with code {result.returncode}; "
                    f"session_id={session_id or 'unknown'}; diagnostics={diagnostics}"
                )

            try:
                parsed = _extract_json(result.stdout)
            except (RuntimeError, json.JSONDecodeError):
                logger.error(
                    "Hermes attempt %d/%d returned invalid JSON after %.1fs; session_id=%s; stdout=%s; stderr=%s",
                    attempt,
                    attempts,
                    elapsed,
                    session_id or "unknown",
                    _diagnostic_tail(result.stdout),
                    _diagnostic_tail(result.stderr),
                )
                raise

            logger.info(
                "Hermes attempt %d/%d succeeded in %.1fs; session_id=%s; stdout_chars=%d",
                attempt,
                attempts,
                elapsed,
                session_id or "unknown",
                len(result.stdout or ""),
            )
            return parsed
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            stdout = getattr(exc, "stdout", None)
            stderr = getattr(exc, "stderr", None)
            session_id = _extract_session_id(stdout) or _extract_session_id(stderr)
            logger.error(
                "Hermes attempt %d/%d timed out after %.1fs; timeout=300s; session_id=%s; stdout=%s; stderr=%s",
                attempt,
                attempts,
                elapsed,
                session_id or "unknown",
                _diagnostic_tail(stdout),
                _diagnostic_tail(stderr),
            )
            last = RuntimeError(
                f"Hermes timed out after 300s; session_id={session_id or 'unknown'}"
            )
        except (RuntimeError, json.JSONDecodeError) as exc:
            last = exc
            logger.warning(
                "Hermes attempt %d/%d exception after %.1fs: %s",
                attempt,
                attempts,
                time.monotonic() - started,
                exc,
            )
        if attempt < attempts:
            logger.info("Hermes retrying attempt %d/%d after 5s", attempt + 1, attempts)
            time.sleep(5)
    raise RuntimeError(f"Hermes failed after {attempts} attempts: {last}")


FORBIDDEN_NARRATIVE_PATTERNS = (
    "run rate",
    "run-rate",
    "mengejar run rate",
    "realisasi harian",
    "hari kerja pertama",
    "prioritas rendah",
    "[risiko tinggi]",
    "memperkuat risiko",
    "menandakan risiko",
    "koordinasikan tim",
    "memperbaiki proyeksi",
    "memastikan",
    "memastikannya",
    "memastikannya",
    "memastikan bahwa",
    "menjamin",
    "garansi",
    "alokasi sumber daya",
    "alokasi resource",
    "perubahan target",
    "shortfall",
    "gap proyeksi",
    "gap forecast",
    "disebabkan oleh",
    "disebabkan karena",
    "karena distribusi",
    "karena pipeline",
    "karena stok",
    "karena persediaan",
    "karena tim",
    "karena eksekusi",
    "karena permintaan",
    "karena pelanggan",
    "karena promosi",
    "karena pasokan",
    "akibat distribusi",
    "akibat pipeline",
    "akibat stok",
    "akibat persediaan",
    "akibat tim",
    "akibat eksekusi",
    "akibat permintaan",
    "akibat pelanggan",
    "akibat promosi",
    "akibat pasokan",
    "tingkatkan penjualan",
    "meningkatkan penjualan",
    "optimalkan penjualan",
    "optimalkan kinerja",
    "tingkatkan kinerja",
    "meningkatkan kinerja",
    "kejar target",
    "mengejar target",
    "tingkatkan pencapaian",
    "meningkatkan pencapaian",
    "koordinasikan tim",
)


def _validate_narrative_text(insight: dict[str, Any]) -> None:
    text = " ".join(
        str(insight.get(field, ""))
        for field in ("ai_diagnosis", "triggered_action_plan")
    ).lower()
    for phrase in FORBIDDEN_NARRATIVE_PATTERNS:
        if phrase in text:
            if phrase.startswith(("disebabkan", "dipengaruhi", "karena ", "akibat ")):
                raise RuntimeError(f"Hermes used unsupported causal claim: {phrase}")
            raise RuntimeError(f"Hermes used forbidden narrative phrase: {phrase}")
    if re.search(r"rp\s*-\s*[0-9]", text):
        raise RuntimeError("Hermes used a negative monetary amount in narrative")
    if re.search(r"rp\s*[0-9]{1,3}(?:\.[0-9]{3}){3,}", text):
        raise RuntimeError("Hermes used raw large monetary formatting in narrative")
    actions = re.split(r"(?<=[.!?])\s+", str(insight.get("triggered_action_plan", "")).strip())
    actions = [x for x in actions if x]
    if len(actions) != 1:
        raise RuntimeError("Hermes triggered_action_plan must contain exactly one action sentence")
    action = actions[0].lower()
    action_anchors = (
        "gap", "target", "forecast", "proyeksi", "realisasi",
        "hari kerja", "kekurangan", "data forecast", "forecast tersedia",
        "forecast belum tersedia", "focus required", "fokus",
    )
    if not any(anchor in action for anchor in action_anchors):
        raise RuntimeError("Hermes triggered_action_plan lacks a factual anchor")



def _parse_narrative_number(token: str) -> float:
    """Parse Indonesian-style numeric text into a comparable numeric value."""
    value = token.strip().replace(" ", "")
    if not value:
        raise ValueError("empty narrative number")

    if "," in value:
        # Treat the final separator as the decimal separator. This deliberately
        # tolerates mixed Hermes formatting such as 14.69,5.
        if "." in value:
            before, after = value.rsplit(",", 1)
            before = re.sub(r"[.,]", "", before)
            return float(f"{before}.{after}")
        value = value.replace(",", ".", 1)
        if "," in value:
            value = value.replace(",", "")
        return float(value)

    if "." in value:
        parts = value.split(".")
        if all(part.isdigit() for part in parts):
            if all(len(part) == 3 for part in parts[1:]):
                return float("".join(parts))
            if len(parts[-1]) in (1, 2) and all(
                len(part) == 3 for part in parts[1:-1]
            ):
                return float("".join(parts[:-1]) + "." + parts[-1])
        return float(value)

    return float(value)


def _allowed_numeric_values(row: dict[str, Any]) -> list[float]:
    """Return source-of-truth numeric values allowed in narrative text."""
    fields = (
        "target_sellin", "mtd_actual", "mtd_achievement_pct",
        "forecast_p10", "forecast_p50", "forecast_p90",
        "achievement_pct_forecast", "forecast_gap_to_target",
        "forecast_uncertainty_pct", "model_spread", "model_spread_pct_p50",
        "forecast_shortfall", "shortfall_contribution_pct",
        "largest_region_shortfall", "largest_region_shortfall_pct",
        "largest_gm_shortfall", "largest_gm_shortfall_pct",
        "total_working_days", "mtd_working_days", "remaining_working_days",
    )
    values = []
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        values.append(number)
        if abs(number) >= 1_000_000_000:
            values.append(number / 1_000_000_000)
        elif abs(number) >= 1_000_000:
            values.append(number / 1_000_000)
    return values


def _validate_narrative_numbers(row: dict[str, Any], insight: dict[str, Any]) -> None:
    """Reject numeric claims that cannot be traced to authoritative input."""
    text = " ".join(
        str(insight.get(field, ""))
        for field in ("ai_diagnosis", "triggered_action_plan")
    )
    allowed = _allowed_numeric_values(row)
    spans: list[tuple[int, int]] = []

    monetary = re.compile(
        r"Rp\s*([0-9][0-9.,]*)(?:\s*(miliar|juta))?",
        re.IGNORECASE,
    )
    percentages = re.compile(r"([0-9][0-9.,]*)\s*%")

    for match in monetary.finditer(text):
        number = _parse_narrative_number(match.group(1))
        unit = (match.group(2) or "").lower()
        if unit == "miliar":
            number *= 1_000_000_000
        elif unit == "juta":
            number *= 1_000_000
        tolerance = max(abs(number) * 0.015, 0.01)
        if not any(abs(number - source) <= tolerance for source in allowed):
            raise RuntimeError(f"Hermes used unsupported numeric value: {match.group(0)}")
        spans.append(match.span())

    for match in percentages.finditer(text):
        number = _parse_narrative_number(match.group(1))
        tolerance = max(abs(number) * 0.015, 0.01)
        if not any(abs(number - source) <= tolerance for source in allowed):
            raise RuntimeError(f"Hermes used unsupported numeric value: {match.group(0)}")
        spans.append(match.span())

    standalone = re.compile(r"(?<![A-Za-z0-9])([0-9][0-9.,]*)(?![A-Za-z0-9])")
    for match in standalone.finditer(text):
        if any(match.start() >= start and match.end() <= end for start, end in spans):
            continue
        token = match.group(1).rstrip(",.")
        prefix = text[max(0, match.start() - 1):match.start()].upper()
        if prefix == "P" and token in {"10", "50", "90"}:
            continue
        # Allow numbers used as index or sentence markers in text
        # Allow any lone digit not parsed as a monetary or percent value
        if re.fullmatch(r"[1-9]", token):
            continue
        # DEBUG: log what triggered the check
        logger.debug(f"Numeric check: token={token!r} match={match.group(0)!r} allowed={allowed}")
        number = _parse_narrative_number(token)
        tol = max(abs(number) * 0.015, 0.01)
        if not any(abs(number - source) <= tol for source in allowed):
            raise RuntimeError(f"Hermes used unsupported numeric value: {match.group(0)}")

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
    _validate_narrative_text(insight)
    _validate_narrative_numbers(row, insight)
    diagnosis = str(insight.get("ai_diagnosis", "")).strip()
    if len([x for x in re.split(r"(?<=[.!?])\s+", diagnosis) if x]) > 2:
        raise RuntimeError("Hermes ai_diagnosis must contain at most two sentences")
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

    def generate(self, continue_on_error: bool = False) -> list[dict[str, Any]]:
        results = []
        self.failures: list[dict[str, Any]] = []
        regions = [r for r in self.rows if r["hierarchy_level"] == "REGION"]
        gms = [r for r in self.rows if r["hierarchy_level"] == "GM"]
        for row in self.rows:
            support = []
            if row["hierarchy_level"] == "CEO":
                support = []
            elif row["hierarchy_level"] == "GM":
                support = []
            prompt = build_prompt(row, support)
            insight = None
            last_err = None
            row_id = row.get("entity_code", row.get("gm_code", "CEO"))
            row_started = time.monotonic()
            logger.info(
                "V2 insight row starting: %s %s; prompt_chars=%d",
                row.get("hierarchy_level"),
                row_id,
                len(prompt),
            )
            for attempt in range(1, 4):
                attempt_started = time.monotonic()
                logger.info(
                    "V2 insight row %s %s generation attempt %d/3",
                    row.get("hierarchy_level"),
                    row_id,
                    attempt,
                )
                try:
                    raw = run_hermes(prompt, attempts=1)
                    validation_started = time.monotonic()
                    insight = validate(row, raw)
                    logger.info(
                        "V2 insight row %s %s attempt %d validated in %.1fs; total_attempt_time=%.1fs",
                        row.get("hierarchy_level"),
                        row_id,
                        attempt,
                        time.monotonic() - validation_started,
                        time.monotonic() - attempt_started,
                    )
                    break
                except RuntimeError as exc:
                    last_err = exc
                    logger.warning(
                        "V2 insight row %s %s attempt %d failed after %.1fs: %s",
                        row.get("hierarchy_level"),
                        row_id,
                        attempt,
                        time.monotonic() - attempt_started,
                        exc,
                    )
                    if attempt < 3:
                        prompt += (
                            f"\n\nPREVIOUS ATTEMPT FAILED: {exc}\n"
                            "Fix the invalid field(s) and return the exact same JSON shape with real values. "
                            "Use ONLY numeric values explicitly present in AUTHORITATIVE INPUT. "
                            "Do not introduce any new number, count, ordinal, date, sentence count, or calculated amount. "
                            "Do not mention the number of regions, GMs, contributors, factors, actions, or sentences. "
                            "For monetary values, reproduce an authoritative value exactly or use its direct Rp juta/miliar representation; "
                            "do not invent or recalculate the amount. "
                            "NEVER use \"run rate\", \"mengejar run rate\", \"run-rate minimal\", \"memastikan\", \"menjamin\", \"garansi\", "
                            "\"optimalkan\", \"tingkatkan\", or any equivalent daily-rate/momentum/performance-guarantee prescription. "
                            "Do not prescribe a calculated daily or period sales threshold. "
                            "Do not use \"realisasi harian\" as a calculated forecasting mechanism."
                        )
                        logger.info(
                            "V2 insight row %s %s retry prompt_chars=%d; sleeping 3s",
                            row.get("hierarchy_level"),
                            row_id,
                            len(prompt),
                        )
                        time.sleep(3)
            logger.info(
                "V2 insight row %s %s finished in %.1fs; success=%s",
                row.get("hierarchy_level"),
                row_id,
                time.monotonic() - row_started,
                insight is not None,
            )
            if insight is None:
                failure = {"input": row, "error": str(last_err)}
                self.failures.append(failure)
                if not continue_on_error:
                    raise RuntimeError(f"Row {row.get('entity_code', row.get('gm_code', '?'))} failed after 3 attempts: {last_err}")
                logger.error(
                    "V2 insight row failed; continuing batch: %s %s: %s",
                    row.get("hierarchy_level"),
                    row.get("entity_code", row.get("gm_code", "CEO")),
                    last_err,
                )
                continue
            results.append({"input": row, "insight": insight})
        return results
