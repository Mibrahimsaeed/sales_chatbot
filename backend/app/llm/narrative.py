"""
Narrative response layer (P7 of the NLU rework). Two stages with a hard
safety boundary between them:

1. compute_facts(ir, rows) — DETERMINISTIC. Every number that can appear
   in the reply is computed here in Python from the SQL result rows:
   top/bottom entries, count, average, spread, per-subject deltas. The
   LLM never sees raw rows, only this facts dict.
2. polish_reply(facts, template_reply) — the LLM's only job is phrasing:
   rewrite the templated reply into 2-3 natural sentences using ONLY the
   facts provided. Any failure (no key, timeout, bad output) returns the
   templated reply unchanged — same fail-soft contract as the parser.

This mirrors the compiler's safety property ("the LLM never touches a
SQL string"): here, the LLM never originates a number.
"""

from __future__ import annotations

import json
import re

from app.core.config import settings
from app.core.logger import get_logger
from app.llm.llm_client import call_llm_json
from app.llm.metric_ontology import METRICS
from app.llm.query_ir import QueryIR

log = get_logger("llm.narrative")


def _round(value) -> float | None:
    return round(float(value), 2) if value is not None else None


def compute_facts(ir: QueryIR, rows: list[dict]) -> dict:
    """Deterministic per-intent insight computation from compiled rows."""
    metric_key = ir.sort.metric or (ir.metric.key if ir.metric else None)
    metric_label = METRICS[metric_key].label if metric_key in METRICS else metric_key

    facts: dict = {
        "intent": ir.intent,
        "metric": metric_label,
        "level": ir.subject_level,
        "result_count": len(rows),
        "filters": [
            {"field": f.field, "operator": f.operator, "value": f.value} for f in ir.filters
        ],
    }
    values = [r["value"] for r in rows if r.get("value") is not None]
    if not values:
        return facts

    facts["top"] = {"name": rows[0]["name"], "value": _round(rows[0]["value"])}
    facts["bottom"] = {"name": rows[-1]["name"], "value": _round(rows[-1]["value"])}
    facts["average"] = _round(sum(values) / len(values))
    facts["spread"] = _round(max(values) - min(values))

    if ir.intent == "comparison":
        facts["subjects"] = [
            {"name": r["name"], "value": _round(r["value"])} for r in rows
        ]
        if len(values) >= 2:
            lead = values[0] - values[-1]
            facts["winner"] = rows[0]["name"]
            facts["lead"] = _round(lead)
            if values[-1]:
                facts["lead_pct"] = _round(lead * 100.0 / abs(values[-1]))

    return facts


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers_in(obj) -> set[str]:
    """Every numeric literal reachable in a facts dict / reply string,
    normalized so 90 and 90.0 compare equal."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            found |= _numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found |= _numbers_in(v)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        found.add(f"{float(obj):g}")
    elif isinstance(obj, str):
        for m in _NUMBER_RE.findall(obj):
            found.add(f"{float(m):g}")
    return found


def polish_reply(facts: dict, template_reply: str) -> str:
    """LLM phrasing polish over deterministic facts. The polished text is
    accepted only if every number in it appears in the facts (or the
    template it was given) — otherwise the templated reply is served."""
    if not settings.nlu_narrative:
        return template_reply

    prompt = (
        "You write one short executive summary line for a sales-operations chatbot reply.\n"
        f"Facts (the ONLY numbers you may use, verbatim): {json.dumps(facts)}\n"
        "Write 2-3 natural sentences summarizing the key insight (who leads, by how much, "
        "what stands out). Do not invent, recompute, or round any number. "
        'Return ONLY JSON: {"summary": "<your sentences>"}'
    )
    raw = call_llm_json(prompt)
    if not raw or not isinstance(raw.get("summary"), str) or not raw["summary"].strip():
        return template_reply

    summary = raw["summary"].strip()
    allowed = _numbers_in(facts)
    used = _numbers_in(summary)
    if not used <= allowed:
        log.warning(f"Narrative introduced numbers not in facts ({used - allowed}) — serving template")
        return template_reply

    return f"{summary}\n\n{template_reply}"
