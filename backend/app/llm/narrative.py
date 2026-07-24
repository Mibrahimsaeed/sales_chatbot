"""
Narrative response layer (P7 of the NLU rework; Part 11 redesigned this
into an EVIDENCE-AWARE explanation layer). Three stages with a hard
safety boundary between the first two and the LLM:

1. compute_facts(ir, rows) — DETERMINISTIC. Every number that can appear
   in the reply is computed here in Python from the SQL result rows:
   top/bottom entries, count, average, spread, per-subject deltas.

2. build_explanation(ir, rows) — DETERMINISTIC, zero LLM involved. This is
   what makes a reply say WHY an answer is correct instead of just
   restating a value: ranking justification ("3rd of 8 shown"), percentage
   interpretation ("75% of target, 25% short of the goal"), comparison
   explanations ("A leads B by X"), and the relationship between a filter
   and the ranking metric. Every number in the output is read straight
   from `rows` — there is no path from here to a fabricated figure.

3. polish_explanation(explanation, facts) — the LLM's only job is a LIGHT
   copy-edit of the already-correct explanation from stage 2, for flow —
   never composing it from scratch, never adding a claim or number. Any
   failure (no key, timeout, bad output, or introducing a number not
   already present) returns the deterministic explanation UNCHANGED — so
   a disabled/unavailable LLM never costs the explanation itself, only
   its polish. Same fail-soft contract as the parser.

compute_insights() (within-result anomaly detection) and compute_trends()
(period-over-period, from real AdvisorHistory snapshots) are additional,
optional evidence — both pure/deterministic, both capped to stay concise.

This mirrors the compiler's safety property ("the LLM never touches a SQL
string"): here, the LLM never originates or alters a number.
"""

from __future__ import annotations

import re
import statistics

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import get_logger
from app.database.models import AdvisorHistory
from app.llm.llm_client import call_llm_json
from app.llm.metric_ontology import METRICS, is_percentage_metric, metric_label
from app.llm.query_ir import QueryIR

log = get_logger("llm.narrative")

_LEVEL_PLURAL = {"advisor": "advisors", "team": "teams", "company": "companies"}


def _round(value) -> float | None:
    return round(float(value), 2) if value is not None else None


def compute_facts(ir: QueryIR, rows: list[dict]) -> dict:
    """Deterministic per-intent insight computation from compiled rows."""
    metric_key = ir.sort.metric or (ir.metric.key if ir.metric else None)
    metric_label_ = METRICS[metric_key].label if metric_key in METRICS else metric_key

    facts: dict = {
        "intent": ir.intent,
        "metric": metric_label_,
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


# ---------------------------------------------------------------------
# Evidence-aware explanation (Part 11) — deterministic, zero LLM.
# ---------------------------------------------------------------------

def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _filters_clause(ir: QueryIR) -> str:
    """Relates the ranking metric to whatever ELSE the result was filtered
    by — e.g. "revenue" ranking among advisors filtered by attendance_rate
    — using only ir.filters, which the validator already grounded. This is
    the "explain relationships between metrics" requirement applied to the
    one relationship the retrieved data actually carries: sort metric vs
    filter condition."""
    if not ir.filters:
        return ""
    parts = []
    for f in ir.filters:
        label = metric_label(f.field) if f.field in METRICS else f.field
        parts.append(f"{label} {f.operator} {f.value}")
    return " filtered by " + ", ".join(parts)


def explain_subject(
    row: dict,
    metric_key: str | None,
    level: str,
    ir: QueryIR,
    rank: int | None = None,
    total: int | None = None,
    include_rank: bool = True,
) -> str:
    """One deterministic sentence for a single result row: percentage
    interpretation (if the metric is percentage-shaped) and/or ranking
    justification — e.g. 'Ali has achieved 75% of the assigned target,
    ranking 3rd of 8 advisors shown, remaining 25% short of the monthly
    goal.' Every number here is `row["value"]` or arithmetic on it — never
    a new figure."""
    value = row.get("value")
    if value is None:
        return ""
    name = row["name"]
    label = metric_label(metric_key)
    percentage = is_percentage_metric(metric_key)

    if percentage:
        pct = round(value, 1)
        parts = [f"{name} has achieved {pct:g}% of the assigned target"]
    else:
        parts = [f"{name} has {value:,.0f} {label}"]

    if include_rank and rank is not None and total and total > 1:
        plural = _LEVEL_PLURAL.get(level, level + "s")
        parts.append(f"ranking {_ordinal(rank)} of {total} {plural} shown{_filters_clause(ir)}")

    if percentage:
        pct = round(value, 1)
        if pct < 100:
            parts.append(f"remaining {100 - pct:g}% short of the monthly goal")
        elif pct > 100:
            parts.append(f"exceeding the monthly goal by {pct - 100:g}%")
        else:
            parts.append("exactly meeting the monthly goal")

    return ", ".join(parts) + "."


def explain_comparison(rows: list[dict], metric_key: str | None, level: str, ir: QueryIR) -> str:
    """A sentence per compared subject (ranking justification + percentage
    interpretation where applicable), plus one closing sentence on the
    gap between the top and bottom subject — the comparison explanation
    requirement. Bounded by construction: `subjects` for a comparison
    intent is always a small, user-named list, never a scan-sized one."""
    values = [r["value"] for r in rows if r.get("value") is not None]
    if len(values) < 2:
        return ""
    total = len(rows)
    sentences = [
        explain_subject(r, metric_key, level, ir, rank=i + 1, total=total)
        for i, r in enumerate(rows)
    ]

    lead = values[0] - values[-1]
    if is_percentage_metric(metric_key):
        # "by 13 percentage points" — "by 13 Target Achievement %" reads as
        # though 13 were itself a percentage of something, which it isn't
        lead_sentence = f"{rows[0]['name']} leads {rows[-1]['name']} by {round(lead, 1):g} percentage points"
    else:
        label = metric_label(metric_key)
        lead_sentence = f"{rows[0]['name']} leads {rows[-1]['name']} by {lead:,.0f} {label}"
        if values[-1]:
            lead_pct = round(lead * 100.0 / abs(values[-1]), 1)
            lead_sentence += f" ({lead_pct:g}% more)"
    sentences.append(lead_sentence + ".")

    return " ".join(s for s in sentences if s)


def build_explanation(ir: QueryIR, rows: list[dict], total_count: int | None = None) -> str:
    """The primary evidence-aware explanation. Comparisons explain every
    subject; leaderboards/filtered_lists explain only the headline (top)
    row — a full paragraph per row would violate "concise" for a
    10-50-row list, and every other row already has its own line in the
    formatted reply. `total_count` (the true match count, if known) makes
    the ranking denominator honest even when the reply itself only shows
    one page — defaults to len(rows) when the caller doesn't have it.
    "" for an empty result or a shape with nothing to explain (e.g. no
    metric at all)."""
    if not rows:
        return ""
    metric_key = ir.sort.metric or (ir.metric.key if ir.metric else None)
    level = ir.subject_level

    if ir.intent == "comparison":
        return explain_comparison(rows, metric_key, level, ir)

    # filtered_list asked for matches, not a ranking (response_planner.py
    # draws the same distinction) — explain the value, skip "ranking Nth"
    include_rank = ir.intent == "leaderboard"
    total = total_count or len(rows)
    return explain_subject(rows[0], metric_key, level, ir, rank=1, total=total, include_rank=include_rank)


_OUTLIER_Z = 1.5


def compute_insights(ir: QueryIR, rows: list[dict]) -> list[str]:
    """Part 8: within-result anomaly/outlier detection, pure Python (no
    LLM — zero hallucination risk, no quota dependency). Flags entries far
    from their peers' mean in the CURRENT result set. This is deliberately
    NOT period-over-period trend detection (see compute_trends() below for
    that); it answers "which of these results stands out and why", not
    "what changed since last time". Every number quoted is read straight
    from `rows`, so it's automatically consistent with compute_facts() —
    no separate evidence guard needed, unlike polish_explanation()'s
    LLM-output check."""
    metric_key = ir.sort.metric or (ir.metric.key if ir.metric else None)
    label = METRICS[metric_key].label if metric_key in METRICS else (metric_key or "value")

    values = [r["value"] for r in rows if r.get("value") is not None]
    if len(values) < 3:
        return []

    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    if not stdev:
        return []

    insights: list[str] = []
    for r in rows:
        value = r.get("value")
        if value is None:
            continue
        z = (value - mean) / stdev
        if abs(z) < _OUTLIER_Z:
            continue
        direction = "above" if z > 0 else "below"
        pct = abs(value - mean) / abs(mean) * 100 if mean else 0
        insights.append(
            f"{r['name']}'s {label} ({value:,.0f}) is {pct:.0f}% {direction} the group average ({mean:,.0f})."
        )
        if len(insights) == 3:
            break

    return insights


# ---------------------------------------------------------------------
# Trend summaries (Part 11) — real period-over-period deltas from
# AdvisorHistory, the append-only daily-snapshot table etl/history_
# snapshot.py already writes on every sync run. This is NOT the "trend"
# QueryIR intent (still correctly rejected by ir_validator.py — that needs
# a user-chosen historical window, not implemented yet); this instead
# attaches an optional trend note to whatever the CURRENT query already
# returned, comparing each row's value against the most recent stored
# snapshot for that same wid. Advisor-level only (AdvisorHistory has no
# team/company rollup) and only for metrics with a direct history column
# — anything else silently yields no trend note rather than guessing.
# ---------------------------------------------------------------------

_TREND_METRIC_TO_HISTORY_FIELD = {
    "mtd_cleared": "mtd_cleared",
    "total_connects": "connects",
    "total_meetings": "meetings",
    "overdue": "overdue",
    "overdue_amount": "overdue",
}

_MAX_TRENDS = 2


def compute_trends(ir: QueryIR, rows: list[dict], db: Session) -> list[str]:
    """Deterministic trend notes against the most recent AdvisorHistory
    snapshot — real historical DB rows, never a fabricated "change". []
    whenever trend data isn't available or applicable (team/company level,
    a metric with no history column, no snapshot yet, or nothing actually
    moved) rather than guessing at a direction."""
    if ir.subject_level != "advisor":
        return []
    metric_key = ir.sort.metric or (ir.metric.key if ir.metric else None)
    history_field = _TREND_METRIC_TO_HISTORY_FIELD.get(metric_key)
    if not history_field:
        return []

    wids = [r["wid"] for r in rows if r.get("wid") is not None]
    if not wids:
        return []

    latest_snapshot_at = db.query(func.max(AdvisorHistory.snapshot_at)).scalar()
    if latest_snapshot_at is None:
        return []

    prior_by_wid = {
        h.wid: getattr(h, history_field)
        for h in db.query(AdvisorHistory)
            .filter(AdvisorHistory.wid.in_(wids), AdvisorHistory.snapshot_at == latest_snapshot_at)
            .all()
    }

    label = metric_label(metric_key)
    trends: list[str] = []
    for r in rows:
        prior = prior_by_wid.get(r.get("wid"))
        current = r.get("value")
        if prior is None or current is None:
            continue
        delta = current - prior
        if not delta:
            continue
        direction = "up" if delta > 0 else "down"
        trends.append(
            f"{r['name']}'s {label} is {direction} {abs(delta):,.0f} since the last sync "
            f"({prior:,.0f} -> {current:,.0f})."
        )
        if len(trends) == _MAX_TRENDS:
            break

    return trends


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


def polish_explanation(explanation: str, facts: dict) -> str:
    """Optional LLM copy-edit of an ALREADY-CORRECT deterministic
    explanation (build_explanation() above) — the LLM's only job is
    smoothing phrasing/flow into 1-3 natural sentences, never composing
    the explanation or any number in it from scratch. The polished text
    is accepted only if every number in it already appeared in the
    explanation or facts — otherwise (or on any failure, or the feature
    flag being off) the deterministic explanation is served unchanged,
    never blanked out."""
    if not explanation or not settings.nlu_narrative:
        return explanation

    prompt = (
        "Lightly copy-edit the following explanation from a sales-operations chatbot so it reads "
        "naturally in 1-3 concise sentences. Do NOT add, remove, invent, recompute, or round any "
        "number or claim in it — phrasing and flow only.\n"
        f"Explanation: {explanation}\n"
        'Return ONLY JSON: {"summary": "<your rewrite>"}'
    )
    raw = call_llm_json(prompt)
    if not raw or not isinstance(raw.get("summary"), str) or not raw["summary"].strip():
        return explanation

    summary = raw["summary"].strip()
    allowed = _numbers_in(facts) | _numbers_in(explanation)
    used = _numbers_in(summary)
    if not used <= allowed:
        log.warning(f"Narrative introduced numbers not in the explanation ({used - allowed}) — serving it unpolished")
        return explanation

    return summary
