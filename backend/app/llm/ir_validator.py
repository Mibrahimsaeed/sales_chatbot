"""
IR Validator / Grounder (Part 5.4) — the safety layer between the LLM
semantic parser and the query compiler. Rule-based on purpose: this is
exactly the "bounded, auditable set of real queries" property
sql_generator.py's RESOLVERS registry used to provide, just applied to a
compiler input instead of gating entire query types.

Confirms:
- every metric key (sort metric + filter metrics) exists in the ontology
  AND has a binding for the requested level
- every subject (team/company/advisor name) matches a real gazetteer entry
  above a confidence floor
- every non-metric filter field is one of the known entity fields

Anything that doesn't pass is added to `missing[]` instead of silently
dropped — per-field, not per-message, so the clarification composer can
ask about just the unresolved piece.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.llm.fuzzy_match import best_match
from app.llm.metric_ontology import METRICS
from app.llm.entity_extractor import get_known_teams, get_known_companies
from app.llm.query_ir import QueryIR, Subject

_MATCH_FLOOR = 0.55
_CONFIDENCE_FLOOR = 0.5     # below this, treat an LLM-supplied field as if it weren't provided
_NON_METRIC_FILTER_FIELDS = {"team", "company", "advisor", "attendance_status"}
_UNSUPPORTED_INTENTS = {
    # "lookup" is intentionally handled by the pre-existing rule-based
    # query_planner.py path (nlu_pipeline.py never routes a plain single-
    # advisor lookup into the IR pipeline). If the LLM emits it anyway —
    # e.g. for a compound query that also contains a lookup-shaped clause
    # — the compiler has no lookup-specific query (it would otherwise
    # silently treat it as a one-metric leaderboard, which is wrong:
    # a lookup wants ALL of an advisor's fields, not a ranking by one).
    "lookup": "lookup queries are answered through advisor search, not the metric compiler",
    # "trend" needs the append-only monthly snapshot table described in
    # the redesign's Phase 4 — Performance only stores the CURRENT row
    # per period, so there is no "last month" to diff against yet.
    # Silently running a snapshot query and calling it a trend would be a
    # wrong answer, not a degraded one, so this is a hard reject.
    "trend": "trend/period-over-period comparisons need historical snapshots not yet stored (Phase 4, not implemented)",
}


@dataclass
class ValidationResult:
    ir: QueryIR
    missing: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.missing


def _ground_subject(subject: Subject, db: Session) -> tuple[Subject, str | None]:
    if subject.type == "team":
        match = best_match(subject.value, get_known_teams(db), kind="team", floor=_MATCH_FLOOR)
    elif subject.type == "company":
        match = best_match(subject.value, get_known_companies(db), kind="company", floor=_MATCH_FLOOR)
    else:
        # advisor names are matched at lookup time against the DB view —
        # grounding here only affects filter confidence, not existence.
        return subject, None

    if not match:
        return subject, f"subject:{subject.type}:{subject.value}"

    name, score = match
    return Subject(type=subject.type, value=name, resolved_id=name, match_confidence=score), None


def validate_ir(ir: QueryIR, db: Session) -> ValidationResult:
    missing: list[str] = []

    if ir.intent in _UNSUPPORTED_INTENTS:
        missing.append(f"unsupported_intent:{ir.intent}:{_UNSUPPORTED_INTENTS[ir.intent]}")
        ir.missing = missing
        return ValidationResult(ir=ir, missing=missing)

    # ---- metric (sort/primary) — presence AND confidence floor ----
    if ir.intent in ("leaderboard", "comparison", "filtered_list"):
        metric_key = ir.sort.metric or (ir.metric.key if ir.metric else None)
        metric_confidence = ir.metric.confidence if ir.metric else 1.0
        if not metric_key:
            missing.append("metric")
        elif metric_key not in METRICS:
            missing.append(f"metric:{metric_key}")
        elif metric_confidence < _CONFIDENCE_FLOOR:
            # the field is present but the parser itself wasn't sure —
            # per-field confidence (Part 5.1) means this is treated the
            # same as "missing", not silently trusted.
            missing.append(f"metric_low_confidence:{metric_key}")
        elif ir.subject_level not in METRICS[metric_key].bindings:
            # requested level has no resolver for this metric — fall back
            # to the metric's primary level rather than hard-failing,
            # mirroring the old query_planner.py behavior.
            ir.subject_level = METRICS[metric_key].primary_level

    # ---- filters — presence, validity, AND confidence floor ----
    grounded_filters = []
    for f in ir.filters:
        if f.confidence < _CONFIDENCE_FLOOR:
            missing.append(f"filter_low_confidence:{f.field}")
            continue
        if f.field in _NON_METRIC_FILTER_FIELDS:
            grounded_filters.append(f)
            continue
        if f.field in METRICS:
            grounded_filters.append(f)
            continue
        missing.append(f"filter:{f.field}")
    ir.filters = grounded_filters

    # ---- subjects (comparisons / named entities) ----
    grounded_subjects = []
    for s in ir.subjects:
        if s.match_confidence < _CONFIDENCE_FLOOR:
            missing.append(f"subject_low_confidence:{s.type}:{s.value}")
            continue
        grounded, problem = _ground_subject(s, db)
        if problem:
            missing.append(problem)
        else:
            grounded_subjects.append(grounded)
    ir.subjects = grounded_subjects

    if ir.intent == "comparison" and len(ir.subjects) < 2:
        missing.append("subjects")

    # intent="clarify" is the parser explicitly saying "ask the user" — it
    # must NEVER validate clean and get executed. If nothing more specific
    # was flagged above, ask about the metric (the most common gap).
    if ir.intent == "clarify" and not missing:
        missing.append("metric" if ir.metric is None else f"metric_low_confidence:{ir.metric.key}")

    ir.missing = missing
    return ValidationResult(ir=ir, missing=missing)


# One slot per turn (P6): asking for three things at once gets zero of
# them answered. Highest-priority unresolved slot wins; the rest get
# asked on subsequent turns once this one is filled.
_CLARIFY_PRIORITY = ("unsupported_intent:", "metric", "subject", "filter")


def _ask_for(item: str) -> str:
    if item == "metric":
        return "which metric you'd like (revenue, connects, achievement %, overdue, etc.)"
    if item.startswith("metric:"):
        return f"'{item.split(':', 1)[1]}' isn't a metric I track — which metric did you mean"
    if item.startswith("metric_low_confidence:"):
        return f"you meant '{item.split(':', 1)[1]}' as the metric — I wasn't confident enough to assume that"
    if item.startswith("filter:"):
        return f"what you mean by '{item.split(':', 1)[1]}'"
    if item.startswith("filter_low_confidence:"):
        return f"the '{item.split(':', 1)[1]}' condition — I wasn't confident I understood it correctly"
    if item.startswith("subject_low_confidence:"):
        _, s_type, s_value = item.split(":", 2)
        return f"which {s_type} you meant by '{s_value}' — I wasn't confident enough to assume that"
    if item.startswith("subject:team:"):
        return f"which team you meant by '{item.split(':', 2)[2]}'"
    if item.startswith("subject:company:"):
        return f"which company you meant by '{item.split(':', 2)[2]}'"
    if item == "subjects":
        return "which two (or more) things you'd like to compare"
    if item.startswith("unsupported_intent:"):
        _, intent, reason = item.split(":", 2)
        return f"I can't answer a '{intent}'-style question yet ({reason})"
    return item


def pick_clarification_slot(missing: list[str]) -> str | None:
    """The single highest-priority missing item to ask about this turn."""
    if not missing:
        return None
    for prefix in _CLARIFY_PRIORITY:
        for item in missing:
            if item == prefix or item.startswith(prefix):
                return item
    return missing[0]


def build_targeted_clarification(missing: list[str]) -> str:
    """Per-field clarification instead of one generic 'I didn't understand'
    (Root Cause #9 / Part 5.6). Asks about ONE slot — the highest-priority
    unresolved one — per turn."""
    item = pick_clarification_slot(missing)
    if item is None:
        return "I need a bit more detail to answer that."
    return "I need a bit more detail — " + _ask_for(item) + "?"


def clarification_options(item: str | None, db: Session) -> list[str]:
    """Suggested answers for the slot build_targeted_clarification() just
    asked about (Part 8) — e.g. the actual metric labels when the gap is
    'which metric', or the real team/company gazetteer when the gap is
    which team/company was meant. Empty list when the slot has no
    enumerable option set (e.g. 'subjects', an unsupported intent) — the
    plain question text alone is still a complete answer in that case."""
    if not item:
        return []
    if item == "metric" or item.startswith("metric:") or item.startswith("metric_low_confidence:"):
        return sorted({m.label for m in METRICS.values()})
    parts = item.split(":")
    if len(parts) >= 2 and parts[-2] == "team":
        return get_known_teams(db)
    if len(parts) >= 2 and parts[-2] == "company":
        return get_known_companies(db)
    return []


# ---------------------------------------------------------------------
# Confidence breakdown (Part 8) — per-field confidence already exists on
# QueryIR (metric.confidence, filters[].confidence, subjects[].match_
# confidence, overall_confidence); this derives a single {intent, entities,
# metric, time, filters} view from it instead of asking the LLM for new
# output fields (which can't be iteratively verified while the API key is
# quota-exhausted).
# ---------------------------------------------------------------------

def confidence_breakdown(ir: QueryIR) -> dict:
    intent_confidence = 0.0 if ir.intent == "clarify" else ir.overall_confidence

    metric_confidence = ir.metric.confidence if ir.metric else 0.0
    if any(m.startswith("metric") for m in ir.missing):
        metric_confidence = 0.0

    subject_scores = [s.match_confidence for s in ir.subjects]
    entities_confidence = sum(subject_scores) / len(subject_scores) if subject_scores else 1.0

    filter_scores = [f.confidence for f in ir.filters]
    filters_confidence = sum(filter_scores) / len(filter_scores) if filter_scores else 1.0

    # TimeRange has no native confidence field — this is the one honest
    # approximation here: an explicitly non-default period implies the
    # parser had a real signal to act on; the default (MTD) is ambiguous
    # between "the user asked for this month" and "nothing else matched".
    time_confidence = 0.9 if ir.time_range.period != "MTD" else 0.6

    return {
        "intent": round(intent_confidence, 2),
        "entities": round(entities_confidence, 2),
        "metric": round(metric_confidence, 2),
        "time": round(time_confidence, 2),
        "filters": round(filters_confidence, 2),
    }
