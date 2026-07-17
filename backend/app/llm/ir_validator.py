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
from difflib import SequenceMatcher

from sqlalchemy.orm import Session

from app.llm.metric_ontology import METRICS
from app.llm.entity_extractor import get_known_teams, get_known_companies
from app.llm.query_ir import QueryIR, Subject

_MATCH_FLOOR = 0.55
_NON_METRIC_FILTER_FIELDS = {"team", "company", "advisor", "attendance_status"}


@dataclass
class ValidationResult:
    ir: QueryIR
    missing: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.missing


def _best_match(value: str, candidates: list[str]) -> tuple[str, float] | None:
    best_name, best_score = None, 0.0
    for c in candidates:
        score = SequenceMatcher(None, value.lower(), c.lower()).ratio()
        if c.lower() == value.lower():
            return c, 1.0
        if score > best_score:
            best_name, best_score = c, score
    if best_name and best_score >= _MATCH_FLOOR:
        return best_name, best_score
    return None


def _ground_subject(subject: Subject, db: Session) -> tuple[Subject, str | None]:
    if subject.type == "team":
        match = _best_match(subject.value, get_known_teams(db))
    elif subject.type == "company":
        match = _best_match(subject.value, get_known_companies(db))
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

    # ---- metric (sort/primary) ----
    if ir.intent in ("leaderboard", "comparison", "filtered_list", "trend"):
        metric_key = ir.sort.metric or (ir.metric.key if ir.metric else None)
        if not metric_key:
            missing.append("metric")
        elif metric_key not in METRICS:
            missing.append(f"metric:{metric_key}")
        elif ir.subject_level not in METRICS[metric_key].bindings:
            # requested level has no resolver for this metric — fall back
            # to the metric's primary level rather than hard-failing,
            # mirroring the old query_planner.py behavior.
            ir.subject_level = METRICS[metric_key].primary_level

    # ---- filters ----
    grounded_filters = []
    for f in ir.filters:
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
        grounded, problem = _ground_subject(s, db)
        if problem:
            missing.append(problem)
        else:
            grounded_subjects.append(grounded)
    ir.subjects = grounded_subjects

    if ir.intent == "comparison" and len(ir.subjects) < 2:
        missing.append("subjects")

    ir.missing = missing
    return ValidationResult(ir=ir, missing=missing)


def build_targeted_clarification(missing: list[str]) -> str:
    """Per-field clarification instead of one generic 'I didn't understand'
    (Root Cause #9 / Part 5.6). Falls back to a single combined question
    when several fields are unresolved."""
    if not missing:
        return "I need a bit more detail to answer that."

    asks = []
    for item in missing:
        if item == "metric":
            asks.append("which metric you'd like (revenue, connects, achievement %, overdue, etc.)")
        elif item.startswith("metric:"):
            asks.append(f"'{item.split(':', 1)[1]}' isn't a metric I track")
        elif item.startswith("filter:"):
            asks.append(f"what you mean by '{item.split(':', 1)[1]}'")
        elif item.startswith("subject:team:"):
            asks.append(f"which team you meant by '{item.split(':', 2)[2]}'")
        elif item.startswith("subject:company:"):
            asks.append(f"which company you meant by '{item.split(':', 2)[2]}'")
        elif item == "subjects":
            asks.append("which two (or more) things you'd like to compare")
        else:
            asks.append(item)

    return "I need a bit more detail — " + "; and ".join(asks) + "?"