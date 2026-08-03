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

Part 10 (confidence-aware generation) makes this module the single place
that decides whether an IR is safe to run at all, not just which slot to
ask about: validate_ir() now also populates ir.ambiguity_reasons (the
human-readable form of `missing[]`) and ir.confidence_level — see
classify_confidence() below for the three-tier gate nlu_pipeline.py uses
to choose between executing immediately, asking a targeted clarifying
question, or rejecting the query outright and asking the user to rephrase.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.config import settings
from app.llm import advisor_resolver, entity_linker, hierarchy
from app.llm.fallback_reasoning import fuzzy_resolve_metric
from app.llm.fuzzy_match import best_match
from app.llm.metric_ontology import METRICS
from app.llm.query_compiler import is_answerable
from app.llm.entity_extractor import (
    get_known_teams, get_known_companies, get_known_offices,
    get_known_unit_heads, get_known_zonal_heads, get_known_bcms,
    get_known_regions,
)
from app.llm.query_ir import ConfidenceLevel, QueryIR, Subject

_MATCH_FLOOR = 0.55
_CONFIDENCE_FLOOR = 0.5     # below this, treat an LLM-supplied field as if it weren't provided
# Every hierarchy group level (team/company/unit_head/zonal_head/business_
# center) is a valid non-metric filter field, plus "advisor" and
# "attendance_status" which aren't part of the hierarchy grouping mapping.
_NON_METRIC_FILTER_FIELDS = set(hierarchy.GROUP_LEVELS) | {"advisor", "attendance_status"}

# Gazetteer fetch function per GROUPABLE level that has one (advisor names
# are grounded at lookup time against the DB view instead — see
# _ground_subject). One dict instead of a growing if/elif chain, per
# requirement to drive level-specific behavior from the hierarchy mapping.
# One fetcher per GROUPABLE level. Keyed by the hierarchy's own level
# names so a rebind there cannot leave grounding pointed at a level that
# no longer exists — the completeness test asserts the two agree.
_SUBJECT_GAZETTEERS = {
    "team": get_known_teams,
    "company": get_known_companies,
    "unit_head": get_known_unit_heads,
    "zonal_head": get_known_zonal_heads,
    "bcm": get_known_bcms,
    "office": get_known_offices,
    "region": get_known_regions,
}
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
    # Phase 3 made this reachable (intent_catalog.TREND_RE +
    # query_planner._score_trend) and therefore user-facing, so the
    # wording is now addressed to the person asking rather than to the
    # roadmap. It stays the single source: nlu_pipeline._UNSUPPORTED_
    # ACTIONS and response_planner._capability_problem both read it.
    "trend": (
        "I can't show trends over time yet — I only hold the current figures "
        "for each period, so there's no earlier value to compare against. I can "
        "give you where things stand right now for any metric, team, or advisor."
    ),
}


@dataclass
class ValidationResult:
    ir: QueryIR
    missing: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.missing

    @property
    def confidence_level(self) -> ConfidenceLevel:
        return classify_confidence(self.ir, self.missing)


def _ground_subject(subject: Subject, db: Session) -> tuple[Subject, str | None]:
    gazetteer_fn = _SUBJECT_GAZETTEERS.get(subject.type)
    if gazetteer_fn is not None:
        match = best_match(
            subject.value, gazetteer_fn(db), kind=hierarchy.match_kind_for(subject.type), floor=_MATCH_FLOOR
        )
    else:
        # Phase 1 identity refactor: an advisor subject is resolved to a
        # WID here rather than left as a bare name for the compiler to
        # match on. A name that maps to exactly one person yields
        # resolved_wid (the compiler then filters Advisor.wid); a name
        # that maps to SEVERAL real people is deliberately left without
        # one, so the compiler falls back to name matching rather than
        # this layer silently picking a person the user never chose.
        # Grounding still never rejects an advisor for non-existence —
        # unchanged from before.
        resolution = advisor_resolver.resolve_by_name(subject.value, db)
        if resolution.is_resolved:
            identity = resolution.identity
            return Subject(
                type=subject.type,
                value=identity.name,
                resolved_id=identity.name,
                resolved_wid=identity.wid,
                match_confidence=subject.match_confidence,
            ), None
        return subject, None

    if not match:
        # RapidFuzz found nothing above floor — a real paraphrase or
        # abbreviation still might resolve semantically (Part 9) before
        # giving up and asking for clarification.
        semantic = entity_linker.semantic_candidates(subject.value, subject.type, db, top_k=1)
        if not semantic:
            return subject, f"subject:{subject.type}:{subject.value}"
        match = semantic[0]["value"], semantic[0]["score"]

    name, score = match
    return Subject(type=subject.type, value=name, resolved_id=name, match_confidence=score), None


def validate_ir(ir: QueryIR, db: Session) -> ValidationResult:
    missing: list[str] = []

    if ir.intent in _UNSUPPORTED_INTENTS:
        missing.append(f"unsupported_intent:{ir.intent}:{_UNSUPPORTED_INTENTS[ir.intent]}")
        ir.missing = missing
        ir.ambiguity_reasons = [_ask_for(item) for item in dict.fromkeys(missing)]
        ir.confidence_level = classify_confidence(ir, missing)
        return ValidationResult(ir=ir, missing=missing)

    # ---- metric (sort/primary) — presence AND confidence floor ----
    if ir.intent in ("leaderboard", "comparison", "filtered_list"):
        metric_key = ir.sort.metric or (ir.metric.key if ir.metric else None)
        metric_confidence = ir.metric.confidence if ir.metric else 1.0

        if metric_key and metric_key not in METRICS:
            # the LLM sometimes invents a close-but-wrong key (e.g.
            # "achievement" instead of "achievement_pct") despite the
            # prompt's explicit "only use catalog keys" instruction — small
            # local models aren't perfectly reliable about this. Recover it
            # the same way a raw user typo/synonym would be recovered
            # (fallback_reasoning.fuzzy_resolve_metric's exact-synonym-
            # substring pass — "achievement" IS literally one of achievement_
            # pct's synonyms) instead of asking the user to repeat
            # themselves for something this unambiguous.
            corrected = fuzzy_resolve_metric(metric_key)
            if corrected:
                metric_key = corrected
                if ir.metric:
                    ir.metric.key = corrected
                ir.sort.metric = corrected

        if not metric_key:
            missing.append("metric")
        elif metric_key not in METRICS:
            missing.append(f"metric:{metric_key}")
        elif metric_confidence < _CONFIDENCE_FLOOR:
            # the field is present but the parser itself wasn't sure —
            # per-field confidence (Part 5.1) means this is treated the
            # same as "missing", not silently trusted.
            missing.append(f"metric_low_confidence:{metric_key}")
        elif not is_answerable(metric_key, ir.subject_level):
            # The level has no resolver for this metric — degrade rather
            # than hard-fail. is_answerable (not a bare `in
            # METRICS[key].bindings` check) so a new hierarchy level
            # answerable only via the compiler's generic rollup fallback
            # isn't incorrectly reset here.
            #
            # Phase 2 narrowed this deliberately. It used to run as the
            # validator's own opinion about the right level, which made
            # it a SECOND owner of a decision subject_level.decide()
            # already made — and since it always chose primary_level, it
            # silently undid a correctly chosen entity level on the way
            # to the compiler. It now fires ONLY when the chosen level is
            # genuinely uncomputable, and records that it did, so a
            # degraded level is visible rather than looking like the
            # planner's choice.
            from app.llm import routing

            routing.decide(
                "Level", METRICS[metric_key].primary_level,
                f"DEGRADED from {ir.subject_level!r}: {metric_key} has no resolver "
                f"at that level, so the metric's own level is the nearest answerable one",
            )
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

    # "breakdown" (Part: hierarchy rework phase 2) is about exactly ONE
    # named entity — mirrors comparison's 2+ check above. Not a metric
    # compiler operation (see chat_service._dispatch_breakdown), so no
    # metric/is_answerable check applies here, only that the subject itself
    # is present and grounded.
    if ir.intent == "breakdown" and len(ir.subjects) < 1:
        missing.append("subjects")

    # intent="clarify" is the parser explicitly saying "ask the user" — it
    # must NEVER validate clean and get executed. If nothing more specific
    # was flagged above, ask about the metric (the most common gap).
    if ir.intent == "clarify" and not missing:
        missing.append("metric" if ir.metric is None else f"metric_low_confidence:{ir.metric.key}")
    elif not missing and ir.overall_confidence < settings.confidence_high_threshold:
        # every individual field cleared its own floor, but the parser's
        # own holistic confidence is still mediocre — Part 10 catches this
        # as its own slot rather than letting a shaky-but-technically-
        # complete IR through as if it were fully confident. Rule-based
        # paths (plan_to_ir, a filled clarification slot) set overall_
        # confidence high enough to clear this on purpose — see their own
        # comments — so this only fires for a genuinely hedged LLM parse.
        missing.append("overall_low_confidence")

    ir.missing = missing
    ir.ambiguity_reasons = [_ask_for(item) for item in dict.fromkeys(missing)]
    ir.confidence_level = classify_confidence(ir, missing)
    return ValidationResult(ir=ir, missing=missing)


def classify_confidence(ir: QueryIR, missing: list[str]) -> ConfidenceLevel:
    """Three-tier execution gate (Part 10), built from configurable
    thresholds (settings.confidence_high_threshold / _low_threshold):

    - "high"   — nothing unresolved AND overall_confidence clears the high
                 floor: execute immediately.
    - "medium" — something specific is unresolved (or overall_confidence
                 fell short of "high" despite every field individually
                 passing), but overall_confidence isn't so low that the
                 parse itself is suspect: ask about that one slot.
    - "low"    — overall_confidence is below the low floor: don't trust
                 `missing` enough to ask a targeted question, reject the
                 query outright and ask the user to rephrase instead.

    Never inflates: an IR with nothing in `missing` still isn't "high"
    unless overall_confidence itself clears the high floor (see the
    "overall_low_confidence" slot validate_ir() adds above)."""
    if not missing:
        return "high"
    if ir.overall_confidence < settings.confidence_low_threshold:
        return "low"
    return "medium"


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
    if item.startswith("subject:"):
        parts = item.split(":", 2)
        if len(parts) == 3:
            _, s_type, s_value = parts
            return f"which {hierarchy.label_for(s_type).lower()} you meant by '{s_value}'"
    if item == "subjects":
        return "which two (or more) things you'd like to compare"
    if item.startswith("unsupported_intent:"):
        _, intent, reason = item.split(":", 2)
        return f"I can't answer a '{intent}'-style question yet ({reason})"
    if item == "overall_low_confidence":
        return "I'm not fully confident I understood that correctly — could you rephrase or add more detail"
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
        return "Just need a bit more detail to answer that — could you fill me in?"
    return "Quick question — " + _ask_for(item) + "?"


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
    if len(parts) >= 2 and parts[-2] in _SUBJECT_GAZETTEERS:
        return _SUBJECT_GAZETTEERS[parts[-2]](db)
    return []


# ---------------------------------------------------------------------
# Confidence breakdown (Part 8, extended by Part 10) — per-field confidence
# already exists on QueryIR (metric.confidence, filters[].confidence,
# subjects[].match_confidence, time_range.confidence, intent_confidence,
# overall_confidence); this derives a single {intent, entities, metric,
# time, filters} view from it for display/logging.
#
# intent and time both take min(heuristic, explicit field) rather than the
# explicit field alone: an IR built the OLD way (rule-based plan_to_ir,
# ir_patcher, any hand-built QueryIR, or the hand-written examples in
# ir_examples.py from before this field existed) never sets intent_
# confidence/time_range.confidence, so they default to 1.0 — min() against
# the existing heuristic means those callers see EXACTLY the same numbers
# as before. A real (lower) value from the LLM can only pull the score
# down further, never inflate it above what the heuristic alone would say.
# ---------------------------------------------------------------------

def confidence_breakdown(ir: QueryIR) -> dict:
    intent_confidence = 0.0 if ir.intent == "clarify" else min(ir.intent_confidence, ir.overall_confidence)

    metric_confidence = ir.metric.confidence if ir.metric else 0.0
    if any(m.startswith("metric") for m in ir.missing):
        metric_confidence = 0.0

    subject_scores = [s.match_confidence for s in ir.subjects]
    entities_confidence = sum(subject_scores) / len(subject_scores) if subject_scores else 1.0

    filter_scores = [f.confidence for f in ir.filters]
    filters_confidence = sum(filter_scores) / len(filter_scores) if filter_scores else 1.0

    # The heuristic: an explicitly non-default period implies the parser
    # had a real signal to act on; the default (MTD) is ambiguous between
    # "the user asked for this month" and "nothing else matched". Blended
    # with the parser's own time_range.confidence per the note above.
    time_heuristic = 0.9 if ir.time_range.period != "MTD" else 0.6
    time_confidence = min(time_heuristic, ir.time_range.confidence)

    return {
        "intent": round(intent_confidence, 2),
        "entities": round(entities_confidence, 2),
        "metric": round(metric_confidence, 2),
        "time": round(time_confidence, 2),
        "filters": round(filters_confidence, 2),
    }
