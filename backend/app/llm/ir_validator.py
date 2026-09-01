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
# The score at which a pre-LLM gazetteer hit OUTRANKS the level the model
# declared. Deliberately near-exact: this is the difference between "the
# extractor recognised this name" and "the extractor found something
# vaguely like it", and only the first may overrule a parse. A fuzzy hit
# at 0.6 competing with the model's reading would be two guesses, and
# picking the deterministic one is not obviously better.
_AUTHORITATIVE_FLOOR = 0.95
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
# Operations whose answer IS a measure: a ranking, a comparison, one
# group's own figure, or a list reported with one. Keyed on the operation
# rather than on `intent` for the same reason every other consumer is —
# `intent` is the legacy name and, for an operation that declares none
# (population, group_metric), it is not normalised and therefore says
# nothing. Reading it here meant a `group_metric` carrying no measure at
# all could skip this check entirely.
#
# `population` is the deliberate omission: query_compiler documents that
# operation="population" marks the ABSENCE of a measure as intended,
# which is what separates it from an IR that failed to resolve one. A
# hierarchy read is the structural omission — it enumerates people, with
# nothing to rank them by.
# Operations where `subject_level` names the level being ENUMERATED and
# the subject is the SCOPE it is enumerated within — "the teams IN <a
# company>", "the advisors IN <a team>". The subject is not the thing
# being reported there, so copying its level over `subject_level`
# collapses a ranking of the members into the single figure of the
# container: "top teams in <company> by revenue" answered with that
# company's own total instead of its teams.
#
# The other operations report the subject's OWN figure, which is the case
# the normalization exists for. Measured cleanly across both families —
# a scope-shaped question arrives as `leaderboard`, an own-figure one as
# `filtered_list`/`group_metric`.
_SUBJECT_IS_A_SCOPE = ("leaderboard", "population")

_MEASURED_OPERATIONS = ("leaderboard", "comparison", "filtered_list", "group_metric")

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


def _repair(ir: QueryIR, field: str, before, after, why: str) -> None:
    """Record a meaning-changing rewrite, once, to BOTH sinks.

    `ir.repairs` is the structured record — {field, from, to, why} — which
    persists with the IR into ChatLog.resolved_ir, so a production answer
    can be replayed backwards to the model's raw parse months later. The
    routing trace is the human-readable one, in line with every other
    decision the request made.

    Written from here rather than at each site for the reason
    routing.decide() exists: a repair that lands in one sink and not the
    other is how "the model got it wrong" and "we rewrote it afterwards"
    became indistinguishable in the first place.
    """
    from app.llm import routing

    ir.repairs.append({"field": field, "from": before, "to": after, "why": why})
    routing.decide(f"Repair:{field}", f"{before!r} -> {after!r}", why)


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


def _authoritative_levels(entities: dict | None) -> dict[str, tuple[str, float]]:
    """What deterministic extraction already PROVED about each name.

    Entity extraction runs before the model and matches against the live
    gazetteers. Its output reached the model as advisory prompt text
    ("Entities already found by rule-based grounding — use these, don't
    re-derive"), and advice is all it was: for "revenue of AMD year to
    date" extraction grounded AMD as a TEAM at 1.0, the prompt said so,
    and the model typed it `company` anyway. Grounding then failed at the
    declared level, the subject was DROPPED, and the user was asked which
    company they meant by a name that is not one.

    This turns that advice into a constraint the validator can act on.

    ONLY UNAMBIGUOUS NAMES. A value that grounds at more than one level is
    exactly the case the pipeline already asks the user about
    (`clarify_ambiguous`), and forcing a level here would answer a
    question it is entitled to ask — "Haseeb Arslan" is a unit_head, a
    zonal_head, a bcm and an advisor, and which one is meant is not
    extraction's to decide. Those are skipped, so the model keeps its
    reading and the existing clarification still fires.

    `portfolio_lead` / `management_lead` are extraction's older names for
    `zonal_head` / `bcm`, so counting them would make every manager look
    ambiguous with themselves. Only the levels the validator actually
    grounds against are counted.
    """
    if not entities:
        return {}

    by_value: dict[str, set[str]] = {}
    scores: dict[tuple[str, str], float] = {}
    for level in list(_SUBJECT_GAZETTEERS) + ["advisor"]:
        for match in entities.get(f"{level}_matches") or []:
            value, score = match.get("value"), match.get("score", 0.0)
            if not value or score < _AUTHORITATIVE_FLOOR:
                continue
            key = value.strip().lower()
            by_value.setdefault(key, set()).add(level)
            scores[(key, level)] = max(scores.get((key, level), 0.0), score)

    return {
        value: (next(iter(levels)), scores[(value, next(iter(levels)))])
        for value, levels in by_value.items()
        if len(levels) == 1
    }


def _grounds_here(subject: Subject, db: Session) -> float:
    """How well `subject.value` matches at its OWN declared level, 0.0 if
    not at all. Read through the same matcher `_ground_subject` uses, so
    "does the model's reading hold up" is answered exactly once."""
    fetch = _SUBJECT_GAZETTEERS.get(subject.type)
    if fetch is None:
        # An advisor subject: resolution, not a gazetteer scan.
        resolution = advisor_resolver.resolve_by_name(subject.value, db)
        return 1.0 if resolution.is_resolved else 0.0
    match = best_match(subject.value, fetch(db),
                       kind=hierarchy.match_kind_for(subject.type), floor=_MATCH_FLOOR)
    return match[1] if match else 0.0


def _apply_extraction_identity(subject: Subject, db: Session,
                               authoritative: dict, ir: QueryIR) -> Subject:
    """Re-type a subject the extractor identified more reliably.

    THE TEST IS COMPARATIVE, not a blanket override. Extraction's level
    wins only when it is near-exact AND the model's declared level does
    not hold up at least as well — so a name the model typed correctly is
    never touched, and a name that genuinely reads as two different
    entities is left to the clarification that already exists.

    That ordering is what keeps "top advisors in Graana by connects"
    working: Graana grounds as a company AND as an office, so it is not
    in `authoritative` at all and the model's reading stands.
    """
    known = authoritative.get((subject.value or "").strip().lower())
    if known is None:
        return subject
    level, score = known
    if level == subject.type:
        return subject
    if _grounds_here(subject, db) >= score:
        # The model's reading is at least as good a match as the
        # extractor's. Two defensible readings, and the parse is the one
        # that saw the whole sentence.
        return subject

    _repair(ir, "subjects[].type", subject.type, level,
            f"deterministic extraction matched {subject.value!r} at {level!r} "
            f"with confidence {score:.2f} against the live gazetteer, and it "
            f"does not match at {subject.type!r} — the level was a guess about "
            "a name the extractor had already identified")
    return subject.model_copy(update={"type": level})


def _retyped_subject(subject: Subject, db: Session, authoritative: dict,
                     ir: QueryIR) -> Subject:
    """The level this name actually belongs to, or the declared one.

    TWO SOURCES, in order, and both are evidence rather than a guess:

      1. What deterministic extraction PROVED about the name, when it is
         near-exact and unambiguous and the declared level does not hold
         up as well (_apply_extraction_identity).
      2. Failing that, what the other gazetteers say — asked only when the
         declared level does not claim the name at all
         (_reground_scope_subject).

    Never invents a level for a name nothing claims: an unknown value
    keeps its declared type and is refused downstream, exactly as before.
    """
    subject = _apply_extraction_identity(subject, db, authoritative, ir)
    if _SUBJECT_GAZETTEERS.get(subject.type) is None:
        # An advisor subject is resolved by identity, not by gazetteer
        # scan; _ground_subject owns that and never rejects on existence.
        return subject
    if _grounds_here(subject, db) > 0.0:
        return subject
    retyped = _reground_scope_subject(subject, db)
    if retyped is None:
        return subject
    _repair(ir, "subjects[].type", subject.type, retyped.type,
            f"no {subject.type} called {subject.value!r} exists, and the "
            f"{retyped.type} gazetteer claims it — the level was the parser's "
            "guess, and the value is what the user actually named")
    # The TYPE is corrected here; the VALUE is left alone so the grounding
    # loop below still resolves it through the one path that records a
    # match confidence and a resolved_id.
    return subject.model_copy(update={"type": retyped.type})


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


def _reground_scope_subject(subject: Subject, db: Session) -> Subject | None:
    """Find the level a subject actually belongs to, when its declared one
    does not claim it.

    A parser can attach a value to the wrong level in two ways. A
    hierarchy read names two levels at once — the role and the scope it
    sits in ("the Unit Head in <team>") — and the scope's VALUE can land
    on the role's level. An ordinary query needs no such excuse: the model
    simply guesses, and "revenue of AMD" arrives with AMD typed
    `company`. Both end the same way — grounding fails at the declared
    level and the user is asked which company they meant by a team name
    they never claimed was one.

    So when a subject does not ground where it was declared, ask the other
    gazetteers what the name IS. This is the same question entity
    extraction already answers for free text, asked here for a value the
    parser typed optimistically.

    NO LONGER HIERARCHY-ONLY. It was gated on `is_hierarchy_read()`, which
    described where the defect had been OBSERVED rather than where it can
    occur — the guess is a property of the parser, not of the query shape,
    and outside that gate the subject was simply dropped.

    Still only reached AFTER the declared level has failed, which is what
    keeps it safe: a subject the model typed correctly never gets here, so
    a legitimate reading cannot be overridden by a fuzzy match at some
    other level.

    Returns a re-typed Subject, or None when no level claims the name.
    """
    for level, fetch in _SUBJECT_GAZETTEERS.items():
        if level == subject.type:
            continue
        match = best_match(subject.value, fetch(db),
                           kind=hierarchy.match_kind_for(level), floor=_MATCH_FLOOR)
        if match:
            name, score = match
            return Subject(type=level, value=name, resolved_id=name,
                           match_confidence=score)
    return None


def validate_ir(ir: QueryIR, db: Session,
                entities: dict | None = None) -> ValidationResult:
    """`entities` is deterministic extraction's output for the same
    message, when the caller has it.

    OPTIONAL, so the deterministic callers that build their own IRs
    (ir_patcher, the pending-slot fill) and every existing test keep
    working unchanged. It matters for one thing: a subject whose level the
    model guessed, where extraction had already identified the name
    against the live gazetteer. See _authoritative_levels.
    """
    missing: list[str] = []

    if ir.intent in _UNSUPPORTED_INTENTS:
        missing.append(f"unsupported_intent:{ir.intent}:{_UNSUPPORTED_INTENTS[ir.intent]}")
        ir.missing = missing
        ir.ambiguity_reasons = [_ask_for(item) for item in dict.fromkeys(missing)]
        ir.confidence_level = classify_confidence(ir, missing)
        return ValidationResult(ir=ir, missing=missing)

    # ---- ONE AUTHORITATIVE FIELD ------------------------------------
    #
    # `operation` decides what this query IS: resolved_operation() is what
    # the compiler, the response planner and chat_service._dispatch all
    # read. `intent` is the legacy second name for the same thing, and
    # nothing made the two agree — so a model could satisfy the schema
    # with a coherent operation and an intent that meant something else,
    # and whichever field a given consumer happened to read decided the
    # answer. That is how a correctly-parsed "what is X's <measure>?" —
    # right metric, right subject, right level — was answered with a
    # generic entity summary: `intent` said "breakdown" and the dispatcher
    # believed it.
    #
    # The registry already declares the correspondence, so it is READ here
    # rather than restated. An operation with no `ir_intent` (population,
    # group_metric) has no compatibility name to derive and keeps whatever
    # it arrived with — those two are unreachable from the intent
    # vocabulary by construction, which is precisely why the LLM-facing
    # enum is derived from the operations on offer.
    from app.llm import operations

    declared = operations.OPERATIONS.get(ir.resolved_operation())
    if declared is not None and declared.ir_intent and ir.intent != declared.ir_intent:
        _repair(ir, "intent", ir.intent, declared.ir_intent,
                f"the operation registry declares {declared.name!r} as intent "
                f"{declared.ir_intent!r}; the parse said otherwise, and every "
                "consumer reads resolved_operation()")
        ir.intent = declared.ir_intent
    elif declared is not None and declared.ir_intent:
        ir.intent = declared.ir_intent

    # ---- A LIST WITH NOTHING TO RANK BY *IS* A POPULATION ------------
    #
    # `population` and `filtered_list` are the same answer shape with and
    # without a measure, and the model was the only thing deciding which
    # name a metric-free list got. It decided on WORDING, not on meaning:
    #
    #   "advisors in Blue Area or DownTown"   -> population    (answered)
    #   "all advisors excluding Blue Area"    -> filtered_list (refused)
    #
    # Two queries with identical structure — an entity constraint, no
    # measure anywhere — and opposite outcomes, because `filtered_list`
    # is in _MEASURED_OPERATIONS and the second one therefore failed the
    # metric check. The user was asked "which metric would you like?"
    # about a question that correctly named none, and the boolean filter
    # machinery this shape exists for never ran.
    #
    # Strengthening the prompt is not the fix, though it is worth doing
    # and was: it moved the "or" phrasing and left the "excluding" one,
    # which is the same failure one wording along. A decision the IR's own
    # content already determines must not be left to a sampled label.
    #
    # THE RULE, structural like the subject_level one below: an operation
    # that reports members rather than a figure, naming NO measure in any
    # of the four places one can live, and carrying at least one thing it
    # was constrained by, is a population. Nothing failed to resolve — the
    # question asked WHO.
    #
    # WHAT THIS DELIBERATELY DOES NOT TOUCH. `leaderboard`, `comparison`
    # and `group_metric` all keep the metric requirement, because for
    # those the measure IS the answer: a ranking with nothing to rank by,
    # or a group's "figure" with no figure named, is a genuinely
    # incomplete parse and must still be refused. So must a filtered_list
    # that constrains nothing at all — "show me the advisors" with no
    # filter, no subject and no measure is an empty parse wearing a
    # label, and the clarifying question is the right answer to it.
    #
    # Ordered BEFORE the subject_level block below on purpose:
    # _SUBJECT_IS_A_SCOPE lists `population`, so a re-labelled query must
    # be re-labelled first or its named group is read as the ANSWER's
    # level rather than as the scope — "advisors in Blue Area" would
    # report Blue Area instead of listing the people in it.
    if (ir.resolved_operation() == "filtered_list"
            and not ir.is_hierarchy_read()
            and ir.primary_metric() is None
            and (ir.filter_leaves() or ir.subjects)):
        _repair(ir, "operation", ir.resolved_operation(), "population",
                "the query constrains who is in the list but names no measure "
                "in metric, sort, metrics or any filter — that is a question "
                "about WHO, and attaching a measure would join a fact table "
                "and silently drop everyone with no row in it")
        ir.operation = "population"

    # ---- WHAT LEVEL IS THIS NAME, ACTUALLY? --------------------------
    #
    # Runs as a PRE-PASS, before anything downstream reads a subject's
    # type, because two later steps do: `subject_level` is copied from
    # `subjects[0].type` immediately below, and the metric block then asks
    # whether the measure is answerable AT that level.
    #
    # Correcting the type inside the grounding loop further down — where
    # this started — left `subject_level` holding the type the model
    # guessed while `subjects` held the corrected one. For "revenue of AMD
    # year to date" that meant subject_level="company" with a `team`
    # subject: the compiler filters by team and GROUPS BY company, so the
    # answer is AMD's revenue reported under a company's name. A wrong
    # answer in place of a wrong question is not an improvement.
    ir.subjects = [_retyped_subject(s, db, _authoritative_levels(entities), ir)
                   for s in ir.subjects]

    # ---- THE ANSWER IS ABOUT THE SUBJECT THAT WAS NAMED --------------
    #
    # Same class of defect as the one above, in the other pair of fields.
    # The parser states WHAT the question is about in `subjects[].type`
    # and states WHERE the answer is reported in `subject_level`, and
    # nothing made those agree either — so it emitted
    # `subjects=[<a group>]` with `subject_level="advisor"` and the
    # compiler faithfully answered about one arbitrary member instead of
    # the group that was asked for. The parse was right; only the pairing
    # was wrong, and the prompt alone did not hold it: the violation rate
    # tracked surface phrasing rather than meaning.
    #
    # PURELY STRUCTURAL. It reads no metric, no wording and no entity —
    # only the shape of the IR — which is exactly why it fixes a failure
    # that varied by phrasing.
    #
    # The three exclusions are the cases where the answer is deliberately
    # NOT at the subject's own level, and each is a different question:
    #   group_by      reports one level's figures broken out at another
    #                 ("<group>'s advisors by connects").
    #   target_level  is a hierarchy read: it enumerates a level BENEATH
    #                 the subject, so the subject's own level is the one
    #                 level it must not be.
    #   2+ subjects   is a comparison; the sides carry their own levels
    #                 and may legitimately differ from each other.
    # A subject whose type is not a real level is left alone rather than
    # copied in, so a malformed parse cannot write a bad level here.
    #
    # Ordered BEFORE the metric block on purpose: the degrade there
    # (`is_answerable` -> primary_level) exists to rescue a level the
    # compiler cannot serve, so it must keep the last word over this.
    if (len(ir.subjects) == 1
            and ir.group_by is None
            and ir.target_level is None
            and ir.resolved_operation() not in _SUBJECT_IS_A_SCOPE
            and hierarchy.is_valid_level(ir.subjects[0].type)):
        if ir.subject_level != ir.subjects[0].type:
            _repair(ir, "subject_level", ir.subject_level, ir.subjects[0].type,
                    f"the query names one subject and it is a "
                    f"{ir.subjects[0].type}, so that is the level the answer is "
                    "about; the two fields disagreed and the compiler believes "
                    "subject_level")
        ir.subject_level = ir.subjects[0].type

    # ---- metric (sort/primary) — presence AND confidence floor ----
    # A POPULATION is metric-free BY DEFINITION — "who matches this", with
    # nothing to rank by. The prompt instructs the model to emit
    # `metric: null` for it, so requiring one here rejected the exact
    # output the prompt asked for and answered a valid question with a
    # clarifying one.
    # A HIERARCHY READ enumerates people beneath a subject and has no
    # measure to rank by, exactly as a population does. Requiring one
    # would refuse the shape the IR was just widened to express.
    if (ir.resolved_operation() in _MEASURED_OPERATIONS
            and not ir.is_hierarchy_read()):
        # ONE READING of "which measure is this query about", shared with
        # query_compiler._effective_metric. Reading `sort`/`metric` alone
        # missed the two shapes the parser legitimately produces —
        # several measures in `metrics[]`, and a measure used as a
        # CONDITION ("advisors with connects above 1000") — so a query
        # that named its metric plainly was reported as having none.
        metric_key = ir.primary_metric()
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
                _repair(ir, "metric.key", metric_key, corrected,
                        f"{metric_key!r} is not an ontology key; it was recovered "
                        "by the same synonym match a user's typo goes through, "
                        "rather than asking about a measure this unambiguous")
                metric_key = corrected
                # Write back only to fields that were ALREADY set. The
                # correction fixes a spelling; it must not also decide
                # that a filtered list is a ranking. Assigning
                # `sort.metric` on an IR that deliberately had none would
                # do exactly that — response_planner reads the operation,
                # but every other consumer reads the sort.
                if ir.metric:
                    ir.metric.key = corrected
                if ir.sort and ir.sort.metric:
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
            _repair(ir, "subject_level", ir.subject_level,
                    METRICS[metric_key].primary_level,
                    f"DEGRADED: {metric_key} has no resolver at "
                    f"{ir.subject_level!r}, so the metric's own level is the "
                    "nearest answerable one")
            ir.subject_level = METRICS[metric_key].primary_level

    # ---- an unstated sort direction is the MEASURE'S, not "asc" ------
    #
    # `sort.metric` is null exactly when the query expressed no ranking of
    # its own — a filtered list, a population, one group's figure. The
    # direction beside it is then a placeholder the model still had to
    # emit, and gpt-4o-mini emits "asc" for it (4 of 6 sampled parses).
    #
    # The compiler does not treat it as a placeholder. `_run_advisor_
    # rooted` orders by `ir.sort.direction` unconditionally, and
    # `primary_metric()` falls back to a measure named in a FILTER — so
    # "advisors with connects above 1000" ranked the qualifying advisors
    # WORST-FIRST and presented that as the answer.
    #
    # The rule path never had this: plan_to_ir calls `default_direction`,
    # which reads the metric's own `lower_is_better`. This is the same
    # call, applied to the path that was missing it, and only where the
    # query said nothing — an explicit "bottom 5 by X" sets `sort.metric`
    # and keeps its direction untouched.
    if ir.sort is not None and not ir.sort.metric:
        from app.llm.query_compiler import default_direction

        wanted = default_direction(ir.primary_metric())
        if ir.sort.direction != wanted:
            _repair(ir, "sort.direction", ir.sort.direction, wanted,
                    "the query named no ranking of its own (sort.metric is "
                    f"null), so the direction it carried was a placeholder — "
                    f"{ir.primary_metric()!r} orders {wanted} by its own polarity, "
                    "and the compiler applies this value literally")
            ir.sort.direction = wanted

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
        _repair(ir, "filters", f.field, None,
                f"{f.field!r} is neither a metric nor an entity field, so the "
                "condition could not be grounded and was dropped — which can "
                "only WIDEN the result, never narrow it")
    ir.filters = grounded_filters

    # ---- filter tree — validated, never PRUNED ----
    #
    # The flat list above drops a bad filter and carries on, which only
    # ever widens the result. A tree cannot be treated that way: dropping
    # a child of an `or` widens it, and dropping a child of a `not`
    # INVERTS it. So a bad leaf is recorded and the tree is left intact —
    # the compiler refuses to build a disjunction it cannot express
    # rather than quietly answering a different question
    # (query_compiler.UncompilableFilterTree).
    if ir.filter_tree is not None:
        for f in ir.filter_tree.leaves():
            if f.confidence < _CONFIDENCE_FLOOR:
                missing.append(f"filter_low_confidence:{f.field}")
            elif (f.field not in _NON_METRIC_FILTER_FIELDS
                  and f.field not in METRICS):
                missing.append(f"filter:{f.field}")

    # ---- subjects (comparisons / named entities) ----
    grounded_subjects = []
    for s in ir.subjects:
        if s.match_confidence < _CONFIDENCE_FLOOR:
            missing.append(f"subject_low_confidence:{s.type}:{s.value}")
            continue
        # Types were corrected in the pre-pass above, so this grounds at
        # the level the name actually belongs to and only has to resolve
        # the VALUE — the canonical spelling, its match confidence, and
        # (for an advisor) the wid.
        grounded, problem = _ground_subject(s, db)
        if problem:
            missing.append(problem)
            _repair(ir, "subjects", f"{s.type}:{s.value}", None,
                    "no entity of that name and level exists, so the subject was "
                    "removed — the query is refused rather than run unscoped, "
                    "which would answer about everybody")
        else:
            grounded_subjects.append(grounded)
    ir.subjects = grounded_subjects

    # ---- hierarchy read: the levels must be real, and ordered ----
    if ir.is_hierarchy_read():
        for field, value in (("target_level", ir.target_level),
                             ("subject_of", ir.subject_of)):
            if value and not hierarchy.is_valid_level(value):
                missing.append(f"filter:{field}:{value}")
        # The target must sit BELOW the level it is scoped beneath.
        # Inverting them ("the unit heads under an advisor") is not a
        # narrower question, it is an unanswerable one, and the scope
        # filters would silently return nothing.
        if (ir.subject_of and ir.target_level
                and hierarchy.is_chain_level(ir.subject_of)
                and hierarchy.is_chain_level(ir.target_level)
                and hierarchy.depth(ir.target_level) is not None
                and hierarchy.depth(ir.subject_of) is not None
                and hierarchy.depth(ir.target_level) <= hierarchy.depth(ir.subject_of)):
            # ITS OWN SLOT, not a sentence smuggled into the subject
            # slot. This read
            #
            #     f"subject:{ir.subject_of}:{ir.target_level} is not beneath it"
            #
            # and `_ask_for` splits a "subject:" entry on ":" into a LEVEL
            # and a VALUE — so the value became the literal string "team
            # is not beneath it" and the user was asked:
            #
            #     "which zonal head you meant by 'team is not beneath it'?"
            #
            # for the ordinary question "How many people are in ZH1's
            # team?". A structural contradiction in the parse was rendered
            # as a question about a name nobody typed.
            missing.append(f"inverted_hierarchy:{ir.subject_of}:{ir.target_level}")
        if not ir.subjects:
            missing.append("subjects")

    # THE AUTHORITATIVE FIELD, here too. These two checks are structural
    # requirements OF AN OPERATION — a comparison needs two things to
    # compare, a breakdown needs something to break down — so they must
    # read the field that says which operation this is. Keyed on `intent`
    # they fired on operations they do not describe: `population` and
    # `group_metric` declare no ir_intent, so nothing normalises their
    # intent, and a stale "comparison" left there by the parser demanded
    # two subjects of a query that correctly has none.
    if ir.resolved_operation() == "comparison" and len(ir.subjects) < 2:
        missing.append("subjects")

    # "breakdown" (Part: hierarchy rework phase 2) is about exactly ONE
    # named entity — mirrors comparison's 2+ check above. Not a metric
    # compiler operation (see chat_service._dispatch_breakdown), so no
    # metric/is_answerable check applies here, only that the subject itself
    # is present and grounded.
    if ir.resolved_operation() == "breakdown" and len(ir.subjects) < 1:
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
# `inverted_hierarchy:` outranks everything answerable. The other slots
# ask the user to SUPPLY something; this one says the two levels they
# named do not nest, which no answer of theirs can fix — and asking "which
# metric?" about a query whose shape is impossible sends them round a loop
# they cannot exit. It is also raised above "subject" because the same
# parse sets `subjects` as well, and "which two things would you like to
# compare?" is a worse question still.
_CLARIFY_PRIORITY = ("inverted_hierarchy:", "unsupported_intent:", "metric",
                     "subject", "filter")


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
    if item.startswith("inverted_hierarchy:"):
        _, scope, target = item.split(":", 2)
        # Addressed to the person asking, in their words, and WITHOUT the
        # internal level names or the rule that was broken: "target_level
        # must sit below subject_of" is true and useless to them. What
        # they can act on is that the two things they named do not nest
        # in that direction, and which direction does.
        return (
            f"which way round you meant — {hierarchy.label_for(scope)}s sit under "
            f"{hierarchy.label_for(target)}s, not the other way round"
        )
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
