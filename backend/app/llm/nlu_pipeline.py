"""
Resolution order, cheapest and most deterministic first:

1. Shortcuts — greeting / help / attendance_check. Fixed patterns, zero
   cost, handled by intent_detector.py. Unchanged from the previous
   design.
2. Entity grounding — advisor/team/company gazetteer match (ALL matches,
   not just the first — see entity_extractor.py), thresholds, period,
   limit.
3. Rule-based query planner (query_planner.py) — still handles "lookup"
   (a specific advisor by name), "summary" (team/company overview), and
   "attendance_filter" directly, exactly as before. These action types
   were never the compound-query problem area the redesign targeted, so
   they're intentionally left on the simpler, already-working path.
4. Metric-shaped queries ("leaderboard" action, or anything the rule-based
   planner couldn't resolve) go through semantic_parser.parse(), which:
     - takes the rule-based fast path (skip the LLM) when the query
       doesn't look compound,
     - widens the metric match via fuzzy synonym comparison next,
     - and only then calls the LLM Semantic Parser to produce a full
       QueryIR — capable of expressing multi-filter, multi-subject,
       thresholded, boolean-combined queries the old flat QueryPlan
       could not represent at all (Root Cause #1 in the review).
   The LLM call is genuinely last resort, and optional: if the provider
   is unavailable, rate-limited, or misconfigured, this degrades to a
   schema-grounded clarifying question — never a dead end, never a
   blocking retry loop.
"""

from dataclasses import dataclass

from sqlalchemy.orm import Session

import re

from app.llm import (
    advisor_resolver, conversation_context, conversation_memory,
    cross_turn_resolver, hierarchy, llm_planner, metric_intent, metric_ontology,
    multi_intent, routing, semantic_parser,
)
from app.llm.conversation_memory import MAX_CLARIFY_ATTEMPTS, PendingClarification
from app.llm.entity_extractor import extract_entities, PROVENANCE_KEY
from app.llm.fallback_reasoning import fuzzy_resolve_metric
from app.llm.intent_detector import classify_intent as classify_shortcut
from app.llm.ir_patcher import try_patch
from app.llm.ir_validator import (
    build_targeted_clarification,
    clarification_options,
    pick_clarification_slot,
    validate_ir,
)
from app.llm.preprocessing import normalize
from app.llm.response_formatter import format_person_disambiguation_reply
from app.llm.query_ir import MetricRef, QueryIR, Subject
from app.llm.query_planner import build_query_plan, QueryPlan
from app.llm.metric_ontology import describe_available_metrics
from app.core import audit, tracing
from app.core.logger import get_logger

log = get_logger("llm.nlu_pipeline")

SHORTCUT_INTENTS = ("greeting", "thanks", "help", "attendance_check")
# "breakdown" (unit_head/zonal_head/business_center nested-by-team view) is
# the same simple "bare entity mention" shape as "summary" — see
# query_planner.build_query_plan — so it stays on the same rule-based path.
# "clarify_person" (Phase 1 identity refactor) is rule-based by nature —
# it's the deterministic "this name matches several real people" answer,
# and must never be routed to the LLM, which has no way to know which WID
# was meant either.
# Plan actions that name a capability this system does not have, mapped
# to the registry entry that explains why. DERIVED from
# ir_validator._UNSUPPORTED_INTENTS rather than restated: the same
# limitation reached through the rule planner and through the LLM parser
# must give the same answer, and two copies of the wording is how they
# would stop doing so.
def _unsupported_actions() -> dict[str, str]:
    from app.llm.ir_validator import _UNSUPPORTED_INTENTS

    return {"trend": _UNSUPPORTED_INTENTS["trend"]}


_UNSUPPORTED_ACTIONS = _unsupported_actions()

_RULE_BASED_ACTIONS = (
    "lookup", "summary", "breakdown", "attendance_filter", "clarify_person",
    # M7: "connects of X" is one metric off one advisor row, reached
    # through the ontology binding — deterministic, and there is nothing
    # for the LLM to contribute that the resolved metric key doesn't
    # already say.
    "advisor_metric",
    # "reverse_hierarchy" ("who is X's BM") is a direct column read off one
    # advisor row — there is nothing for the metric compiler or the LLM to
    # contribute, so it stays on the deterministic path.
    "reverse_hierarchy",
    # "ancestry" ("the full hierarchy above X") is the same kind of read,
    # repeated once per level of the chain. Deterministic by construction
    # — the chain decides which levels, and the columns hold the values.
    "ancestry",
    # "roster" ("all advisors in X") is a plain filtered list off one
    # Advisor column — deterministic, nothing for the LLM to add.
    "roster",
    # "direct_reports" ("who reports directly to X") is the same kind of
    # read as roster and reverse_hierarchy: one column match on the
    # advisor rows, with the column chosen by the chain. Deterministic,
    # and nothing the LLM could add that hierarchy.direct_scope_filter
    # does not already decide.
    #
    # An action missing from this tuple does not degrade gracefully — it
    # is routed to the semantic parser instead, and when that is
    # unreachable the whole query answers "I'm not tracking that one",
    # discarding a plan the rule planner had already built correctly.
    "direct_reports",
    # Phase 5B: "comparison" is CONDITIONAL — see _is_rule_based() below.
    # A comparison that names a metric is now an ordinary comparison IR
    # and inherits everything QueryIR owns. A comparison that names NONE
    # stays here, because it answers with a multi-metric KPI table that
    # the single-metric IR cannot express.
    "comparison",
    # "comparison_incomplete" is a CLARIFICATION ("I could only find one
    # side"), not a query — there is nothing for the compiler to run.
    "comparison_incomplete",
)


def _is_rule_based(plan: QueryPlan) -> bool:
    """Does this plan stay on the deterministic plan path?

    Phase 5B split comparison by whether it names a measure:

      metric named  -> the IR path. It inherits _effective_metric (so
                       "compare … year to date" executes YTD rather than
                       resolving it and running MTD), conversation memory
                       (so the next turn keeps both subjects),
                       ir_validator and the response planner — all four
                       of which the plan path bypassed.

      no metric     -> comparison_service, which answers with a table of
                       the default KPI SET. QueryIR carries one metric,
                       so expressing that would mean multi-metric IR:
                       a real extension, not a routing change, and out of
                       scope here. Documented as remaining debt rather
                       than silently dropped — a no-metric comparison is
                       a useful answer, and losing it to reach "one
                       pipeline" would be a regression sold as a
                       refactor.
    """
    if plan.action == "comparison":
        # Phase 13B extends the same rule to a comparison naming SEVERAL
        # measures, for the reason the no-metric case is here: QueryIR
        # carries one metric, and comparison_service carries a tuple. A
        # two-measure comparison sent down the IR path arrives with one
        # of them already gone — which is what "compare X and Y on
        # connects and answered calls" did, answering only on whichever
        # alias string was longer.
        #
        # One measure still takes the IR path and keeps everything that
        # depends on it: _effective_metric, conversation memory,
        # ir_validator and the response planner.
        return plan.metric is None or len(plan.metrics) > 1
    return plan.action in _RULE_BASED_ACTIONS

# Actions complete enough to serve directly even when looks_compound()
# would otherwise send the query to the LLM.
_COMPOUND_EXEMPT_ACTIONS = ("comparison", "comparison_incomplete")

# Plan actions that reach the plan path but ANSWER WITH A QUESTION.
# chat_service._dispatch renders these as kind="clarification", so the
# conversation's topic has not moved and its context must survive — the
# opposite of every other plan action, which answers and therefore
# replaces what the conversation is about.
#
# Listed rather than derived because the distinction lives in
# chat_service's dispatch, which cannot be imported here (it imports this
# module). Kept to the single action that has it, so a divergence is one
# line to see.
_CLARIFYING_ACTIONS = ("comparison_incomplete",)

# Part 8: typed "show more" — the alternative to clicking the button
# (POST /chat/more, see app/api/chat.py). Only recognized when there's an
# active pagination cursor for this session (conversation_memory); with
# no cursor, "more" et al. fall through to the normal pipeline unchanged
# (verified against intent_detector's rules — none of them match these
# phrases, so this can't misfire and steal a real query).
_SHOW_MORE_RE = re.compile(r"^(show more|more|next|next page|load more)$", re.I)

# The level `_pin_stated_level` settled on when it OVERRULED the word in
# the text — see _authoritative_role. Written only in that case, so a
# query whose level word was taken at face value carries nothing here and
# the planner reads the text exactly as it always has. Underscore-
# prefixed because it is meta rather than an entity: consumers that
# serialise the entity dict already skip these keys.
PINNED_LEVEL_KEY = "_pinned_level"


@dataclass
class Resolution:
    kind: str                          # "shortcut" | "plan" | "ir" | "clarify"
    shortcut_intent: str | None = None
    plan: QueryPlan | None = None
    ir: QueryIR | None = None
    entities: dict | None = None
    used_llm_fallback: bool = False
    clarify_message: str | None = None
    clarify_options: list[str] | None = None
    sections: list[tuple[str, "Resolution"]] | None = None  # kind == "multi" only


def _fill_pending_slot(
    pending: PendingClarification, text: str, entities: dict
) -> QueryIR | None:
    """Merge a short answer ("revenue", "Blue Area") into the pending
    partial IR's asked-about slots. Returns the filled copy if anything
    was actually filled, else None (the message didn't answer the
    question)."""
    ir = pending.partial_ir.model_copy(deep=True)
    filled = False

    if any(m == "metric" or m.startswith("metric") for m in pending.missing):
        metric = fuzzy_resolve_metric(text)
        if metric:
            ir.metric = MetricRef(key=metric, confidence=1.0)
            ir.sort.metric = metric
            filled = True

    if any(m.startswith("subject") for m in pending.missing):
        existing = {s.value for s in ir.subjects}
        for team in entities.get("teams", []):
            if team not in existing:
                ir.subjects.append(Subject(type="team", value=team, match_confidence=1.0))
                filled = True
        for company in entities.get("companies", []):
            if company not in existing:
                ir.subjects.append(Subject(type="company", value=company, match_confidence=1.0))
                filled = True

    if not filled:
        return None

    # the gap that made the parser punt to "clarify" is now filled —
    # promote to an executable intent so revalidation can pass
    if ir.intent == "clarify":
        ir.intent = "comparison" if len(ir.subjects) >= 2 else "leaderboard"

    # Part 10: overall_confidence/intent_confidence reflected doubt about
    # WHAT the user wanted before they answered directly — a slot the user
    # just answered in response to our own question is as trustworthy as a
    # rule-based match, so a stale low holistic score from the original
    # ambiguous parse must not keep tripping the "low confidence" gate in
    # ir_validator.classify_confidence() after the ambiguity is resolved.
    ir.overall_confidence = max(ir.overall_confidence, 0.9)
    ir.intent_confidence = max(ir.intent_confidence, 0.9)
    return ir


def _clarify(missing: list[str], db: Session) -> tuple[str, list[str]]:
    """Part 8: the targeted question plus suggested options for it (real
    metric labels / real team or company names), computed from the same
    highest-priority slot build_targeted_clarification() already picks."""
    slot = pick_clarification_slot(missing)
    return build_targeted_clarification(missing), clarification_options(slot, db)


_GIVE_UP_MESSAGE = (
    "I'm having trouble following that one — let's start fresh. Try something like "
    "'top 5 advisors by revenue' or 'compare Blue Area with Downtown on achievement %'."
)


def _handle_pending(
    pending: PendingClarification,
    cleaned: str,
    entities: dict,
    plan: QueryPlan,
    db: Session,
    session_id: str | None,
) -> Resolution | None:
    """Multi-turn clarification (P6): try to read this message as the
    answer to the question we asked last turn. Returns a Resolution to
    serve, or None to fall through to normal processing (the message is
    its own new query, or we've given up on this clarification)."""
    is_short = len(re.findall(r"\S+", cleaned)) <= 4
    filled = _fill_pending_slot(pending, cleaned, entities) if is_short else None

    if filled is not None:
        result = validate_ir(filled, db)
        if result.is_valid:
            conversation_memory.set(session_id, result.ir)  # also closes the pending
            return _ir_resolution(result.ir, entities)
        if pending.attempts >= MAX_CLARIFY_ATTEMPTS or result.confidence_level == "low":
            conversation_memory.clear_pending(session_id)
            return Resolution(kind="clarify", entities=entities, clarify_message=_GIVE_UP_MESSAGE)
        conversation_memory.set_pending(session_id, result.ir, result.missing)
        message, options = _clarify(result.missing, db)
        return Resolution(
            kind="clarify", ir=result.ir, entities=entities,
            clarify_message=message, clarify_options=options,
        )

    # nothing filled: a self-standing query means the user moved on —
    # drop the pending and answer the new question instead
    if plan.action != "unresolved" or semantic_parser.looks_compound(cleaned, entities):
        conversation_memory.clear_pending(session_id)
        return None

    # short but unhelpful answer — re-ask once, then give up gracefully
    if pending.attempts >= MAX_CLARIFY_ATTEMPTS:
        conversation_memory.clear_pending(session_id)
        return Resolution(kind="clarify", entities=entities, clarify_message=_GIVE_UP_MESSAGE)
    conversation_memory.set_pending(session_id, pending.partial_ir, pending.missing)
    message, options = _clarify(pending.missing, db)
    return Resolution(
        kind="clarify", ir=pending.partial_ir, entities=entities,
        clarify_message=message, clarify_options=options,
    )


# A possessive attaches what follows it to whoever precedes it. Counting
# them is how this module tells "one subject, several measures" from
# "several subjects, one measure each" — see _distributes_metrics below.
_POSSESSIVE_RE = re.compile(r"\w's\b", re.I)


def _distributes_metrics(text: str, plan: QueryPlan) -> bool:
    """Does this turn attach DIFFERENT measures to DIFFERENT subjects?

        "Zainab's connects and answered calls"          one subject   -> False
        "Zainab's connects and Awais's answered calls"  two subjects  -> True

    Both name two measures, and only the first can be answered as two
    measures about one person. The second is a (subject, measure) PAIRING,
    which no structure here can carry: QueryPlan holds one subject and a
    flat metric list, so answering it means attaching both measures to
    whichever person resolved — and that is not a partial answer, it is
    the wrong person's number under the right label.

    WHY POSSESSIVES AND NOT THE ENTITY DICT. Because the entity dict
    cannot see it: advisor_resolver.resolve_all_from_text finds both
    people in "Zainab's connects and Awais's connects" and only ONE in
    "Zainab's connects and Awais's answered calls" — the name span runs
    into the measure phrase and fails to match. That is a pre-existing
    extraction limitation, and a safety check that trusted it would be
    blind in exactly the case it exists to catch.

    So the signal is structural and deliberately over-cautious: two
    possessives plus two measures means the measures are distributed,
    and this refuses. It can over-trigger ("Zainab's team's connects and
    answered calls" reads as distributed and gets a clarification rather
    than an answer) — which is the right way round, because the failure
    it prevents is a confident wrong attribution and the failure it
    causes is one extra question.
    """
    if len(plan.metrics) < 2:
        return False
    return len(_POSSESSIVE_RE.findall(text)) >= 2


# A follow-up that carries a pronoun or a bare possessive is asking about
# the person already under discussion ("what about his overdue", "and her
# team"). Deliberately narrow — a message naming its own subject resolves
# that subject normally and never reaches this.
_PERSON_FOLLOWUP_RE = re.compile(
    r"\b(he|him|his|she|her|hers|they|them|their|this person|that person)\b", re.I
)


def _looks_like_person_followup(text: str) -> bool:
    return bool(_PERSON_FOLLOWUP_RE.search(text))


def _match_level(text: str, levels: list[str]) -> str | None:
    """Which offered level this message names, or None.

    Matched against hierarchy.label_for — the same labels the question
    was PHRASED with — so the vocabulary the user is answering in is the
    vocabulary they were shown, and no new synonym list appears here.
    Requires exactly one match: "bcm or advisor" is not a choice.
    """
    lowered = text.strip().lower().strip("?.!")
    hits = [
        level for level in levels
        if lowered == level.lower()
        or lowered == (hierarchy.label_for(level) or "").lower()
    ]
    return hits[0] if len(hits) == 1 else None


def _handle_pending_level(
    pending, cleaned: str, db: Session, session_id: str | None
) -> Resolution | None:
    """Read this message as the answer to "which level did you mean?".

    Returns a Resolution to serve, or None to fall through when the
    message is plainly a new question. Deliberately the same shape as
    _handle_pending_person: match, then RE-RUN the original query with
    the choice applied, rather than making the user retype it.
    """
    chosen = _match_level(cleaned, pending.levels)

    if chosen is not None:
        conversation_memory.clear_pending_level(session_id)
        routing.decide(
            "Clarification", f"level={chosen}",
            f"read {cleaned!r} as the answer to the pending "
            f"{pending.levels} question about {pending.value!r}; re-running "
            f"the original query with that level pinned",
        )
        return resolve(pending.original_text, db, session_id=session_id,
                       _depth=1, _pin=(pending.value, chosen))

    # Not a choice, and long enough to stand alone -> the user moved on.
    if len(re.findall(r"\S+", cleaned)) > 4:
        conversation_memory.clear_pending_level(session_id)
        return None

    if pending.attempts >= MAX_CLARIFY_ATTEMPTS:
        conversation_memory.clear_pending_level(session_id)
        options = ", ".join(hierarchy.label_for(l) or l for l in pending.levels)
        return Resolution(
            kind="clarify", entities={},
            clarify_message=(
                f"I still couldn't tell which {pending.value!r} you meant. "
                f"Try naming it in the question, e.g. "
                f"'{options.split(', ')[0]} {pending.value}'."
            ),
        )

    conversation_memory.set_pending_level(
        session_id, pending.value, pending.levels, pending.original_text)
    options = [hierarchy.label_for(l) or l for l in pending.levels]
    return Resolution(
        kind="clarify", entities={},
        clarify_message=(
            f"'{pending.value}' could mean the {' or the '.join(options)} — "
            "which did you mean?"
        ),
        clarify_options=options,
    )


def _handle_pending_person(
    pending, cleaned: str, db: Session, session_id: str | None
) -> Resolution | None:
    """Phase 5: read this message as the answer to "which <name>?".

    Returns a Resolution to serve, or None to fall through when the
    message is plainly a new question rather than a choice. Picking is
    delegated to advisor_resolver.resolve_choice, which returns None
    unless the answer identifies exactly ONE candidate — re-asking beats
    guessing a second time."""
    chosen = advisor_resolver.resolve_choice(cleaned, pending.candidates)

    if chosen is not None:
        conversation_memory.set_resolved_advisor(session_id, chosen.wid, chosen.name)
        # Re-run the ORIGINAL question now that identity is settled, so
        # "which Yasir Ali?" -> "2" answers what was actually asked
        # instead of making the user retype it.
        return resolve(pending.original_text, db, session_id=session_id, _depth=1)

    # not a choice, and it stands on its own as a query -> user moved on
    if len(re.findall(r"\S+", cleaned)) > 4:
        conversation_memory.clear_pending_person(session_id)
        return None

    if pending.attempts >= MAX_CLARIFY_ATTEMPTS:
        conversation_memory.clear_pending_person(session_id)
        return Resolution(
            kind="clarify", entities={},
            clarify_message=(
                "I still couldn't tell which person you meant — try including their team, "
                "e.g. 'Yasir Ali in North/KPK'."
            ),
        )

    conversation_memory.set_pending_person(session_id, pending.candidates, pending.original_text)
    return Resolution(
        kind="clarify",
        entities={},
        clarify_message=format_person_disambiguation_reply(
            pending.candidates[0].name if pending.candidates else "that name", pending.candidates
        ),
        clarify_options=[c.label() for c in pending.candidates],
    )


def _plan(text: str, entities: dict, db: Session, session_id: str | None) -> QueryPlan:
    """Choose a planner (USE_LLM_PLANNER), with the rule-based one as the
    fallback in every failure case.

    The fallback is what makes the flag safe to flip in either direction:
    if the LLM planner is off, unreachable, out of quota, or returns
    something unusable, planning silently continues with the rule-based
    planner — so the worst case is exactly today's behaviour, never an
    error. Both planners emit the SAME QueryPlan shape, so nothing
    downstream (dispatch, resolver, compiler, formatter) can tell which
    one ran, which is also what makes an A/B comparison meaningful."""
    if llm_planner.is_enabled():
        try:
            planned = llm_planner.plan_query(text, entities, db, session_id)
            if planned is not None:
                return planned
            log.debug("LLM planner unavailable for %r — using rule-based planner", text)
        except Exception:
            # planning must never be the reason a request fails
            log.exception("LLM planner raised — falling back to the rule-based planner")

    return build_query_plan(text, entities)


def _ir_resolution(ir, entities: dict, **kwargs) -> Resolution:
    """Return an IR resolution, or a refusal when routing validation says
    the IR is complete but not answerable as asked.

    Every kind="ir" exit goes through here so the last gate before the
    compiler cannot be bypassed by adding a fourth return site later.
    """
    problem = routing.validate_route(ir)
    if problem is not None:
        routing.decide("Validation", "refused", problem)
        return Resolution(kind="clarify", ir=ir, entities=entities,
                          clarify_message=problem, **kwargs)
    routing.decide("Validation", "passed", "metric, level and period are all answerable")
    return Resolution(kind="ir", ir=ir, entities=entities, **kwargs)


def _pin_level(entities: dict, value: str, level: str) -> dict:
    """Drop the groundings of `value` at every level EXCEPT `level`.

    The ambiguity is that one name grounded at several levels at once;
    removing the ones the user did not pick leaves exactly the reading
    they chose, so _detect_ambiguous_entity no longer fires and the
    planner sees an unambiguous query.

    Only groundings of THIS value are touched — another entity that
    happens to be in the same message keeps its own.
    """
    from app.llm.entity_extractor import _AMBIGUITY_LEVELS

    pinned = dict(entities)
    pinned.pop("ambiguous_entity", None)
    lowered = value.lower()
    for other in _AMBIGUITY_LEVELS:
        if other == level:
            continue
        if other == "advisor":
            if str(pinned.get("advisor_name", "")).lower() == lowered:
                for key in ("advisor_name", "advisor_wid", "advisor_wids",
                            "advisor_names", "advisor_matches",
                            "advisor_resolution", "advisor_match_score",
                            "advisor_ambiguous", "advisor_multi"):
                    pinned.pop(key, None)
        elif str(pinned.get(other, "")).lower() == lowered:
            pinned.pop(other, None)
            plural = f"{other}s"
            if isinstance(pinned.get(plural), list):
                pinned[plural] = [v for v in pinned[plural] if v.lower() != lowered]
                if not pinned[plural]:
                    pinned.pop(plural, None)
    return pinned


def _asks_about_the_person(text: str, entities: dict, ambiguous: dict) -> bool:
    """Is this a bare question about the NAMED PERSON's own figures?

    True only when the name really is a person (it grounded at `advisor`
    too), the turn names a measure, and nothing in it asks for a group:

      "connects of Fawad Hafeez"          -> True   his own 54
      "Fawad Hafeez's team connects"      -> False  the zone's 949
      "Zonal Head Fawad Hafeez connects"  -> False  pinned already
      "top advisors under Fawad Hafeez"   -> False  a ranking wants members

    Each exclusion is a signal that already exists and is already owned
    elsewhere — the relation by reference_parser, the level word by
    intent_catalog, the ranking by TurnSpec — so this decides nothing on
    its own, it only reads them together.
    """
    from app.llm import intent_catalog as cat, reference_parser

    spec = conversation_context.specified(text, entities)
    if "advisor" not in (ambiguous.get("levels") or []):
        return False
    if not (spec and spec.metric):
        return False          # no measure named — a profile/roster question
    if reference_parser.parse(text):
        return False          # "X's team" asks for the group
    if cat.detect_level(text) is not None:
        return False          # an explicit level was stated
    if spec.ranking:
        return False          # a ranking enumerates members
    return True


# The levels at which a name means A PERSON HOLDING A ROLE. Derived from
# the chain by dropping `team`, the one chain level whose values are group
# names rather than people — so the priority below IS hierarchy.CHAIN's
# order (unit_head > zonal_head > bcm > advisor) and cannot drift from it.
_ROLE_LEVELS = tuple(lvl for lvl in hierarchy.CHAIN if lvl != "team")


def _highest_role(levels) -> str | None:
    """The senior-most role among the levels a name grounded at.

    Grounding IS the hierarchy relationship: a name reaches `unit_head`
    only because some advisor's `rm` column names them, `zonal_head` only
    via `portfolio_lead`, `bcm` only via `management_lead`. So the levels
    already in hand say which roles the person holds, and choosing the
    highest is a read of CHAIN — no traversal, no second resolver, and
    nothing here to keep in sync when the chain is rebound.

    None when the name is not purely a person: a value that is also a
    TEAM or COMPANY name is a different entity that happens to share the
    spelling, and no role ordering can settle which was meant.
    """
    levels = list(levels or [])
    # `region` mixes places with people (hierarchy.AMBIGUOUS_LEVELS): the
    # master-sheet rows still carry a regional head's NAME, which is why
    # every Unit Head here also grounds at `region`. It is the same person
    # under a stale column, not a fifth role, so it neither blocks the
    # decision nor wins it.
    ranked = [lvl for lvl in levels if lvl != "region"]
    if not ranked or any(lvl not in _ROLE_LEVELS for lvl in ranked):
        return None
    for level in _ROLE_LEVELS:          # CHAIN order: senior first
        if level in ranked:
            return level
    return None


def _authoritative_role(stated: str, levels) -> str:
    """The role a ROLE-SPECIFIC query about this person actually addresses.

    A named role is a way of pointing at someone, not a claim about which
    of their jobs the question is about. Haseeb Arslan is a Unit Head over
    75 advisors and, because those directly under him also name him in
    `management_lead`, he grounds at `bcm` over exactly one. "BCM Haseeb
    Arslan connects" therefore answered 0 — a true statement about a
    scope of one that reads as a false statement about him. There is one
    Haseeb Arslan and he leads 12,004 connects.

    So the stated role SELECTS the person, and the person's own hierarchy
    decides which scope that is: the senior-most role they hold. Same
    rule Phase 28 applies when no role is stated, from the same ranking
    (_highest_role over hierarchy.CHAIN), so naming the role and omitting
    it cannot reach different scopes for the same person.

    TWO CASES ARE LEFT ALONE, both deliberately:

    `advisor` is not promoted. "Advisor X" asks for the person as a leaf
    — their own figure — and promoting it would make a manager's own
    record unreachable, which is exactly the defect Phase 22 fixed.

    A role that is ALREADY the person's highest passes through unchanged,
    as does a role held by someone with nothing above it: Person C, a BCM
    and nothing more, stays a BCM. Promotion only ever moves UP the chain
    the codebase already declares, never down and never sideways.
    """
    if stated == "advisor":
        return stated
    authoritative = _highest_role(levels)
    if authoritative is None or authoritative == stated:
        return stated
    if _ROLE_LEVELS.index(authoritative) >= _ROLE_LEVELS.index(stated):
        return stated          # nothing senior to promote to
    routing.decide(
        "Level", f"read {authoritative!r} not {stated!r}",
        f"the query names {stated!r}, but this person's own hierarchy puts "
        f"them at {authoritative!r} — the senior role is who they are, and "
        f"the {stated!r} reading is a scope of a different size for the "
        "same name",
    )
    return authoritative


def _asks_for_the_group(text: str, entities: dict) -> bool:
    """Does this turn ask about the people UNDER the named person?

    Reads the signals that already own the question — a relation
    ("X's team", "under X") by reference_parser, a ranking by TurnSpec —
    rather than adding a third opinion. It is the complement of
    _asks_about_the_person above, and deliberately not its negation: a
    turn that is neither still means the person themselves.
    """
    from app.llm import reference_parser

    # A DIRECT question is about the people under someone by definition:
    # "how many advisors directly report to X" has no reading in which it
    # asks for X's own figure. Without this the name pins to `advisor` —
    # every manager here has an advisor row — and the turn answers with
    # the manager's own profile, discarding a direct_reports plan the
    # planner had already built.
    if entities.get("direct"):
        return True
    if reference_parser.parse(text):
        return True
    spec = conversation_context.specified(text, entities)
    return bool(spec and spec.ranking)


# "how many people are UNDER X" names no level at all, yet asks about the
# people beneath someone. Kept minimal and separate from the relation
# vocabulary reference_parser owns, which covers possessives ("X's team")
# and does not parse this shape.
#
# `under` and `below` are ALSO comparator words (comparators.py: "less
# than", "below", "under"), so a following number disqualifies the match
# — "advisors under 50 connects" is a threshold, not a manager.
#
# The VERB'S INFLECTION IS OPTIONAL. This listed "reports to" and
# "reporting to" but not the bare "report to", which is the form a plural
# subject produces: "advisors directly REPORT TO X". So the relation went
# unrecognised, "advisors" was read as naming who X is rather than what to
# return, and a question about X's reports answered with X's own profile.
_UNDER_RE = re.compile(
    r"\b(under|beneath|below|report(s|ing)?\s+to)\b(?!\s*\d)", re.I)

# "Who DOES Ali Murtaza REPORT TO" is the same two words pointing the
# other way: there the named person is the SUBORDINATE, and reading it as
# a relation makes the question about the people under him. The auxiliary
# is what separates the two — a forward relation ("advisors report to X")
# never has one — so it is the whole discriminator rather than a list of
# reverse phrasings kept in step with intent_catalog's.
_REVERSE_REPORT_RE = re.compile(r"\b(does|do|did)\s+.+?\s+report\s+to\b", re.I)


def _names_a_forward_relation(text: str) -> bool:
    """Does `text` place people UNDER the named person?"""
    return bool(_UNDER_RE.search(text)) and not _REVERSE_REPORT_RE.search(text)


def _measures_the_group(text: str, stated: str | None, ambiguous_levels) -> bool:
    """Is the level word about the GROUP under the person, not the person?

    Three shapes, all of which name a level while meaning the people
    beneath someone:

        "show all ADVISORS under Kaleem Satti"   the OUTPUT is advisors
        "TEAM size of Haseeb Arslan"             the TEAM is measured
        "how many people are UNDER Haseeb"       no level word at all

    A GENUINE TEAM SURVIVES THIS. `team` counts only when the ambiguous
    value is NOT itself grounded at team — if a person shares a name with
    a real team, `team` is one of the readings on offer and the question
    is exactly which was meant, so it is left to the clarification. And
    the whole function is reached only for a value that grounded at
    several hierarchy levels; "connects of Blue Area" grounds at team
    alone, produces no ambiguity, and never arrives here.

    `advisor` and `team` are the only levels that can be read this way —
    the one names the people a roster lists, the other the group they
    form — so no other level word is second-guessed.
    """
    from app.llm import intent_catalog as cat

    # ROSTER_RE wants the noun adjacent to its preposition ("advisors
    # under X"); "how many advisors ARE under X" is the same question with
    # a verb in between, so the relation phrase counts too.
    if stated == "advisor" and (cat.ROSTER_RE.search(text) or _names_a_forward_relation(text)):
        return True
    if stated == "team" and "team" not in (ambiguous_levels or []):
        return True
    if stated is None and _names_a_forward_relation(text):
        return True
    return False


def _subject_level_word(text: str, ambiguous_levels) -> str | None:
    """The level word that says WHO THE SUBJECT IS, not what to return.

    A question about a group names two levels and means different things
    by them:

        "show all ADVISORS under UNIT HEAD Kaleem Satti"
                   ^ what to return      ^ who Kaleem is

        "TEAM size of Haseeb Arslan"
         ^ what is measured    ^ who he is — his hierarchy decides

    detect_level returns the first entry in LEVEL_KEYWORDS order, so the
    output noun outranked the qualifier purely by table position.
    _pin_stated_level then pinned Kaleem Satti to `advisor` and deleted
    his unit_head grounding, and the question about 137 people was
    answered with one man's profile. "team size of X" fared differently
    and no better: `team` is not one of the person's role levels, so the
    pin declined outright and the reply asked which of four readings of
    one man was meant.

    So when the level word describes the GROUP, the subject's level comes
    from the sentence's other qualifier if it has one, and otherwise from
    the person's own hierarchy — `_highest_role`, the same ranking the
    possessive form ("X's team size") has used since Phase 28, which is
    why all these phrasings now reach one number.

    Two qualifiers settle nothing and fall through to the clarification
    rather than guessing, and a stated junior role is still corrected
    upward downstream by `_authoritative_role`.
    """
    from app.llm import intent_catalog as cat, token_match

    stated = cat.detect_level(text)
    if not _measures_the_group(text, stated, ambiguous_levels):
        return stated

    # An explicit level for the SUBJECT still wins: "advisors under Unit
    # Head X" names both, and the qualifier is the one about the person.
    others = [
        level for level, keywords in cat.LEVEL_KEYWORDS.items()
        if level != stated
        and level in (ambiguous_levels or [])
        and token_match.contains_any(text, keywords)
    ]
    if len(others) == 1:
        routing.decide(
            "Level", f"read {others[0]!r} not {stated!r}",
            f"{stated!r} names what the question measures while "
            f"{others[0]!r} names who the subject is — the first only won "
            "before because it comes first in LEVEL_KEYWORDS",
        )
        return others[0]
    if others:
        return None

    # No qualifier — the person's own hierarchy answers it, exactly as it
    # does when no level word is present at all.
    role = _highest_role([lvl for lvl in (ambiguous_levels or [])
                          if lvl in _ROLE_LEVELS])
    if role is None or role == "advisor":
        return None
    routing.decide(
        "Level", f"pinned {role}",
        f"the question measures the group under this person and names no "
        f"role for them, so their own hierarchy answers it — {role!r} is "
        "the senior-most role they hold",
    )
    return role


def _pin_stated_level(text: str, entities: dict) -> dict:
    """Narrow an ambiguous name to the level the QUERY ITSELF names.

    "connects of Zonal Head Faisal Hussain Naqvi" says which Faisal is
    meant. The level word was already detected — and then ignored for the
    purpose of choosing the entity: the name stayed grounded at
    zonal_head, bcm, region and advisor simultaneously, and
    _Intent.group_entity() picked whichever came first in
    GROUP_LEVEL_ORDER, which is `bcm`. The user's own words were
    outranked by an ordering constant, so the query answered with the
    BCM's four reports (227) instead of the Zonal Head's eleven (763).

    The narrowing itself is _pin_level, unchanged — the same function the
    clarification answer uses. That is the point: stating the level in
    the sentence and choosing it from the offered list are the same
    request, and they now go through the same code, so they cannot give
    different scopes.

    Deliberately narrow. It fires only when the text names a level AND
    the ambiguous value is actually grounded at that level, so a level
    word belonging to something else — a team called "Beverly Center", a
    metric containing "unit" — cannot pin anything.
    """
    from app.llm import intent_catalog as cat

    ambiguous = entities.get("ambiguous_entity")
    if not ambiguous:
        return entities

    stated = _subject_level_word(text, ambiguous.get("levels") or [])
    if stated is None or stated not in (ambiguous.get("levels") or []):
        return entities

    # A level word consumed by a POSSESSIVE names the ANSWER, not the
    # subject: "Ali Murtaza's unit head" asks who his unit head IS, and
    # pinning Ali to unit_head turns that into a query about the group
    # under him. reference_parser already owns which level a possessive
    # points at, so it is asked rather than re-detected here.
    from app.llm import reference_parser

    if any(ref.target_level == stated
           for ref in reference_parser.parse(text)):
        return entities

    promoted = _authoritative_role(stated, ambiguous.get("levels") or [])

    routing.decide(
        "Level", f"pinned {promoted}",
        f"the query names {stated!r} explicitly, which settles which "
        f"{ambiguous.get('value')!r} was meant — the same narrowing the "
        "clarification answer applies, so both reach the same scope",
    )
    pinned = _pin_level(entities, ambiguous["value"], promoted)
    if promoted != stated:
        # The planner reads the level word from the TEXT again, and the
        # text still says "BCM". Left alone it would scope the answer to
        # the promoted subject while shaping it as a list of BCMs — one
        # question resolved two ways. Recorded only when a promotion
        # actually overruled the text, so every other query keeps
        # detect_level's answer untouched. Meta, hence the underscore:
        # consumers that serialise entities skip these keys.
        pinned[PINNED_LEVEL_KEY] = promoted
    return pinned


def resolve(text: str, db: Session, session_id: str | None = None, _depth: int = 0,
            _pin: tuple | None = None) -> Resolution:
    cleaned = normalize(text)
    # One trace per user message. A split sub-query (_depth > 0) keeps
    # appending to its parent's trace rather than starting a new one, so
    # a compound message reads as a single ordered story.
    if _depth == 0:
        routing.start_trace(cleaned)

    if _SHOW_MORE_RE.match(cleaned.strip()) and conversation_memory.get_pagination(session_id) is not None:
        audit.decision("routing", "paginate",
                       "matched the 'show more' pattern AND this session has an active "
                       "pagination cursor — no re-parse, the stored IR is reused")
        return Resolution(kind="paginate", entities={})

    # Part 8 (light multi-intent): checked BEFORE shortcut classification
    # on purpose — classify_intent scores the whole string, so a compound
    # message ("top advisors by revenue; who was late today") can contain
    # a shortcut-triggering word (here "late") that would otherwise hijack
    # the ENTIRE message into a single shortcut instead of ever reaching
    # the splitter. A genuinely compound utterance is split into
    # independent sub-queries, each resolved through this same pipeline
    # (including its own shortcut check), and stitched into labeled
    # sections by chat_service. Depth-capped at 1 — a split segment is
    # never re-split, and only the LAST segment's IR ends up persisted to
    # conversation_memory (each resolve() call below writes it in turn) —
    # a known, documented limitation of the light version.
    if _depth == 0:
        segments = multi_intent.split_subqueries(cleaned)
        if segments is not None:
            audit.decision("routing", "multi_intent",
                           f"split_subqueries() found {len(segments)} independent sub-queries: "
                           f"{segments} — each resolved separately")
            sections = [(seg, resolve(seg, db, session_id, _depth=1)) for seg in segments]
            return Resolution(kind="multi", entities={}, sections=sections)

    # Phase 5: an in-flight "which Yasir Ali did you mean?" takes
    # precedence — "2" or "the one in Downtown" only means something as
    # the answer to that question. Answering it re-runs the ORIGINAL
    # query against the chosen wid, so the user doesn't retype it.
    pending_person = conversation_memory.get_pending_person(session_id)
    if pending_person is not None:
        served = _handle_pending_person(pending_person, cleaned, db, session_id)
        if served is not None:
            return served

    # An in-flight "the BCM or the Advisor?" takes precedence for the
    # same reason: "BCM" is a level WORD, it grounds to no entity, and it
    # only means anything as the answer to that question. Placed here,
    # before extraction, so the normal path never sees it as a query.
    pending_level = conversation_memory.get_pending_level(session_id)
    if pending_level is not None:
        served = _handle_pending_level(pending_level, cleaned, db, session_id)
        if served is not None:
            return served

    entities = extract_entities(cleaned, db)

    # A level the user just chose in answer to "the BCM or the Advisor?".
    # Applied here, immediately after extraction, so every stage below —
    # ambiguity detection, the planner, the IR — sees the unambiguous
    # reading rather than re-asking the question that was just answered.
    if _pin is not None:
        entities = _pin_level(entities, _pin[0], _pin[1])
    else:
        entities = _pin_stated_level(cleaned, entities)

    # M4: relations of the person the conversation is already about
    # ("how is his team doing"). Runs BEFORE the carry below on purpose —
    # the carry writes a remembered advisor INTO `entities`, and the
    # cross-turn gate needs to see whether THIS message named anyone.
    # Ordered after extraction and before planning, so an inferred entity
    # reaches the planner exactly as a literally-named one would.
    inferred_levels = cross_turn_resolver.resolve(
        cleaned, entities, db, session_id, PROVENANCE_KEY
    )

    # Phase 5: once the conversation has settled on a person, that choice
    # sticks. Two cases, both of which otherwise throw away what the user
    # already told us:
    #
    #  a) the message re-states the ambiguous NAME (including the re-run
    #     of the original question right after the user picked). Without
    #     this the same "which Yasir Ali?" question is asked forever —
    #     answering it would be impossible, since the answer is exactly
    #     what gets discarded.
    #  b) the message names nobody and refers back by pronoun ("what
    #     about his overdue").
    #
    # M4 adds one exclusion to (b): when cross-turn inference just bound
    # a GROUP the follow-up asked about, the subject of this message is
    # that group, not the person. Carrying the person as well would put
    # advisor_profile in the running against it — and win, since a
    # resolved identity outscores an entity summary — so "how is his team
    # doing" would answer with his own profile. The exclusion can only
    # fire on messages where inference bound something, which is new
    # behaviour by definition, so nothing that worked before changes.
    carried = conversation_memory.get_resolved_advisor(session_id)
    if carried:
        wid, name = carried
        carried_matches_candidates = wid in (entities.get("advisor_wids") or [])
        names_nobody = not entities.get("advisor_wids")
        followup_about_the_person = (
            names_nobody and _looks_like_person_followup(cleaned) and not inferred_levels
        )
        if carried_matches_candidates or followup_about_the_person:
            entities = {**entities, "advisor_wid": wid, "advisor_name": name,
                        "advisor_wids": [wid], "advisor_match_score": 1.0}
            entities.pop("advisor_ambiguous", None)
            entities.pop("advisor_resolution", None)

    # ---- Phase 1 routing gates -------------------------------------
    # All three run HERE, after extraction/cross-turn/carry have produced
    # the full identity picture and before any component commits to a
    # route. Each replaces a decision that used to be made with less
    # information than it needed. See app/llm/routing.py for the defects.

    # P2: a measure this system knows about and cannot compute always
    # explains itself. Checked before every other routing decision
    # because availability is a property of the METRIC, not of the query
    # shape — the old code reached this explanation only on the branch
    # where no person resolved, so naming the person in full downgraded
    # the answer to a profile card.
    unavailable = routing.unavailable_metric(cleaned)
    if unavailable is not None:
        routing.decide(
            "Metric", f"{unavailable.phrase} (unavailable)",
            f"declared in metric_aliases.UNAVAILABLE — {unavailable.reason}",
        )
        message = routing.explain_unavailable(unavailable)
        return Resolution(
            kind="clarify",
            # The plan travels with the refusal, exactly as the
            # clarify_metric branch below does. Consumers (the golden
            # harness, the trace) read plan.action to tell WHICH
            # clarification this is; moving the check earlier must not
            # change that contract, only the point at which it fires.
            plan=QueryPlan(action="clarify_metric",
                           entity_value=unavailable.phrase,
                           reason=message),
            entities=entities,
            clarify_message=message,
            clarify_options=clarification_options("metric", db),
        )

    # P1: shortcuts are a FALLBACK. classify_intent() now receives the
    # real entity dict (it was handed a hardcoded {} before, which made
    # its own entities.get("team") guard dead code in production), and a
    # resolved person or an explicit rate/percentage phrase outranks it.
    shortcut = classify_shortcut(cleaned, entities)
    if shortcut.intent in SHORTCUT_INTENTS:
        allowed, why = routing.shortcut_allowed(cleaned, entities)
        if allowed:
            routing.decide("Shortcut", shortcut.intent, why)
            audit.mark_rule_path(
                f"classify_intent() matched shortcut '{shortcut.intent}' and routing "
                f"allowed it — {why}. Answered from a canned handler.",
                chose=f"shortcut:{shortcut.intent}",
            )
            return Resolution(kind="shortcut", shortcut_intent=shortcut.intent,
                              entities=entities)
        routing.decide("Shortcut", "skipped", why)
    else:
        routing.decide("Shortcut", "skipped", "no shortcut intent matched")

    # P3: the user named a person we could not ground. Answering about
    # their team (metric_def.primary_level) or about the whole roster
    # silently changes the subject of the question, which is the one
    # thing routing must never do.
    unresolved = routing.unresolved_subject(cleaned, entities)
    if unresolved is not None:
        routing.decide(
            "Advisor", f"{unresolved} (NOT FOUND)",
            "the query names a person identity resolution could not ground — "
            "asked who was meant rather than answering about their group",
        )
        return Resolution(
            kind="clarify",
            entities=entities,
            clarify_message=routing.explain_unresolved_subject(unresolved),
        )

    # Part 8: a genuinely unsupported time window (last month, yesterday,
    # this week, past N days, a custom date range) must never silently
    # fall back to MTD — say so plainly instead of guessing wrong.
    if entities.get("period_unsupported"):
        audit.decision("routing", "clarify:period_unsupported",
                       f"entity extraction flagged an unsupported time window "
                       f"({entities['period_unsupported']!r}) — refused rather than "
                       "silently defaulting to MTD")
        return Resolution(
            kind="clarify",
            entities=entities,
            clarify_message=entities["period_unsupported"] + ". Try 'this month', 'year to date', or 'last 3 months' instead.",
        )

    if entities.get("advisor_wids"):
        routing.decide(
            "Advisor",
            f"{entities.get('advisor_name')} ({entities.get('advisor_match_score')})",
            "grounded by entity extraction",
        )
    if entities.get("period"):
        routing.decide("Period", str(entities["period"]), "from the query text")

    plan = _plan(cleaned, entities, db, session_id)
    routing.decide("Planner", plan.action, f"metric={plan.metric} level={plan.level}")
    # Phase 7: record the planner's decision HERE, where it's made —
    # recording it from Resolution in chat_service only captured
    # kind=="plan" outcomes, so a clarification or an IR-routed query (the
    # two shapes most likely to be reported as wrong) showed plan=null,
    # hiding exactly the routing decision the trace exists to explain.
    tracing.record_plan(plan)
    tracing.record_entities(entities)
    # Same two artifacts for the readable audit. It keeps the FULL entity
    # dict (tracing keeps a curated subset), because "the requirement was
    # extracted but then dropped" is only visible against everything the
    # extractor actually produced.
    audit.record_entities(entities)
    audit.record_plan(plan)

    # A name matched under more than one hierarchy level (entity_extractor.
    # _detect_ambiguous_entity) — ask which one was meant instead of
    # silently picking one, fully rule-based so this can't itself introduce
    # a spurious guess the way routing an ambiguous name to the LLM could.
    if plan.action == "clarify_ambiguous":
        ambiguous = plan.ambiguous or {}
        value = ambiguous.get("value", "")
        levels = ambiguous.get("levels", [])

        # THE PERSON IS THE DEFAULT. "connects of X" asks about X, and
        # every manager is also an advisor with their own figures. Being a
        # BCM must not turn a question about someone into a question about
        # the people under them: Fawad Hafeez's own 54 connects came back
        # as his zone's 949, and there was no phrasing that reached the
        # 54 — the person reading was unreachable once the name grounded
        # at a manager level.
        #
        # Only a bare metric question takes this default. A relation
        # ("X's team"), an explicit level ("Zonal Head X", already pinned
        # above) or a ranking all say the group is wanted, and each is
        # excluded here rather than being re-decided later.
        if _asks_about_the_person(cleaned, entities, ambiguous):
            routing.decide(
                "Level", "pinned advisor",
                f"{value!r} names a person and the turn asks for their own "
                "measure — a manager's own record, not the people under them",
            )
            return resolve(cleaned, db, session_id=session_id, _depth=_depth or 1,
                           _pin=(value, "advisor"))

        # THE HIGHEST ROLE IS THE DEFAULT FOR THE GROUP. Asking "which
        # Haseeb Arslan?" of a name that is ONE person wearing four hats
        # is a question with no answer — he is the Unit Head, and being a
        # Unit Head is also why he grounds at zonal_head and bcm (his
        # reports' columns name him at every level beneath). The offered
        # options were four readings of one man, so no choice was wrong
        # and none could be made confidently.
        #
        # Only a turn that asks for the GROUP takes this default, and it
        # takes the senior role because that is the scope the person
        # actually leads. A question about the person is left to the
        # branch above, so RULE 1 (a bare person query answers with their
        # OWN figure) is untouched.
        wants_group = _asks_for_the_group(cleaned, entities)
        role = _highest_role(levels) if wants_group else None
        if role is not None and role != "advisor":
            routing.decide(
                "Level", f"pinned {role}",
                f"{value!r} holds {levels} because subordinates name them at "
                f"each — one person, so the senior role {role!r} is the team "
                "they lead, and there is nothing to ask",
            )
            return resolve(cleaned, db, session_id=session_id, _depth=_depth or 1,
                           _pin=(value, role))

        # AND A QUESTION ABOUT THE PERSON IS ABOUT THE PERSON, even when
        # it names no measure. "who is X?", "details of X" and a bare name
        # fell through both branches above — the first wants a measure
        # named, the second a group — so the most natural way to ask about
        # someone was the one way that got a question back: "the Unit Head
        # or the Zonal Head or the BCM or the Advisor?", four readings of
        # one man, for a sentence that plainly means the man.
        #
        # It also bit a BCM who is nothing else: "details of Abdul Qadir"
        # asked "BCM or Advisor?" of a person whose highest role is not in
        # doubt. Nothing about the PERSON decided any of this — only
        # whether the wording happened to name a measure or a group.
        #
        # `advisor` rather than the senior role, because these words ask
        # who someone IS, and that is the same answer a single-role
        # advisor already gets for the same sentence. Their role and the
        # team they lead belong IN that answer, not instead of it.
        #
        # Gated on _highest_role, which is None as soon as the name also
        # reads as a TEAM or COMPANY — a different entity that happens to
        # share a spelling, where no ranking settles anything and the
        # question is still the honest reply.
        if not wants_group and _highest_role(levels) is not None:
            routing.decide(
                "Level", "pinned advisor",
                f"{value!r} names one person wearing several hats and the turn "
                "asks about them rather than about a measure or their team — "
                "which hat they wear settles nothing a profile does not say",
            )
            return resolve(cleaned, db, session_id=session_id, _depth=_depth or 1,
                           _pin=(value, "advisor"))

        options = [hierarchy.label_for(lvl) for lvl in levels]
        audit.decision("routing", "clarify:ambiguous_entity",
                       f"{value!r} grounded at more than one hierarchy level {levels} — "
                       "asked which was meant rather than picking one")
        # Record that a question is OUTSTANDING. Every other
        # clarification path already does this (set_pending_person for
        # "which Yasir Ali", set_pending for a missing slot); this one
        # asked and stored nothing, so the answer arrived next turn as a
        # bare, contextless "BCM" and resolved to nothing.
        conversation_memory.set_pending_level(session_id, value, levels, cleaned)
        return Resolution(
            kind="clarify",
            # As with clarify_metric: the plan travels with the refusal so
            # a consumer can see WHICH clarification this is, not merely
            # that the pipeline stopped.
            plan=plan,
            entities=entities,
            clarify_message=f"'{value}' could mean the {' or the '.join(options)} — which did you mean?",
            clarify_options=options,
        )

    # F6: the user named a measure this system has no metric for. Handled
    # here, before the rule-based/LLM split, because the answer is the
    # same either way — there is nothing to rank by, and the failure this
    # replaces was answering anyway with whatever default was to hand.
    # Every widening tier (exact, fuzzy, embedding) has already run inside
    # metric_intent.detect(), so this is the end of the line rather than a
    # shortcut past them.
    # Context is established BEFORE any "do we have enough to answer?"
    # decision. Asking the user for a measure the previous turn already
    # supplied is the same defect as answering without one — both ignore
    # what the conversation has already established.
    turn_spec = conversation_context.specified(cleaned, entities)
    prior_ir = conversation_memory.get(session_id)
    _context = conversation_context.ellipsis(turn_spec, prior_ir is not None)

    # try_patch's `plan_action` parameter means "the rule planner's
    # verdict on this message STANDING ALONE", and its gate depends on
    # that reading: "only Graana" plans as `summary`, which is how the
    # patcher recognises a narrowing. The carry below re-plans, replacing
    # that verdict with one made WITH the inherited context, so the
    # standalone one is kept here for the patcher. Handing it the
    # re-planned action instead made it decline every narrowing follow-up
    # and take a full parse — an LLM round trip per turn, and a
    # comparison that lost its sides on the way through.
    standalone_action = plan.action

    # Phase 10: the carry. What the previous turn supplies to THIS turn is
    # decided by conversation_context — the same owner and the same
    # ownership rule as merge() — and handed to the planner in the shape
    # extraction would have produced, so the precedence table re-scores
    # the turn with the full picture and decides what it now is.
    #
    # This generalises a branch that used to fire only on
    # plan.action == "clarify_metric" ("now only IMARAT"). Gating on the
    # planner's verdict meant the carry reached only the turns where the
    # planner had FAILED; a turn it resolved confidently but wrongly —
    # "what about Downtown?" planned as a standalone entity summary —
    # never got here at all, so the measure named one message earlier was
    # dropped and the answer was a generic card. The condition now asks
    # what the TURN specified, which is what the decision was always
    # about.
    #
    # Still the one place a second build_query_plan() runs, and still
    # only for an incomplete follow-up. Re-planning is what keeps intent
    # single-owned — the alternative was writing an action here.
    carry = conversation_context.carry_into_plan(
        prior_ir, entities, turn_spec, _context,
        needs_second_subject=(plan.action == "comparison_incomplete"),
    )
    if carry:
        routing.decide(
            "Context", "carry " + ", ".join(sorted(carry.entities)),
            "the previous turn supplies what this one left unsaid, before "
            "planning: " + "; ".join(carry.reasons),
        )
        plan = _plan(cleaned, {**entities, **carry.entities}, db, session_id)
        if plan.metric is None and "metric" in carry.entities:
            plan.metric = carry.entities["metric"]
        tracing.record_plan(plan)
        audit.record_plan(plan)

    # A (subject, measure) pairing this system cannot represent. Checked
    # before the plan is served, because every path below would answer it
    # by attaching every measure to one subject.
    if _distributes_metrics(cleaned, plan):
        labels = " and ".join(metric_ontology.measure_label(k) for k in plan.metrics)
        routing.decide(
            "Validation", "refused",
            f"the turn names {len(plan.metrics)} measures against more than one "
            "subject; a plan carries one subject and cannot say which measure "
            "belongs to whom — asked rather than attributing both to one person",
        )
        return Resolution(
            kind="clarify", plan=plan, entities=entities,
            clarify_message=(
                f"I can't tell which of {labels} you want for which person in one "
                "question. Ask me for one person at a time, or compare them on the "
                f"same measures — e.g. 'compare them on {labels}'."
            ),
        )

    # A capability this system does not have. Distinct from a
    # clarification (there is nothing the user could add that would make
    # it answerable) and from no-data (the query was never run). The
    # reason comes from ir_validator's registry so there is one list of
    # what we cannot do, with one wording each.
    if plan.action in _UNSUPPORTED_ACTIONS:
        reason = _UNSUPPORTED_ACTIONS[plan.action]
        routing.decide("Response", "unsupported", reason)
        return Resolution(
            kind="unsupported", plan=plan, entities=entities,
            clarify_message=reason,
        )

    if plan.action == "clarify_metric":
        audit.decision("routing", "clarify:metric_unresolved",
                       f"the query named a measure ({plan.entity_value!r}) that resolves to no "
                       "metric — refused rather than ranking by the default and "
                       "presenting it as the answer")
        return Resolution(
            kind="clarify",
            # The plan travels with the refusal so a consumer (the trace,
            # the golden-query harness) can see WHICH clarification this
            # is rather than only that the pipeline gave up.
            plan=plan,
            entities=entities,
            clarify_message=metric_intent.clarification(
                plan.entity_value or plan.reason,
                reason=plan.reason if plan.reason != plan.entity_value else None,
            ),
            # The same option list the validator offers when the gap is
            # 'which metric' — one source for what we can rank by.
            clarify_options=clarification_options("metric", db),
        )

    # an in-flight clarification takes precedence: a bare "revenue" or
    # "Blue Area" only means something as the answer to last turn's question
    pending = conversation_memory.get_pending(session_id)
    if pending is not None:
        served = _handle_pending(pending, cleaned, entities, plan, db, session_id)
        if served is not None:
            audit.decision("routing", "answer_to_pending_clarification",
                           f"a clarification was in flight (missing={pending.missing}); "
                           "this message was read as its answer, not as a new query")
            return served

    # short follow-up modifiers ("only Graana", "top 5", "sort ascending")
    # patch the previous turn's IR deterministically — no LLM round trip.
    # try_patch declines anything that stands alone as its own query.
    if prior_ir is not None:
        patched = try_patch(prior_ir, cleaned, entities, standalone_action, turn_spec)
        if patched is not None:
            result = validate_ir(patched, db)
            if result.is_valid:
                # The patcher performs a merge, so it owes the same
                # account of it. Without this the turns whose inheritance
                # is MOST implicit — "top 5", "year to date", which carry
                # every field of the previous query untouched — were the
                # only ones with no Context step at all.
                patch_merge = conversation_context.describe_patch(
                    prior_ir, result.ir, _context)
                routing.decide("Context", patch_merge.trace(),
                               f"previous: {conversation_context.summarise(prior_ir)}\n"
                               f"{patch_merge.detail()}\n"
                               f"merged: {conversation_context.summarise(result.ir)}")
                conversation_memory.set(session_id, result.ir)
                audit.mark_rule_path(
                    "try_patch() recognised this as a follow-up modifier on the previous "
                    "turn's IR — patched deterministically, no LLM call",
                    chose="ir_patch",
                )
                audit.record_ir(result.ir)
                return _ir_resolution(result.ir, entities)
            # an invalid patch falls through to the normal parse path

    # Phase 5: a name matching several real people — record the question
    # (candidates + the original query) so the next message can answer it,
    # then ask. Never picked here, and never routed to the LLM, which has
    # no way to know which wid was meant either.
    if plan.action == "clarify_person":
        conversation_memory.set_pending_person(session_id, plan.person_candidates, cleaned)
        audit.decision("identity", "clarify_person",
                       f"{plan.entity_value!r} matched {len(plan.person_candidates)} real "
                       "people — asked which one instead of picking")
        audit.record_advisor(None, plan.entity_value, status="ambiguous",
                             candidates=plan.person_candidates)
        return Resolution(
            kind="clarify",
            plan=plan,
            entities=entities,
            clarify_message=format_person_disambiguation_reply(
                plan.entity_value, plan.person_candidates
            ),
            clarify_options=[c.label() for c in plan.person_candidates],
        )

    # Phase 5: a query that resolved to exactly one person establishes who
    # the conversation is about, so a later pronoun follow-up has a
    # subject without asking again.
    if plan.entity_wid is not None:
        conversation_memory.set_resolved_advisor(session_id, plan.entity_wid, plan.entity_value)
        audit.record_advisor(plan.entity_wid, plan.entity_value, status="resolved",
                             candidates=plan.person_candidates)
        audit.decision("identity", f"wid={plan.entity_wid}",
                       f"entity resolution grounded {plan.entity_value!r} to a single wid; "
                       "the conversation's subject is now this person")

    # lookup/summary/attendance_filter stay on the simple rule-based path,
    # but only when the query doesn't look compound — "summary for Graana
    # AND Downtown" belongs to the semantic parser, not a single-entity plan.
    if _is_rule_based(plan) and (
        plan.action in _COMPOUND_EXEMPT_ACTIONS
        or not semantic_parser.looks_compound(cleaned, entities)
    ):
        audit.mark_rule_path(
            f"plan.action={plan.action!r} is in _RULE_BASED_ACTIONS and "
            + (
                f"is compound-exempt ({plan.action!r} in _COMPOUND_EXEMPT_ACTIONS)"
                if plan.action in _COMPOUND_EXEMPT_ACTIONS
                else "looks_compound() returned False"
            )
            + " — returned before semantic_parser.parse(), so NO LLM call is made "
            "for this query regardless of NLU_MODE"
        )
        # Phase 10: this path ANSWERS, and until now it neither read nor
        # wrote conversation state — the one exit that could leave the
        # follow-up base describing a turn the user had already moved on
        # from. An advisor profile, a roster or a hierarchy chain is not a
        # QueryIR, so there is nothing honest to store; what must not
        # happen is the PREVIOUS turn's IR staying behind as though this
        # one never occurred.
        #
        # The clarifying actions are exempt because they do not answer —
        # chat_service._dispatch renders them as a clarification, so the
        # topic has not moved and the context the user is being asked to
        # complete must still be there when they do.
        if plan.action not in _CLARIFYING_ACTIONS:
            if prior_ir is not None:
                routing.decide(
                    "Context", "reset",
                    f"answered on the rule-based plan path as {plan.action!r}, which "
                    f"no QueryIR describes — the previous context "
                    f"({conversation_context.summarise(prior_ir)}) is no longer what "
                    "the conversation is about, so it is dropped rather than left "
                    "for the next turn to inherit",
                )
            conversation_memory.clear_last_ir(session_id)
        return Resolution(kind="plan", plan=plan, entities=entities)

    audit.decision(
        "routing", "semantic_parser",
        f"plan.action={plan.action!r} is "
        + ("not in _RULE_BASED_ACTIONS" if plan.action not in _RULE_BASED_ACTIONS
           else "rule-based but looks_compound() returned True")
        + " — handed to the LLM semantic parser",
    )
    outcome = semantic_parser.parse(cleaned, entities, db, session_id, plan=plan)

    if outcome.ir and not outcome.missing:
        # THE carry-over point. The other two kind="ir" exits already
        # derive from the previous turn — the pending-slot fill answers a
        # question we asked, and try_patch returns a copy of the prior IR
        # — so this fresh parse is the only place context could be lost,
        # and before Phase 2 it always was: plan_to_ir() builds from this
        # turn's words alone.
        merged = conversation_context.merge(
            prior_ir, outcome.ir, turn_spec,
            conversation_context.ellipsis(turn_spec, prior_ir is not None),
        )
        routing.decide("Context", merged.trace(),
                       f"previous: {conversation_context.summarise(prior_ir)}\n"
                       f"{merged.detail()}\n"
                       f"merged: {conversation_context.summarise(merged.ir)}")
        # Phase 4: stored unconditionally, and only here. semantic_parser
        # used to store the pre-merge IR from inside _finish, so the IR
        # that ANSWERED and the IR the next turn INHERITED could differ
        # whenever the merge changed anything. One writer, one value, and
        # it is the one the user was shown.
        conversation_memory.set(session_id, merged.ir)
        return _ir_resolution(merged.ir, entities,
                              used_llm_fallback=outcome.used_llm)

    if outcome.ir and outcome.missing:
        # Part 10: "low" confidence means the parse itself is too shaky to
        # trust `missing` enough to ask about one specific slot — reject
        # outright and ask the user to rephrase, instead of setting a
        # pending clarification that's likely to keep missing the mark.
        # Never executed (this never returns kind="ir").
        if outcome.ir.confidence_level == "low":
            return Resolution(
                kind="clarify",
                ir=outcome.ir,
                entities=entities,
                used_llm_fallback=outcome.used_llm,
                clarify_message=_GIVE_UP_MESSAGE,
            )
        # "medium": remember what we asked, so the next message can answer
        # it (slot-filling, P6) instead of starting from scratch
        conversation_memory.set_pending(session_id, outcome.ir, outcome.missing)
        message, options = _clarify(outcome.missing, db)
        return Resolution(
            kind="clarify",
            ir=outcome.ir,
            entities=entities,
            used_llm_fallback=outcome.used_llm,
            clarify_message=message,
            clarify_options=options,
        )

    return Resolution(
        kind="clarify",
        entities=entities,
        clarify_message=(
            f"I'm not tracking that one, sorry! I can help with: "
            f"{describe_available_metrics()} — or look up an advisor, team, or company by name."
        ),
    )
