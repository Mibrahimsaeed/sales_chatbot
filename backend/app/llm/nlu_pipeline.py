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
    cross_turn_resolver, hierarchy, llm_planner, metric_intent, multi_intent,
    routing, semantic_parser,
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
    # "comparison" is a COMPLETE deterministic parse: two grounded
    # entities plus an explicit comparison phrase. It is also exempt from
    # the looks_compound() gate below — that heuristic routes multi-entity
    # queries to the LLM precisely BECAUSE the rule-based planner used to
    # have no way to express them, which is no longer true.
    "comparison",
    "comparison_incomplete",
)

# Actions complete enough to serve directly even when looks_compound()
# would otherwise send the query to the LLM.
_COMPOUND_EXEMPT_ACTIONS = ("comparison", "comparison_incomplete")

# Part 8: typed "show more" — the alternative to clicking the button
# (POST /chat/more, see app/api/chat.py). Only recognized when there's an
# active pagination cursor for this session (conversation_memory); with
# no cursor, "more" et al. fall through to the normal pipeline unchanged
# (verified against intent_detector's rules — none of them match these
# phrases, so this can't misfire and steal a real query).
_SHOW_MORE_RE = re.compile(r"^(show more|more|next|next page|load more)$", re.I)


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


# A follow-up that carries a pronoun or a bare possessive is asking about
# the person already under discussion ("what about his overdue", "and her
# team"). Deliberately narrow — a message naming its own subject resolves
# that subject normally and never reaches this.
_PERSON_FOLLOWUP_RE = re.compile(
    r"\b(he|him|his|she|her|hers|they|them|their|this person|that person)\b", re.I
)


def _looks_like_person_followup(text: str) -> bool:
    return bool(_PERSON_FOLLOWUP_RE.search(text))


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


def resolve(text: str, db: Session, session_id: str | None = None, _depth: int = 0) -> Resolution:
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

    entities = extract_entities(cleaned, db)

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
        options = [hierarchy.label_for(lvl) for lvl in levels]
        audit.decision("routing", "clarify:ambiguous_entity",
                       f"{value!r} grounded at more than one hierarchy level {levels} — "
                       "asked which was meant rather than picking one")
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

    if plan.action == "clarify_metric" and _context.is_elliptical and (
        prior_ir is not None and prior_ir.metric is not None
    ):
        # "now only IMARAT" after "show Downtown pipeline" names a
        # subject and no measure. Clarifying here would ask which metric
        # the user meant one turn after they said it.
        routing.decide(
            "Context", f"inherit metric {prior_ir.metric.key}",
            "this turn named a subject but no measure, and the previous turn "
            "supplies one — answered instead of re-asking",
        )
        plan.action = "leaderboard"
        plan.metric = prior_ir.metric.key

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
        patched = try_patch(prior_ir, cleaned, entities, plan.action, turn_spec)
        if patched is not None:
            result = validate_ir(patched, db)
            if result.is_valid:
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
    if plan.action in _RULE_BASED_ACTIONS and (
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
        routing.decide("Context", merged.trace(), merged.detail())
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
