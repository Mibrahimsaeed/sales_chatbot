"""
Orchestrates the semantic-parse step of the pipeline (see nlu_pipeline.py's
module docstring for the full order). Two modes, switched by
settings.nlu_mode (NLU_MODE env var):

  "llm_first" (default): the LLM parses EVERY analytical query reaching
      this module into a QueryIR — the rule-based plan is computed only
      as the fail-soft degrade target for LLM failure/invalid output.
      This is the P1 inversion of the NLU rework: regex/rule parsing
      systematically under-expressed real business queries, so rules are
      now the safety net, not the front door.

  "rules_first": the pre-inversion behavior, preserved verbatim as the
      rollback path — rule-based fast path for simple leaderboards, fuzzy
      metric widening, LLM only for compound-looking queries.

Both modes end at the same place: IR Validator/Grounder, and never a
dead end — LLM unavailability degrades to the rule-based plan.
"""

from __future__ import annotations

import re

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core import tracing
from app.core.config import settings
from app.llm import conversation_memory, hierarchy, routing, semantic_retrieval
from app.llm import grounding as grounding_mod
from app.llm import hierarchy_grounding, semantic_validation
from app.llm.entity_extractor import (
    get_known_teams, get_known_companies,
    get_known_unit_heads, get_known_zonal_heads, get_known_bcms,
)
from app.llm.fallback_reasoning import fuzzy_resolve_metric
from app.llm.ir_validator import validate_ir
from app.llm.llm_client import call_llm_structured, QUERY_IR_JSON_SCHEMA
from app.llm.prompt_builder import build_ir_prompt, prompt_fingerprint
from app.llm.query_ir import QueryIR, plan_to_ir
from app.llm.query_planner import QueryPlan, build_query_plan
from app.core.logger import get_logger

log = get_logger("llm.semantic_parser")

_COMPOUND_HINTS = re.compile(
    r"\b(but|compare|comparison|vs\.?|versus|still|and also|except|excluding)\b", re.I
)


def looks_compound(text: str, entities: dict) -> bool:
    """Cheap heuristic gate: does this query look like it needs more than
    query_planner.py's single-metric/single-filter QueryPlan can express?
    Multiple matches at ANY hierarchy level (team/company/unit_head/
    zonal_head/business_center), a threshold, or boolean-language keywords
    — any of these is a signal to route to the LLM semantic parser instead
    of trusting the rule-based fast path."""
    if any(len(entities.get(entity_key, [])) > 1 for entity_key in hierarchy.LEVEL_ENTITY_KEYS.values()):
        return True
    if entities.get("thresholds"):
        return True
    if _COMPOUND_HINTS.search(text):
        return True
    return False


class ParseOutcome:
    def __init__(self, ir: QueryIR | None, missing: list[str], used_llm: bool):
        self.ir = ir
        self.missing = missing
        self.used_llm = used_llm


def _call_llm_for_ir(text: str, entities: dict, db: Session, session_id: str | None) -> QueryIR | None:
    """One LLM round trip -> QueryIR, or None on any failure (no key,
    provider error, schema-invalid output).

    Records the attempt to the request trace either way — see
    tracing.record_llm_parse. "The model was asked and could not answer"
    and "the model was never asked" produce the same absent IR here, and
    without this they produced the same absent trace too.
    """
    prior_ir = conversation_memory.get(session_id)
    prompt = build_ir_prompt(
        text,
        get_known_teams(db),
        get_known_companies(db),
        grounded_entities=entities,
        prior_ir_json=prior_ir.model_dump_json() if prior_ir else None,
        recent_turns=conversation_memory.recent_turns(session_id),
        known_unit_heads=get_known_unit_heads(db),
        known_zonal_heads=get_known_zonal_heads(db),
        known_bcms=get_known_bcms(db),
    )
    def _trace(succeeded: bool, reason: str | None = None) -> None:
        tracing.record_llm_parse(
            attempted=True,
            succeeded=succeeded,
            model=settings.openai_model,
            prompt_hash=prompt_fingerprint(),
            prompt_tokens=len(prompt) // 4,
            fallback_used=not succeeded,
            fallback_reason=reason,
        )

    raw = call_llm_structured(prompt, QUERY_IR_JSON_SCHEMA, schema_name="query_ir")
    # The model's output BEFORE pydantic and before validate_ir touches
    # it. This is the only point where "the LLM got it wrong" and "we
    # lost it afterwards" are still distinguishable — every later log
    # shows an IR that something downstream may already have rewritten.
    log.info("RAW LLM QueryIR: %s", raw)
    if not raw:
        _trace(False, "provider returned nothing (unreachable, timeout, or refused)")
        return None
    try:
        ir = QueryIR.model_validate(raw)
    except ValidationError as e:
        log.warning(f"LLM returned a QueryIR that failed validation: {e}")
        _trace(False, f"schema validation failed: {type(e).__name__}")
        return None
    _trace(True)
    return ir


# Plan actions the rule planner can turn into a complete QueryIR.
# Phase 5B added "comparison": it used to execute on the plan path
# through comparison_service, and once that path was removed the fast
# path below had to recognise it or every comparison degraded to
# missing=["intent"]. Named once so the four checks that used to spell
# out "leaderboard" cannot drift apart.
# Phase 7 added "group_metric": a first-class classification for "one
# group, one measure" that compiles to the same scoped QueryIR a
# leaderboard would. Listed here so the fast path recognises it — without
# this every group-metric query degraded to missing=["intent"].
_IR_ACTIONS = ("leaderboard", "comparison", "group_metric")


def _rule_based_ir(text: str, entities: dict, plan: QueryPlan) -> QueryIR | None:
    """The deterministic degrade target: the rule plan if it resolved to a
    leaderboard, else a widening attempt — fuzzy synonym match first
    (typos, "revnue"), then embedding-based semantic retrieval (Part 8)
    for genuine paraphrases with no lexical overlap ("who's crushing it"
    for achievement_pct). None if nothing produces something answerable.
    """
    if plan.action in _IR_ACTIONS:
        return plan_to_ir(plan, entities)
    if plan.action == "unresolved":
        widened = fuzzy_resolve_metric(text) or semantic_retrieval.retrieve_metric(text)
        if widened:
            widened_entities = {**entities, "metric": widened}
            widened_plan = build_query_plan(text, widened_entities)
            if widened_plan.action in _IR_ACTIONS:
                log.info(f"Fallback reasoning resolved '{text}' via widened metric match: {widened}")
                return plan_to_ir(widened_plan, widened_entities)
    return None


def _finish(ir: QueryIR, db: Session, session_id: str | None, used_llm: bool,
            entities: dict | None = None) -> ParseOutcome:
    """Validate and return. Deliberately does NOT store the IR.

    Phase 4: this used to write conversation_memory itself, making four
    writers of last_ir across two modules. Worse, it wrote the PRE-MERGE
    IR — the one built from this turn's words alone — so the turn that
    actually answered and the turn the next message inherited from could
    differ. nlu_pipeline.resolve() owns conversation state now and stores
    the merged IR once, after conversation_context has run.
    """
    # THE EXTRACTOR'S FINDINGS TRAVEL WITH THE IR. They already reach the
    # model as prompt text, but text is advice: for "revenue of AMD year
    # to date" the prompt said AMD was a team and the model typed it
    # `company` anyway. Handing the same dict to the validator is what
    # makes deterministic grounding a constraint rather than a suggestion
    # — see ir_validator._authoritative_levels.
    result = validate_ir(ir, db, entities=entities)
    result.ir.nlu_mode = settings.nlu_mode
    return ParseOutcome(ir=result.ir, missing=result.missing, used_llm=used_llm)


class Interpretation:
    """What the LLM understood, before any deterministic component has had
    a say.

    `model` is the SemanticModel (Phase 1) — the statement of MEANING.
    `ir` is the same parse in the execution contract, carried alongside so
    the existing compiler, validator and dispatcher keep working unchanged
    while the migration proceeds; Phase 6 replaces that with a conversion
    built from the model.

    `reached_llm` records whether the provider actually answered. It is
    the assertion Phase 2 exists to make: not "the LLM was preferred", but
    "the LLM was asked, for this query, before anything else decided what
    the query meant".

    `grounding` is PHASE 4: what the database says about the entities the
    model named. It is a parallel structure rather than an edit to
    `model`, so the interpretation and its verification stay separable —
    a mismatch is visible as "the model said team, the database has it as
    an advisor" instead of disappearing into a rewritten field.

    `hierarchy` and `verdict` are PHASE 5: whether the requested
    relationship exists in the data, and whether the interpretation is
    executable as stated. Both are reports carried alongside — validation
    has no repair path, so nothing here rewrites the parse.
    """

    def __init__(self, model, ir, reached_llm: bool, reason: str | None = None,
                 grounding=None, hierarchy=None, verdict=None):
        self.model = model
        self.ir = ir
        self.reached_llm = reached_llm
        self.reason = reason
        self.grounding = grounding
        self.hierarchy = hierarchy
        self.verdict = verdict

    @property
    def understood(self) -> bool:
        """Did the model return something the pipeline can act on?"""
        return self.ir is not None


def _resolve_unstated_person_level(ir, entities: dict, db: Session) -> None:
    """A person named with no level is asked about at their HIGHEST role.

    "connects of Naina" names a person and not a job. If Naina is a Unit
    Head who also has an advisor row, answering from the advisor row is a
    true statement about one row and a false one about her: it reports a
    scope of one where the question meant her whole organisation.

    THIS IS RESOLUTION, NOT REPAIR, and the difference is the user's own
    words. Phase 11 switched off the rewrites that overruled a level the
    query STATED; this fires only when the query stated none —
    `entities["level_word"]` is empty — so nothing the user or the model
    asserted is being contradicted. "connects of Naina as an advisor" and
    "connects of unit head Naina" both name a level and are left exactly
    as they are.

    Reuses the existing ranking rather than adding one:
    hierarchy_grounding.highest_level_of reads the levels a name grounds
    at and picks the senior-most by hierarchy.CHAIN — the same answer
    nlu_pipeline._authoritative_role gives on the rule-based path, so the
    two paths cannot disagree about who somebody is.

    Only ever promotes UP the chain, never down and never sideways.
    """
    from app.llm.nlu_pipeline import _ROLE_LEVELS

    if entities.get("level_word"):
        return                      # the query named a level; respect it
    if len(ir.subjects) != 1:
        return
    subject = ir.subjects[0]
    if subject.type not in _ROLE_LEVELS:
        return                      # a team or an attribute is not a person

    # A QUESTION ABOUT A PERSON IS NOT AN ENUMERATION.
    #
    # "connects of Naina" asks for HER figure. The model routinely returns
    # target_level="advisor" with subject_of="unit_head" for it — a
    # hierarchy read — and the answer then comes back as the eleven people
    # under her, each with their own connects. Her own number is absent
    # from a reply that looks entirely reasonable.
    #
    # A read enumerates a level, and this sentence names none: the same
    # test that gates the promotion below, on the same evidence
    # (`level_word`), settles this too. "advisors under Naina" names one
    # and keeps its read; "connects of Naina as an advisor" names one and
    # never reaches here at all.
    if ir.target_level is not None:
        routing.decide(
            "Level", "not a hierarchy read",
            f"the query names {subject.value!r} and no level to enumerate, so "
            f"it asks for that person's own figure — the {ir.target_level!r} "
            "reading would answer with the people beneath them instead",
        )
        ir.target_level = None
        ir.subject_of = None
        ir.relation = "subtree"

    highest = hierarchy_grounding.highest_level_of(subject.value, db)
    if highest is None or highest == subject.type:
        return
    if _ROLE_LEVELS.index(highest) >= _ROLE_LEVELS.index(subject.type):
        return                      # nothing senior to promote to

    routing.decide(
        "Level", f"read {highest!r} not {subject.type!r}",
        f"the query names {subject.value!r} and no level, and this person's "
        f"own hierarchy puts them at {highest!r} — the senior role is who "
        f"they are, and the {subject.type!r} reading is a scope of a "
        "different size for the same name",
    )
    # BOTH fields, together. `subjects[].type` says what the subject IS
    # and `subject_level` says where the answer is reported; moving one
    # without the other is how a query comes to filter at one level and
    # group at another.
    was = subject.type
    subject.type = highest
    subject.resolved_id = subject.value
    subject.resolved_wid = None     # a manager is addressed by name, not a wid
    if ir.subject_level == was:
        ir.subject_level = highest


def interpret(text: str, entities: dict, db: Session,
              session_id: str | None = None) -> Interpretation:
    """THE mandatory first semantic step. Always calls the LLM.

    PHASE 2. Nothing decides whether this runs. The previous design asked
    `nlu_pipeline._plan_is_authoritative()` first — a rule-based
    classifier that answered "does this query need the model?" — and for
    fourteen of the twenty-one operations the answer was permanently no,
    so the model was never consulted for a roster, a profile, a manager
    lookup or an ancestry walk. Whether the LLM saw a query depended on a
    regex scorer's opinion of it.

    Grounding still runs BEFORE this, and deliberately: `entities` is
    handed to the prompt so the model reads real team and person names
    rather than guessing them. That is not a semantic decision — it
    establishes what exists, and the model decides what the user meant
    about it.

    Returns an Interpretation either way. A provider that is unreachable
    yields `reached_llm=False` and no IR, which the caller degrades from —
    an outage must not become a dead end.
    """
    from app.llm.semantic_model import from_query_ir

    # THE ROLLBACK SURVIVES. NLU_MODE="rules_first" is the documented
    # switch back to the pre-inversion routing, and a migration this size
    # should not remove the way out of it. Under that mode the mandatory
    # call is skipped and the legacy path below runs verbatim; under the
    # default "llm_first" nothing can skip it.
    if settings.nlu_mode == "rules_first":
        return Interpretation(None, None, reached_llm=False,
                              reason='NLU_MODE="rules_first" — the rollback path')

    ir = _call_llm_for_ir(text, entities, db, session_id)
    if ir is None:
        # Deliberately one reason for two causes: _call_llm_for_ir
        # collapses "unreachable/refused" and "output failed the schema"
        # into the same None. Both mean the same thing HERE — no semantic
        # model — and the distinction is already recorded on the request
        # trace by record_llm_parse, which is where it is actionable.
        return Interpretation(None, None, reached_llm=False,
                              reason="the model produced nothing usable "
                                     "(unreachable, refused, or schema-invalid)")

    # PHASE 11 — the model's interpretation is not rewritten here.
    #
    # validate_ir still grounds entities, resolves metrics, computes
    # `missing` and records confidence. What it no longer does on this
    # path is change the MEANING: retype a subject from pre-LLM
    # extraction, or null the target_level/subject_of/relation triple.
    # Those were deterministic code overruling the model with no signal
    # in the reply that it had happened. Conflicts now surface through
    # grounding (TYPE_MISMATCH) and validation, which reject or ask.
    result = validate_ir(ir, db, entities=entities, allow_semantic_repair=False)
    _resolve_unstated_person_level(result.ir, entities, db)
    result.ir.nlu_mode = settings.nlu_mode
    model = from_query_ir(result.ir, level_word=entities.get("level_word"))

    # PHASE 4 — GROUND WHAT THE MODEL NAMED.
    #
    #     semantic model -> grounding -> grounded entities -> validation
    #
    # Verification only: grounding_mod.ground() returns a parallel report
    # and modifies neither the model nor the IR, so nothing downstream
    # executes differently for it yet. The structured ambiguity it
    # produces is what a later phase acts on — and having it recorded now
    # is what makes "the model named something that does not exist" a
    # visible state rather than an empty result set.
    grounded = grounding_mod.ground(model, db)
    if grounded:
        routing.decide(
            "Grounding",
            "resolved" if grounded.is_fully_grounded else "unresolved",
            "; ".join(
                f"{e.name} ({e.stated_level or 'no level stated'}) -> {e.status}"
                + (f" at {'/'.join(e.found_at)}" if e.status == grounding_mod.TYPE_MISMATCH else "")
                for e in grounded.entities
            ),
        )

    # PHASE 5 — verify the RELATIONSHIP, then judge the whole thing.
    #
    #     grounded entities -> hierarchy grounding -> validation
    #
    # Both are reports. Neither edits the model or the IR: validation has
    # no repair path by design, so an interpretation that cannot run as
    # stated is rejected or sent for clarification rather than quietly
    # turned into a different query that happens to work.
    hierarchy_result = hierarchy_grounding.verify(model, grounded, db)
    if hierarchy_result.is_hierarchy:
        routing.decide(
            "Hierarchy", hierarchy_result.status,
            f"{hierarchy_result.subject_value} ({hierarchy_result.subject_level}) "
            f"-> {hierarchy_result.target_level} [{hierarchy_result.relation}]: "
            f"{hierarchy_result.member_count} found"
            + (f" — {hierarchy_result.reason}" if hierarchy_result.reason else ""))

    verdict = semantic_validation.validate(model, grounded, hierarchy_result, db,
                                           entities=entities)
    if not verdict.is_valid:
        routing.decide(
            "Validation", verdict.status,
            "; ".join(f"[{f.check}] {f.message}" for f in verdict.findings))

    return Interpretation(model, result.ir, reached_llm=True, grounding=grounded,
                          hierarchy=hierarchy_result, verdict=verdict)


def parse(text: str, entities: dict, db: Session, session_id: str | None,
          plan: QueryPlan | None = None) -> ParseOutcome:
    """Phase 4: `plan` is supplied by the caller.

    nlu_pipeline.resolve() has already planned this message — it routes on
    plan.action — and this function then planned it AGAIN from the same
    text and the same entities, so build_query_plan ran twice per request
    for an identical result. Planning is deterministic and pure, so the
    duplicate was invisible in behaviour and paid for twice in work.

    The parameter is optional so the module stays independently callable
    (its own tests do), but the pipeline always passes the plan it
    already made. One message, one planner run.
    """
    if plan is None:
        plan = build_query_plan(text, entities)

    # ---- llm_first (P1 inversion): LLM parses everything reaching here ----
    if settings.nlu_mode == "llm_first":
        ir = _call_llm_for_ir(text, entities, db, session_id)
        if ir is not None:
            return _finish(ir, db, session_id, used_llm=True, entities=entities)
        # P0 SAFETY: do not degrade a question the plan cannot hold.
        #
        # plan_to_ir builds from ONE metric and a flat conjunction, so
        # degrading a multi-measure or exclusion query produces a
        # well-formed IR for a DIFFERENT, narrower question — and the
        # reply is indistinguishable from a correct one. Returning no IR
        # hands the decision to nlu_pipeline, which says so plainly
        # rather than answering.
        from app.llm.nlu_pipeline import _semantic_gaps

        if _semantic_gaps(text, entities, plan):
            return ParseOutcome(ir=None, missing=["understanding"], used_llm=False)
        degraded = _rule_based_ir(text, entities, plan)
        if degraded is not None:
            return _finish(degraded, db, session_id, used_llm=False, entities=entities)
        return ParseOutcome(ir=None, missing=["intent"], used_llm=False)

    # ---- rules_first: pre-inversion behavior, kept as the rollback path ----
    compound = looks_compound(text, entities)

    # 1. rule-based fast path (skips the LLM for the common case).
    #
    # A comparison is exempt from the compound gate, as it was on the
    # plan path (nlu_pipeline._COMPOUND_EXEMPT_ACTIONS). looks_compound()
    # fires on the word "compare" itself AND on two entities at one
    # level, so EVERY comparison looks compound — the gate exists because
    # the rule planner once could not express multi-entity queries, which
    # stopped being true when comparison_targets() started grounding both
    # sides. Without the exemption every comparison would take an LLM
    # round trip to arrive at the IR the rule planner already had.
    if plan.action in _IR_ACTIONS and (plan.action == "comparison" or not compound):
        return _finish(plan_to_ir(plan, entities), db, session_id, used_llm=False,
                       entities=entities)

    # 2. widen via fuzzy metric match before reaching for the LLM
    if plan.action == "unresolved" and not compound:
        degraded = _rule_based_ir(text, entities, plan)
        if degraded is not None:
            return _finish(degraded, db, session_id, used_llm=False, entities=entities)

    # 3. LLM semantic parser — compound query, or nothing else worked
    ir = _call_llm_for_ir(text, entities, db, session_id)

    # 4. fail-soft degrade
    if ir is None:
        if plan.action not in _IR_ACTIONS:
            return ParseOutcome(ir=None, missing=["intent"], used_llm=False)
        ir = plan_to_ir(plan, entities)

    return _finish(ir, db, session_id, used_llm=True, entities=entities)
