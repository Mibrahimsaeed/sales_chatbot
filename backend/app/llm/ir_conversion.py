"""SemanticModel -> executable QueryIR.

    semantic model -> grounding -> validation -> EXECUTABLE QueryIR

Phase 1 deliberately shipped `from_query_ir` and no inverse: reading an
IR as meaning needs nothing but the IR, while WRITING one needs the two
things that did not exist yet — grounded identifiers and a verdict. This
module is that inverse, and it takes both.

WHAT "EXECUTABLE" MEANS HERE, precisely:

  1. IT IS VALIDATED. Conversion refuses to run on an interpretation that
     validation did not pass. Not a style rule — a VALID verdict is what
     guarantees every named entity resolved, so it is also what
     guarantees the identifier fields below can be filled at all.

  2. IT CARRIES IDENTIFIERS, NOT WORDS. `Subject.value` gets the
     CANONICAL database value, never the user's phrasing: "blue aera"
     resolved to the team "Blue Area" enters the IR as "Blue Area".
     Advisors additionally carry `resolved_wid`, because a name cannot
     address one person — 238 name groups in production map to more than
     one human, so matching by name sums several people into one row.
     Every other level has no identifier of its own in this schema, so
     its canonical value IS the identifier, and it is written to
     `resolved_id` to say so.

  3. ITS HIERARCHY SCOPE IS THE VERIFIED ONE. `target_level`,
     `subject_of` and `relation` are taken from the hierarchy grounding
     report — the traversal that was actually run against the data — not
     from the model's request. They are the same values whenever
     verification succeeded, and verification is the only path here.

NOTHING IS REPAIRED ON THE WAY THROUGH. This module has no fallbacks: it
does not guess a metric, invent a level, or drop an entity it could not
place. Everything it emits was established upstream, which is what keeps
"the query was wrong" and "the conversion was wrong" distinguishable.
"""

from __future__ import annotations

from app.llm import grounding as entity_grounding
from app.llm import hierarchy_grounding, operations
from app.llm.query_ir import (
    Filter, FilterGroup, MetricRef, QueryIR, Sort, Subject, TimeRange,
)
from app.llm.semantic_model import ConditionGroup, SemanticModel
from app.llm.semantic_validation import Verdict


class NotValidated(ValueError):
    """Raised when conversion is attempted on an unvalidated interpretation.

    A caller error rather than a user one: `Verdict.is_executable` is the
    gate, and reaching here without checking it means the pipeline skipped
    a step. Distinct from returning None, which is a real and expected
    state — see to_query_ir.
    """


def _subject_from(entity: entity_grounding.GroundedEntity) -> Subject:
    """One grounded entity as an executable subject.

    Reads `entity.resolved`, which is None for anything that is not
    exactly one record — so an ambiguous or missing entity cannot
    silently contribute a subject here even if the gate above were
    removed.
    """
    record = entity.resolved
    is_advisor = record.level == "advisor"
    return Subject(
        type=record.level,
        # THE CANONICAL VALUE, not what the user typed.
        value=record.value,
        # Managers and groups have no separate identifier in this schema;
        # their canonical value is what the SQL layer matches on, so it
        # is recorded as the resolved id rather than left to `value`.
        resolved_id=None if is_advisor else record.value,
        resolved_wid=record.wid if is_advisor else None,
        match_confidence=record.score,
    )


def _tree_to(group: ConditionGroup) -> FilterGroup:
    """ConditionGroup -> FilterGroup. Same shape by design, so this is a
    rename rather than a translation."""
    return FilterGroup(
        op=group.op,
        children=[
            _tree_to(child) if isinstance(child, ConditionGroup)
            else Filter(field=child.field, operator=child.operator,
                        value=child.value, confidence=child.confidence)
            for child in group.children
        ],
    )


def _subject_entities(model: SemanticModel,
                      grounded: entity_grounding.Grounding) -> list[entity_grounding.GroundedEntity]:
    """Which grounded entities become `subjects[]`.

    Mirrors from_query_ir's split so a round trip is stable: a comparison
    carries its targets, everything else carries the subject and any
    scope. Both restrict the query — the compiler applies `subjects[]` as
    a scope filter for every intent — so a metric query scoped to a team
    keeps both without double-counting, because the model never populates
    `subject` and `scope` with the same entity.
    """
    if model.operation == "comparison":
        return [e for e in grounded.by_role(entity_grounding.COMPARISON) if e.is_resolved]
    return [e for e in grounded.entities
            if e.role in (entity_grounding.SUBJECT, entity_grounding.SCOPE) and e.is_resolved]


def to_query_ir(model: SemanticModel,
                grounded: entity_grounding.Grounding,
                hierarchy_result: hierarchy_grounding.HierarchyGrounding,
                verdict: Verdict,
                *, principal=None) -> QueryIR | None:
    """The validated interpretation as an executable QueryIR.

    Returns None when the operation has no IR representation at all — a
    profile card and a roster are answered from the plan, and no IR
    expresses their shape. That is an expected state, not a failure, and
    it is reported as None rather than as an exception so a caller can
    fall through to the plan path.

    Raises NotValidated when the verdict is not executable.
    """
    if not verdict.is_executable:
        raise NotValidated(
            f"refusing to convert a {verdict.status} interpretation: "
            + "; ".join(f.message for f in verdict.findings))

    operation = operations.OPERATIONS.get(model.operation)
    if operation is None or not operation.expressible_in_ir:
        return None

    # AN OPERATION NEED NOT OWN AN INTENT TO BE EXPRESSIBLE. `ir_intent`
    # is the REVERSE mapping — intent back to operation — so only one
    # operation may claim each intent. `group_metric` and `population`
    # therefore declare None while compiling perfectly well through
    # `filtered_list`: the validator reaches group_metric by RE-LABELLING
    # a filtered_list IR, and every few-shot for both carries
    # intent="filtered_list". Read from the operation that does own that
    # intent, so this cannot drift from the registry.
    intent = operation.ir_intent or operations.OPERATIONS["filtered_list"].ir_intent

    metric_names = [m.name for m in model.metrics]
    ordering = model.ordering
    subject_entities = _subject_entities(model, grounded)

    # WHERE THE ANSWER IS REPORTED. SemanticModel states the default in
    # its own field docs — "for an ordinary metric query this is the
    # subject's own level" — and getting it wrong is not a cosmetic
    # difference: falling back to "advisor" makes "connects of Blue Area"
    # group by advisor inside the team and answer with 48 rows instead of
    # the team's one figure. That is the exact defect the group_metric
    # re-label exists to prevent, reintroduced from the other end.
    subject_level = model.subject_level
    if subject_level is None and subject_entities:
        subject_level = subject_entities[0].resolved.level

    ir = QueryIR(
        operation=operation.name,
        intent=intent,
        subject_level=subject_level or "advisor",
        subjects=[_subject_from(e) for e in subject_entities],
        metric=MetricRef(key=metric_names[0]) if metric_names else None,
        metrics=[MetricRef(key=name) for name in metric_names],
        filters=[
            Filter(field=c.field, operator=c.operator, value=c.value,
                   confidence=c.confidence)
            for c in model.conditions
        ],
        filter_tree=_tree_to(model.condition_tree) if model.condition_tree else None,
        time_range=TimeRange(
            # A period comparison is a MODE, and the model states it by
            # naming a second window rather than by setting a flag.
            mode="compare" if model.time_range.compare_to else "snapshot",
            period=model.time_range.period,
            compare_to=model.time_range.compare_to,
            confidence=model.time_range.confidence,
        ),
        sort=Sort(
            metric=ordering.metric,
            # The measure's own polarity stands unless the user stated a
            # direction; "desc" is QueryIR's own default and is what an
            # unstated ordering has always compiled to.
            direction=ordering.direction or "desc",
        ),
        limit=model.limit,
        group_by=model.group_by,
        overall_confidence=model.operation_confidence,
        intent_confidence=model.operation_confidence,
    )

    # ---- the VERIFIED hierarchy scope -------------------------------
    #
    # Taken from the traversal that was actually run, never from the
    # request. Left at the defaults for a non-hierarchy query, which is
    # what keeps "connects of Blue Area" an ordinary metric query rather
    # than an enumeration of the team.
    if hierarchy_result.is_hierarchy and hierarchy_result.target_level:
        ir.target_level = hierarchy_result.target_level
        ir.subject_of = hierarchy_result.subject_level
        ir.relation = hierarchy_result.relation

    # ---- authorization scope ----------------------------------------
    #
    # Carried, not enforced. There is no authorization policy in this
    # system — tokens hold a `role` claim that nothing consumes, and the
    # posture is an unsettled decision — so inventing a scope here would
    # silently shrink results with no stated reason. The field records
    # the principal the query was built for and nothing reads it yet,
    # which is what makes adding a policy a change in one place.
    ir.authorization_scope = dict(principal) if principal else None

    return ir
