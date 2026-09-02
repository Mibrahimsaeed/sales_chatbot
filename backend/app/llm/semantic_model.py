"""THE semantic contract between the LLM and the rest of the system.

WHAT THIS IS. One structure describing what the USER MEANT, before any
database has been consulted. It is the output of the LLM semantic parser
and the input to entity grounding — the first box in the strict LLM-first
flow:

    User Query + Conversation Context
            -> LLM
            -> SemanticModel          <- this file
            -> Entity & Hierarchy Grounding
            -> Validation
            -> executable QueryIR
            -> database

WHY IT IS NOT QueryIR. `query_ir.QueryIR` is the EXECUTION contract, and
it carries three kinds of field that have no business in a statement of
meaning:

  grounded identity   `Subject.resolved_id`, `Subject.resolved_wid` — the
                      answers grounding produces, not something a user
                      can mean.
  rendering choices   `flat`, and the legacy `intent` that duplicates
                      `operation` under an older vocabulary.
  observability       `nlu_mode`, `repairs`, `confidence_level` — records
                      of what the pipeline did, written after the fact.

Mixing them is what let meaning be edited to suit execution. This model
holds only what a person can actually have meant, so a later stage that
changes it is visibly changing the question rather than adjusting a
detail.

THE DISTINCTION THIS EXISTS TO PRESERVE. QueryIR has one `subjects` list
that means two different things depending on whether `target_level` is
set, and nothing states which reading applies:

    "connects of Blue Area"              Blue Area IS the answer
    "connects of advisors in Blue Area"  Blue Area RESTRICTS the answer

Both arrive as `subjects=[team Blue Area]`. Reading them apart requires
inspecting three other fields, and the pipeline repeatedly got it wrong
in both directions — reporting a team's own figure for a query about its
advisors, and listing 48 advisors for a query about the team.

Here they are separate fields, `subject` and `scope`, and they cannot be
confused. `requested_level` is set ONLY when the user actually named a
level to return; it is never inferred from the fact that a subject has
members underneath it.

WHAT THIS DOES NOT DO. It does not resolve anything. `EntityRef.name` is
the string the user said; whether such a team exists, and which one, is
grounding's question (Phase 4). Nothing here reads the database.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from app.llm import hierarchy, periods
from app.llm.query_ir import LEVEL_NAMES, Level, Operator

# Reused from the hierarchy registry and the period table rather than
# restated — a level or period added there is available here with no edit,
# which is the drift `query_ir.LEVEL_NAMES` was introduced to end.
Period = Literal[periods.PERIODS]

# HOW a subject relates to what is being asked for. `None` on the model
# means the user expressed no relationship at all, which is the ordinary
# case and must never be invented.
#
#   membership   "advisors IN Blue Area", "teams UNDER Faisal"
#   reports_to   "who reports to Faisal" — the same containment, asked as
#                a reporting question; kept distinct because the answer
#                names people rather than a group.
#   manager_of   "the unit head OF Blue Area" — the REVERSE direction:
#                the answer is above the named entity, not beneath it.
RelationshipKind = Literal["membership", "reports_to", "manager_of"]

# How far the relationship reaches. Only meaningful when `relationship`
# is set, and `direct` requires explicit wording ("directly", "immediate")
# — "under" alone is the whole subtree.
RelationshipDepth = Literal["direct", "subtree"]

# What KIND of answer the user asked for, in the registry's own
# vocabulary (app/llm/operations.py). Deliberately not a sixth parallel
# list of names: that file exists because five vocabularies for one set
# of concepts drifted, and a new one here would be the sixth.
SemanticOperation = str


class EntityRef(BaseModel):
    """A thing the user named, exactly as they named it.

    NOT RESOLVED. `name` is their words; grounding decides whether it
    exists and which record it is. `level` is the type the query implies,
    which may be wrong and which grounding may correct — that correction
    is a grounding decision, recorded there, not a silent edit here.
    """

    name: str
    # The level this entity is, when the query says or implies one.
    # `None` means the user named something without saying what it is
    # ("connects of Faisal") — a real state, and the one grounding exists
    # to settle.
    level: Optional[Level] = None
    # Did the user actually SAY the level ("Unit Head Faisal", "the team
    # Blue Area")? A stated level is evidence; an inferred one is a guess,
    # and later stages must be able to tell them apart before overruling.
    level_was_stated: bool = False
    # How sure the parser is that this is an entity reference at all.
    confidence: float = 1.0


class MetricRequest(BaseModel):
    """One measure the user asked for, in the order they asked.

    NO PRIMARY. QueryIR splits `metric` from `metrics[]`, which invites a
    reader to treat the first as the real one and the rest as decoration
    — and a multi-metric question then loses everything after the first.
    A list of equals is what the user actually said; which one ORDERS the
    result is `ordering`, a separate question.
    """

    name: str
    confidence: float = 1.0


class Condition(BaseModel):
    """A constraint the user stated: "connects above 1000", "in Blue Area"."""

    field: str
    operator: Operator = "="
    value: Optional[Union[str, float, int, list]] = None
    confidence: float = 1.0


class ConditionGroup(BaseModel):
    """Boolean structure a flat list cannot hold — OR and NOT.

    Same shape as `query_ir.FilterGroup` on purpose: this is the one part
    of the execution contract that already expresses meaning correctly,
    and inventing a second spelling of it would only create a mapping to
    get wrong.
    """

    op: Literal["and", "or", "not"] = "and"
    children: list[Union["ConditionGroup", Condition]] = Field(default_factory=list)

    def leaves(self) -> list[Condition]:
        found: list[Condition] = []
        for child in self.children:
            found.extend(child.leaves() if isinstance(child, ConditionGroup) else [child])
        return found


ConditionGroup.model_rebuild()


class Relationship(BaseModel):
    """An explicitly expressed hierarchy relationship.

    Present ONLY when the query contains relationship wording. The absence
    of this object is a positive statement: "connects of Blue Area"
    expresses no relationship, so no traversal may be invented for it,
    however many members Blue Area happens to contain.
    """

    kind: RelationshipKind
    depth: RelationshipDepth = "subtree"
    # The level the relationship is WITH, when the query names a ROLE
    # rather than a person: "advisors reporting to the Unit Head in AMD"
    # is a relationship with a `unit_head`, scoped to the team AMD.
    #
    # Without this the model could not express that question at all. The
    # scope (AMD) and the target (advisor) were both carried, and the
    # MANAGER — the whole subject of the sentence — was dropped, leaving
    # a reading indistinguishable from "advisors in AMD".
    #
    # None when the manager is named directly ("advisors under Haseeb"),
    # because the scope entity IS the manager and repeating its level
    # here would say nothing.
    of_level: Optional[Level] = None


class TimeRange(BaseModel):
    """The window the user asked for, and whether they asked at all.

    `stated` is the field QueryIR lacks. Its `time_range.period` defaults
    to MTD, so "the user said this month" and "the user said nothing and
    MTD is the fallback" are the same value — and a follow-up ("what
    about year to date") cannot tell whether it is replacing a stated
    window or filling an unstated one.
    """

    period: Period = "MTD"
    stated: bool = False
    # The window this one is compared AGAINST, when the user asked for a
    # comparison across time.
    compare_to: Optional[Period] = None
    confidence: float = 1.0


class Ordering(BaseModel):
    """How the user asked for the answer to be ordered.

    `stated` separates "rank by revenue descending" from a direction
    nobody asked for. Only a stated direction may override the measure's
    own polarity, which is why the flag is here and not inferred later.
    """

    metric: Optional[str] = None
    direction: Optional[Literal["asc", "desc"]] = None
    stated: bool = False


class ConversationReference(BaseModel):
    """What this turn takes from the conversation rather than saying.

    "what about last month" states a period and nothing else; every other
    field is inherited. Recording WHICH fields were carried is what makes
    a wrong follow-up answer diagnosable — otherwise the merged model is
    indistinguishable from one the user typed in full.
    """

    is_follow_up: bool = False
    # Field names on this model that came from the previous turn.
    inherited: list[str] = Field(default_factory=list)
    # The user's own referring words, when there were any ("his team",
    # "those advisors"), for the trace.
    referring_phrase: Optional[str] = None


class SemanticModel(BaseModel):
    """What the user meant. Nothing about how to execute it."""

    # ---- what kind of question ---------------------------------------
    operation: SemanticOperation
    # How sure the parser is about the SHAPE, independent of any field.
    operation_confidence: float = 1.0

    # ---- what is being measured --------------------------------------
    # Every measure named, in the user's order. Empty is meaningful: a
    # roster or a population asks WHO and names no measure.
    metrics: list[MetricRequest] = Field(default_factory=list)

    # ---- who the question is ABOUT -----------------------------------
    # The entity the answer describes. `None` when the question names no
    # subject ("top 5 advisors by revenue" is about everybody).
    subject: Optional[EntityRef] = None
    # The level the ANSWER is reported at. For an ordinary metric query
    # this is the subject's own level.
    subject_level: Optional[Level] = None

    # ---- what RESTRICTS it -------------------------------------------
    # Entities that narrow the answer without being it: "advisors in Blue
    # Area" is about advisors, scoped by Blue Area. Separate from
    # `subject` so the two readings of one named entity can never be
    # confused.
    scope: list[EntityRef] = Field(default_factory=list)
    # The level the user EXPLICITLY asked to see. Set only when they said
    # it. Never inferred from a subject having members.
    requested_level: Optional[Level] = None
    # Set only when the query expresses one. See Relationship.
    relationship: Optional[Relationship] = None

    # ---- constraints --------------------------------------------------
    conditions: list[Condition] = Field(default_factory=list)
    condition_tree: Optional[ConditionGroup] = None

    # ---- shape of the answer ------------------------------------------
    time_range: TimeRange = Field(default_factory=TimeRange)
    group_by: Optional[Level] = None
    ordering: Ordering = Field(default_factory=Ordering)
    limit: Optional[int] = None
    # Two or more entities set side by side. Distinct from `scope`: these
    # are all subjects, and none of them restricts the others.
    comparison_subjects: list[EntityRef] = Field(default_factory=list)

    # ---- what the parser could not settle ------------------------------
    # Named things the parser itself flagged as ambiguous, in the user's
    # words. Grounding may add more; this is what the LLM already knew.
    ambiguous: list[str] = Field(default_factory=list)
    # Slots the query left unspecified that this operation needs.
    missing: list[str] = Field(default_factory=list)
    conversation: ConversationReference = Field(default_factory=ConversationReference)

    @field_validator("operation")
    @classmethod
    def _known_operation(cls, value: str) -> str:
        """Validated against the registry at runtime rather than pinned as
        a Literal, so operations.py stays the single declaration and this
        model does not become a sixth copy of the vocabulary."""
        from app.llm import operations

        if value not in operations.OPERATIONS:
            raise ValueError(
                f"{value!r} is not an operation — see app/llm/operations.py"
            )
        return value

    # ---- accessors ----------------------------------------------------

    def is_hierarchy_query(self) -> bool:
        """Did the user ask to traverse the hierarchy?

        TRUE ONLY ON EXPLICIT EVIDENCE: a stated relationship, or a
        requested level different from the subject's own. A subject that
        merely HAS members is not a traversal — which is the whole point
        of the model.
        """
        if self.relationship is not None:
            return True
        return bool(
            self.requested_level
            and self.subject is not None
            and self.requested_level != self.subject.level
        )

    def metric_names(self) -> list[str]:
        """Every measure named, in order, deduplicated."""
        seen: set[str] = set()
        return [m.name for m in self.metrics
                if not (m.name in seen or seen.add(m.name))]

    def all_conditions(self) -> list[Condition]:
        """Every condition, flat, tree included. The boolean shape is
        deliberately dropped — callers asking WHICH fields are constrained
        want the set; only the compiler needs the structure."""
        found = list(self.conditions)
        if self.condition_tree is not None:
            found.extend(self.condition_tree.leaves())
        return found

    def entities(self) -> list[EntityRef]:
        """Every entity the user named, whatever role it plays."""
        named = ([self.subject] if self.subject else []) + list(self.scope)
        return named + list(self.comparison_subjects)


# =====================================================================
# Compatibility with the existing execution contract
# =====================================================================
#
# Phase 1 only has to prove the model can HOLD what the system already
# produces. The forward direction — a grounded, validated SemanticModel
# becoming an executable QueryIR — is Phase 6's, and needs the resolved
# identifiers this model deliberately does not carry.


def from_query_ir(ir, *, level_word: Optional[str] = None) -> SemanticModel:
    """Read an existing QueryIR as a statement of meaning.

    The one judgement here is the one QueryIR cannot express: whether its
    `subjects[]` are the ANSWER or a RESTRICTION. Three signals settle it,
    in order — an explicit hierarchy read, a subject reported at a level
    other than its own, and the level word the user actually said
    (`entity_extractor`'s `level_word`, passed in because it is the
    user's wording and not part of the IR).
    """
    from app.llm import operations

    op = ir.resolved_operation()
    if op not in operations.OPERATIONS:
        op = "unresolved"

    subjects = [
        EntityRef(name=s.value, level=s.type, confidence=s.match_confidence)
        for s in ir.subjects
    ]

    is_read = bool(ir.target_level) and bool(ir.subject_of or ir.subjects)
    scoped = is_read or (
        bool(subjects) and subjects[0].level != ir.subject_level
    ) or bool(level_word and subjects and level_word != subjects[0].level)

    model = SemanticModel(
        operation=op,
        operation_confidence=ir.intent_confidence,
        metrics=[MetricRequest(name=k) for k in ir.metric_keys()],
        subject_level=ir.subject_level,
        conditions=[
            Condition(field=f.field, operator=f.operator, value=f.value,
                      confidence=f.confidence)
            for f in ir.filters
        ],
        time_range=TimeRange(
            period=ir.time_range.period,
            compare_to=ir.compare_period(),
            confidence=ir.time_range.confidence,
        ),
        group_by=ir.group_by,
        ordering=Ordering(
            metric=ir.sort.metric if ir.sort else None,
            direction=ir.sort.direction if (ir.sort and ir.sort.metric) else None,
            stated=bool(ir.sort and ir.sort.metric),
        ),
        limit=ir.limit,
        missing=list(ir.missing),
        ambiguous=list(ir.ambiguity_reasons),
    )

    if op == "comparison" and len(subjects) >= 2:
        model.comparison_subjects = subjects
    elif scoped:
        model.scope = subjects
        model.requested_level = ir.target_level or (level_word or None) or ir.subject_level
        if is_read:
            # `subject_of` names the level the target sits beneath. When it
            # differs from the scope's own level the query named a ROLE
            # inside a group — "the Unit Head in AMD" — and that level is
            # the manager, not a restatement of the scope.
            scope_level = subjects[0].level if subjects else None
            of_level = ir.subject_of if ir.subject_of != scope_level else None
            model.relationship = Relationship(kind="membership", depth=ir.relation,
                                              of_level=of_level)
        elif level_word:
            model.relationship = Relationship(kind="membership", depth="subtree")
    elif subjects:
        model.subject = subjects[0]

    if ir.filter_tree is not None:
        model.condition_tree = _tree_from(ir.filter_tree)
    return model


def _tree_from(group) -> ConditionGroup:
    from app.llm.query_ir import FilterGroup

    return ConditionGroup(
        op=group.op,
        children=[
            _tree_from(child) if isinstance(child, FilterGroup)
            else Condition(field=child.field, operator=child.operator,
                           value=child.value, confidence=child.confidence)
            for child in group.children
        ],
    )
