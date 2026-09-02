"""Hierarchy grounding: is the relationship the model asked for real?

    semantic model -> entity grounding -> HIERARCHY GROUNDING -> validation

Entity grounding answers "does this name exist, and as what". This layer
answers the question that only makes sense afterwards: "the model asked
for TEAMS under this Unit Head — are there any, and is that a
relationship this data can express at all?"

VERIFIED AGAINST THE DATA, NOT AGAINST A DECLARED ORDER. Every level is a
column on the denormalised advisor row, so a relationship is checked by
running it: filter the rows the subject scopes and read the distinct
values at the requested level. That is `hierarchy.scope_filter`, which
already states this once for filtering, breakdown, comparison and
aggregation, so grounding cannot disagree with execution about who is in
scope.

That choice matters here for a second reason. The declared chain's
placement of `team` is contested — production containment puts
`team ⊂ unit_head` at 98.0% and the declared parent edge at 13.6% — and
NOTHING IN THIS MODULE READS THAT ORDERING. A traversal is confirmed by
the rows it returns, so this layer stays correct however that question is
eventually settled.

WHAT IT REFUSES TO DO. It never invents a traversal: "connects of Blue
Area" names a team and asks for its figure, and a team having members is
not a reason to enumerate them. It never repairs a relationship into a
different one that would have worked. An unsupported or empty
relationship is REPORTED — the pipeline decides whether that is a
clarification or an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.llm import grounding as entity_grounding
from app.llm import hierarchy
from app.llm.semantic_model import SemanticModel

# The relationship was checked and the data supports it.
VERIFIED = "verified"
# Well-formed, but nobody is there. A true fact about the organisation,
# not a failure of the parse — "no BCMs work under her" is an answer.
EMPTY = "empty"
# The data cannot express this relationship at all: an attribute has
# nothing beneath it, and nothing sits beneath the leaf.
UNSUPPORTED = "unsupported"
# Not a hierarchy question. The default, and the one "connects of Blue
# Area" must keep.
NOT_APPLICABLE = "not_applicable"


@dataclass
class HierarchyGrounding:
    """What the data says about the requested relationship."""
    status: str = NOT_APPLICABLE
    subject_value: str | None = None
    subject_level: str | None = None
    target_level: str | None = None
    relation: str = "subtree"
    # Actual values found at the target level. Capped for reporting; the
    # count is the honest size.
    members: list[str] = field(default_factory=list)
    member_count: int = 0
    reason: str | None = None

    @property
    def is_hierarchy(self) -> bool:
        return self.status != NOT_APPLICABLE

    @property
    def is_verified(self) -> bool:
        return self.status == VERIFIED

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "subject": self.subject_value,
            "subject_level": self.subject_level,
            "target_level": self.target_level,
            "relation": self.relation,
            "member_count": self.member_count,
            "members": self.members[:20],
            "reason": self.reason,
        }


# How many member values to keep for reporting. The count is exact; this
# only bounds what is carried around for tracing and clarification text.
_MEMBER_SAMPLE = 50


def highest_level_of(name: str, db: Session) -> str | None:
    """The senior-most role a person holds.

    "Asked about a person" must answer at their highest level: the same
    man is a Unit Head over 75 advisors and, incidentally, the BCM of the
    people who name him in `management_lead` — a scope of one. Answering
    from the junior role is a true statement about that column and a
    false statement about him.

    Grounding IS the relationship here: a name reaches `unit_head` only
    because some advisor's Unit Head column names them. So the levels the
    name grounds at already say which roles they hold.
    """
    # Ranked senior-first by the pipeline's own definition, so "which of
    # this person's roles governs the answer" is decided in one place.
    # `team` is excluded from that ranking, which is also why it is
    # unaffected by the open question about where `team` sits. Imported
    # lazily: nlu_pipeline imports semantic_parser, which imports this.
    from app.llm.nlu_pipeline import _ROLE_LEVELS, _highest_role

    levels = [
        level for level in _ROLE_LEVELS
        if entity_grounding._candidates_at(name, level, db)
    ]
    return _highest_role(levels) if levels else None


def _members_at(db: Session, subject_level: str, subject_value: str,
                target_level: str, relation: str) -> list[str]:
    """Run the relationship and return the distinct values it reaches.

    Delegates the member definition to hierarchy_service, which already
    decides what a member IS at each target (names at a manager level,
    advisor rows at the leaf) and applies the master-sheet filter.
    Imported lazily for the same reason query_compiler does: the service
    layer imports app.llm.
    """
    from app.services import hierarchy_service

    if relation == "direct":
        predicate = hierarchy.direct_scope_filter(subject_level, subject_value, target_level)
    else:
        predicate = hierarchy.scope_filter(subject_level, subject_value)

    members = hierarchy_service._members_for(db, predicate, target_level)
    return [m["name"] for m in members if m.get("name")]


def verify(model: SemanticModel, grounded: entity_grounding.Grounding,
           db: Session) -> HierarchyGrounding:
    """Check the relationship the interpretation asks for.

    NOT_APPLICABLE unless the model actually expressed one. The test is
    `SemanticModel.is_hierarchy_query()` — the model's own statement of
    meaning — never a guess from the subject's type. A team having
    members is not a request to list them.
    """
    if not model.is_hierarchy_query():
        return HierarchyGrounding(status=NOT_APPLICABLE)

    target = hierarchy.canonical_level(model.requested_level)
    relation = model.relationship.depth if model.relationship else "subtree"

    # The entity the traversal starts from. A hierarchy read puts it in
    # `scope`; only a resolved one can anchor a traversal, because an
    # ambiguous name would silently pick a subtree.
    anchors = [e for e in grounded.by_role(entity_grounding.SCOPE) if e.is_resolved]
    if not anchors:
        return HierarchyGrounding(
            status=UNSUPPORTED, target_level=target, relation=relation,
            reason="no resolved entity to traverse from")

    anchor = anchors[0]
    subject_level, subject_value = anchor.resolved.level, anchor.resolved.value

    if target is None:
        return HierarchyGrounding(
            status=UNSUPPORTED, subject_value=subject_value,
            subject_level=subject_level, relation=relation,
            reason="the interpretation asks for a relationship without naming a level")

    # ---- relationships the data cannot express ----------------------
    #
    # Both refusals are structural facts about the schema rather than
    # opinions about the chain's order, so neither depends on the open
    # question of where `team` sits.
    if not hierarchy.is_chain_level(target):
        return HierarchyGrounding(
            status=UNSUPPORTED, subject_value=subject_value,
            subject_level=subject_level, target_level=target, relation=relation,
            reason=f"'{target}' is an attribute, not a step in the chain — "
                   "nothing sits beneath it")
    if subject_level == "advisor":
        return HierarchyGrounding(
            status=UNSUPPORTED, subject_value=subject_value,
            subject_level=subject_level, target_level=target, relation=relation,
            reason="an advisor is the leaf — nothing reports to one")

    members = _members_at(db, subject_level, subject_value, target, relation)

    return HierarchyGrounding(
        status=VERIFIED if members else EMPTY,
        subject_value=subject_value,
        subject_level=subject_level,
        target_level=target,
        relation=relation,
        members=sorted(members)[:_MEMBER_SAMPLE],
        member_count=len(members),
        reason=None if members
        else f"no {hierarchy.label_for(target)} found beneath "
             f"{subject_value} ({hierarchy.label_for(subject_level)})",
    )
