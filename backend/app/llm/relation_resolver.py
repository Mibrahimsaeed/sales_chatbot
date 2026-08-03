"""
Relationship resolution (M1 of the Relationship Inference Engine).

Given a RESOLVED source identity and a target level, returns the related
value — the second half of "Waqar Haider's team", once the parser has
established that a relationship was referred to and identity resolution
has established who the source is.

TWO RULES THAT ARE NOT NEGOTIABLE:

1. NEVER INFER FROM AN AMBIGUOUS SOURCE. Risk R2 of the design. Eight
   different people are named "Yasir Ali" in production; inferring "Yasir
   Ali's team" from whichever candidate sorted first would produce a
   confident answer about the wrong person's team — the exact failure
   class the identity refactor exists to prevent. An ambiguous source
   yields nothing here, and the pipeline's existing "ask which one"
   behaviour is left to run.

2. M1 RESOLVES ONLY CACHED RELATIONS. `team` and `company` are already
   on every AdvisorIdentity, so they cost nothing. Every other declared
   relation (`bm`, `zm`, `office`, ...) is marked cached=False in the M0
   declarations and is NOT resolved here — reading those is milestone M3,
   which owns the decision about widening the identity cache projection
   versus paying a query per reference. Asking for one returns None
   rather than silently issuing a database read this milestone never
   budgeted for.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.llm import advisor_resolver, relations

# Provenance marker written into the entity dict. Format:
#   inferred:<source_level>:<source_id>
# so a reader can tell not just THAT a value was inferred but from whom.
PROVENANCE_PREFIX = "inferred"


@dataclass(frozen=True)
class ResolvedRelation:
    target_level: str
    value: str
    source_level: str
    source_id: int
    provenance: str
    confidence: float = 1.0


def _provenance(source_level: str, source_id: int) -> str:
    return f"{PROVENANCE_PREFIX}:{source_level}:{source_id}"


def resolve_from_identity(identity, target_level: str) -> ResolvedRelation | None:
    """The related value for one resolved advisor, or None.

    None is returned — never a guess — when the relation isn't declared,
    isn't cached (M3), or the source row simply has no value for it (an
    advisor with no team on file). "I don't have that" is a legitimate
    outcome the caller must handle; fabricating a value is not.
    """
    if identity is None or not target_level:
        return None

    spec = relations.registry.resolve("advisor", target_level)
    if spec is None or not spec.cached:
        return None

    # For cached relations the AdvisorIdentity attribute is named for the
    # level (team -> .team, company -> .company). Asserted by
    # test_relation_resolver.py so a future cached relation whose names
    # diverge fails loudly here instead of silently resolving to None.
    value = getattr(identity, target_level, None)
    if not value:
        return None

    return ResolvedRelation(
        target_level=target_level,
        value=value,
        source_level="advisor",
        source_id=identity.wid,
        provenance=_provenance("advisor", identity.wid),
        confidence=getattr(identity, "score", 1.0),
    )


def resolve_from_resolution(resolution, target_level: str) -> ResolvedRelation | None:
    """Rule 1 enforced at the boundary: only a RESOLVED advisor — exactly
    one real person — can be a source. AMBIGUOUS and NOT_FOUND yield
    nothing."""
    if resolution is None or getattr(resolution, "status", None) != advisor_resolver.RESOLVED:
        return None
    return resolve_from_identity(resolution.identity, target_level)
