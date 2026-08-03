"""
Cross-turn reference resolution (M4 of the Relationship Inference Engine).

Resolves a relationship whose SOURCE is not in the current message — "how
is his team doing" after "tell me about Waqar Haider". The relationship
half is unchanged: the same registry, the same relation_resolver, the
same entity keys. Only where the source comes from is new.

WHY IT IS A SEPARATE MODULE. M1's inference lives inside
entity_extractor, which never receives a session_id and therefore cannot
consult conversation memory — a fact the pre-M4 characterisation tests
pin. Rather than widen that boundary, cross-turn resolution runs where
cross-turn state already lives (nlu_pipeline, beside the existing
resolved-advisor carry) and calls the engine from there. The engine is
reused, not modified.

THE GATE. Requirements 1-5 collapse into one rule, checked before
anything else happens:

    fire only when the message names NO advisor of its own,
    AND contains a pronoun reference,
    AND memory holds exactly one resolved advisor.

Each clause carries its weight. "Names no advisor" makes an explicit name
win automatically and means a message naming TWO people is skipped, since
either would be an arbitrary choice. "Contains a pronoun reference" is
what confines this to genuine follow-ups: a topic change ("top 5 by
revenue") has no pronoun and so cannot inherit a subject. "Memory holds
one" is satisfied only by a settled identity — an unanswered "which Yasir
Ali?" never writes one, so a pronoun after an ambiguity resolves nothing
instead of guessing.

The identity is read from advisor_resolver's cache BY WID rather than
from a copy stored in memory, so a remembered subject's team is as fresh
as a just-named one's.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import get_logger
from app.llm import advisor_resolver, conversation_memory, hierarchy
from app.llm import reference_parser, relation_resolver

log = get_logger("llm.cross_turn_resolver")


def bind_relation(entities: dict, related, provenance_key: str) -> None:
    """Add one resolved relation to the entity dict.

    Shared by BOTH inference paths — entity_extractor's in-message
    inference and this module's cross-turn inference — because the two
    must produce byte-identical entity shapes. A second writer that
    drifted by one key would make a follow-up behave subtly differently
    from the query that established the subject, which is exactly the
    kind of inconsistency this programme exists to remove.

    M6 made this APPEND rather than overwrite. A comparison names two
    entities at the same level ("compare X's team with Downtown"), and
    the plural key is how the planner sees a pair — writing a one-element
    list threw one side away and left a two-sided question with a single
    target. Appending is also what makes two possessive references at the
    same level work at all.

    Precedence (requirement 6) is expressed through the SINGULAR key: it
    is set only if absent, so an entity the user named outright stays the
    primary reading and an inferred one joins it rather than displacing
    it. Duplicates are dropped, so binding the same value twice — two
    references to people on one team — leaves one target, not a
    comparison of a team with itself.

    `provenance_key` is the entity dict's private provenance key, passed
    in rather than imported to keep this module free of a dependency on
    entity_extractor (which imports this one).
    """
    target = related.target_level
    plural = hierarchy.LEVEL_ENTITY_KEYS.get(target, f"{target}s")
    matches_key = f"{target}_matches"

    values = list(entities.get(plural) or [])
    if any(existing.lower() == related.value.lower() for existing in values):
        return

    # Captured BEFORE mutating: a key that already held a gazetteer hit
    # was written by the user's own words, and appending to it must not
    # relabel it as inferred. Provenance is per KEY, so the honest rule is
    # to claim only the keys this bind brings into existence.
    pre_existing = {key for key in (matches_key, plural, target) if key in entities}

    entities[matches_key] = list(entities.get(matches_key) or []) + [
        {"value": related.value, "score": related.confidence}
    ]
    entities[plural] = values + [related.value]
    # Only fill the singular when nothing claimed it — an explicitly
    # named entity keeps the primary slot.
    entities.setdefault(target, related.value)

    marks = entities.setdefault(provenance_key, {})
    for key in (matches_key, plural, target):
        if key in pre_existing:
            continue
        marks.setdefault(key, related.provenance)


def _enabled_levels() -> frozenset[str]:
    if not settings.relation_inference_enabled:
        return frozenset()
    return settings.relation_inference_level_set


def names_an_advisor(entities: dict) -> bool:
    """Did THIS message identify a person? Any number of candidates
    counts — one means the user said who they meant, several mean the
    pipeline must ask rather than inherit a different subject entirely."""
    return bool(entities.get("advisor_wids"))


def resolve(text: str, entities: dict, db: Session, session_id: str | None,
            provenance_key: str) -> list[str]:
    """Infer relations for the remembered advisor, in place.

    Returns the levels that were bound, for logging and tests. Empty
    whenever the gate above is not fully satisfied — which is the common
    case and must stay cheap.
    """
    levels = _enabled_levels()
    if not levels:
        return []

    # Requirement 1/2/3: an explicit name in this message always wins,
    # and several names mean we must not choose one.
    if names_an_advisor(entities):
        return []

    # Requirement 5: only a pronoun follow-up inherits a subject.
    references = [
        reference for reference in reference_parser.parse_pronoun(text)
        if reference.target_level in levels
    ]
    if not references:
        return []

    # Requirement 1/4: a settled identity, or nothing.
    remembered = conversation_memory.get_resolved_advisor(session_id)
    if remembered is None:
        return []

    wid, _name = remembered
    identity = advisor_resolver.identity_for_wid(wid, db)
    if identity is None:
        # The remembered person is no longer in the master sheet. Saying
        # nothing is right: inventing a subject would be worse than
        # answering as though no follow-up context existed.
        return []

    bound: list[str] = []
    for reference in references:
        target = reference.target_level
        # An entity named outright in this message still wins.
        if entities.get(target):
            continue
        related = relation_resolver.resolve_from_identity(identity, target)
        if related is None:
            continue
        bind_relation(entities, related, provenance_key)
        bound.append(target)

    if bound:
        log.debug("cross-turn inference from wid=%s bound %s", wid, bound)
    return bound
