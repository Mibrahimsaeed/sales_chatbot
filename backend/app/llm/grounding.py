"""Entity grounding: does what the LLM named actually exist?

    LLM Semantic Model -> GROUNDING -> grounded entities -> validation

The division of labour this module exists to enforce:

    the LLM decides WHAT the user named and WHAT TYPE it is;
    this module decides WHETHER THAT EXISTS and WHICH RECORD it is.

GROUNDING VERIFIES. IT DOES NOT REINTERPRET. If the model says "Blue
Area" is a team, the question asked here is "is there a team called Blue
Area" — not "what is the best reading of the string 'Blue Area'". A name
that also matches an advisor does NOT turn a stated team into an advisor.
That silent retype is the specific failure this contract forbids: it
answers a question the user did not ask, and it looks like a correct
answer because a real record comes back.

When the stated type does not exist, the answer is TYPE_MISMATCH with the
levels the name WAS found at — reported, never applied. Choosing what to
do about it is the pipeline's decision, not grounding's.

WHY A NEW MODULE RATHER THAN entity_extractor. That module answers a
different question: given raw TEXT, which known values appear in it. It
scans the sentence against every gazetteer because before the LLM there
was nothing else to go on. It stays exactly as it is — it still feeds the
prompt's grounded-entity hints, and the rule planner still depends on it.
This module starts from names the model already identified, so it never
guesses at spans and never has to decide what part of a sentence is an
entity.

NOTHING HERE PICKS A WINNER. Several matches produce AMBIGUOUS with every
candidate kept, and `resolved` is None for anything that is not exactly
one record — the same structural guarantee ResolvedAdvisor makes, for the
same reason: a caller must not be able to read one entity out of a result
that names several.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy.orm import Session

from app.llm import advisor_resolver, entity_extractor, fuzzy_match, hierarchy, relations
from app.llm.semantic_model import EntityRef, SemanticModel

# The four outcomes. The first three mirror advisor_resolver's statuses
# deliberately — a caller that already handles a ResolvedAdvisor handles
# these — and TYPE_MISMATCH is the one this layer adds, because it is the
# state that only exists once something else has asserted a type.
RESOLVED = "resolved"
AMBIGUOUS = "ambiguous"
NOT_FOUND = "not_found"
TYPE_MISMATCH = "type_mismatch"

# Where an entity sat in the sentence. Kept so the pipeline can treat a
# failure differently by role: an unresolvable SUBJECT means the question
# cannot be answered, while an unresolvable SCOPE may just be a filter to
# drop or ask about.
SUBJECT = "subject"
SCOPE = "scope"
COMPARISON = "comparison"

# Reuse the resolver's margin rather than choosing a second one: "these
# two are too close to call" is one judgement and belongs in one place.
AMBIGUITY_MARGIN = advisor_resolver.AMBIGUITY_MARGIN


def _level_aliases() -> dict[str, str]:
    """Level synonyms the user's vocabulary already supports.

    "Management Lead" and "BCM" are the same level (both bind
    Advisor.management_lead), as are "Portfolio Lead"/"Zonal Head" and
    "Regional"/"RM"/"Unit Head". Read from relations.role_alias_pairs(),
    which already owns this mapping for the rule planner, so a new alias
    registered there is understood here without a second edit.
    """
    aliases = {word.replace(" ", "_"): level for word, level in relations.role_alias_pairs()}
    aliases.update({word: level for word, level in relations.role_alias_pairs()})
    return aliases


def canonical_level(level: str | None) -> str | None:
    """A level name in any supported spelling -> the registry's name."""
    if not level:
        return None
    key = level.strip().lower()
    resolved = _level_aliases().get(key, key)
    # hierarchy owns legacy renames (business_center -> bcm); role
    # aliases and legacy names are different vocabularies, so both run.
    return hierarchy.canonical_level(resolved)


# Every level whose values can be grounded, and the getter that supplies
# them. Sourced from entity_extractor so this module and the prompt's
# gazetteer can never disagree about what exists, and so the master-sheet
# filter applies here too without being restated.
_VALUE_SOURCES: dict[str, Callable[[Session], list[str]]] = {
    "team": entity_extractor.get_known_teams,
    "company": entity_extractor.get_known_companies,
    "unit_head": entity_extractor.get_known_unit_heads,
    "zonal_head": entity_extractor.get_known_zonal_heads,
    "bcm": entity_extractor.get_known_bcms,
    "office": entity_extractor.get_known_offices,
    "region": entity_extractor.get_known_regions,
    "advisor": entity_extractor.get_known_advisor_names,
}

GROUNDABLE_LEVELS: tuple[str, ...] = tuple(_VALUE_SOURCES)


@dataclass(frozen=True)
class Candidate:
    """One real database record the named entity could be.

    `wid` is populated for advisors only, and that is a fact about the
    data rather than an omission: the hierarchy table stores Unit Heads,
    Zonal Heads and BCMs by NAME in their column, with no identifier of
    their own. For those levels the canonical `value` IS the identifier
    every downstream stage should use.
    """
    value: str
    level: str
    wid: int | None = None
    score: float = 1.0


@dataclass(frozen=True)
class GroundedEntity:
    """What the model named, and what the database says about it.

    `ref` is kept verbatim. Grounding never edits the interpretation it
    was handed — a reader can always see what the model said next to what
    was found, which is what makes a mismatch reviewable instead of
    invisible.
    """
    ref: EntityRef
    role: str
    status: str
    candidates: list[Candidate] = field(default_factory=list)
    # Levels the name WAS found at when the stated level did not match.
    # Reported for the pipeline to act on; never applied here.
    found_at: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.ref.name

    @property
    def stated_level(self) -> str | None:
        return canonical_level(self.ref.level)

    @property
    def resolved(self) -> Candidate | None:
        """The one record — None for every status but RESOLVED.

        Deliberately not `candidates[0]`: an ambiguous result has a
        best-scoring candidate too, and returning it is exactly how a
        coin-flip gets made by accident.
        """
        return self.candidates[0] if self.status == RESOLVED and self.candidates else None

    @property
    def wid(self) -> int | None:
        resolved = self.resolved
        return resolved.wid if resolved else None

    @property
    def value(self) -> str | None:
        resolved = self.resolved
        return resolved.value if resolved else None

    @property
    def is_resolved(self) -> bool:
        return self.status == RESOLVED

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "stated_level": self.stated_level,
            "level_was_stated": self.ref.level_was_stated,
            "status": self.status,
            "value": self.value,
            "wid": self.wid,
            "found_at": list(self.found_at),
            "candidates": [
                {"value": c.value, "level": c.level, "wid": c.wid, "score": round(c.score, 2)}
                for c in self.candidates
            ],
        }


@dataclass
class Grounding:
    """Every named entity in one interpretation, grounded.

    The structured ambiguity state the pipeline consumes later: it does
    not raise, does not drop entities, and does not rank problems. It
    reports what each name turned out to be so the caller can decide
    whether to ask, degrade, or proceed.
    """
    entities: list[GroundedEntity] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.entities)

    def by_role(self, role: str) -> list[GroundedEntity]:
        return [e for e in self.entities if e.role == role]

    @property
    def subject(self) -> GroundedEntity | None:
        subjects = self.by_role(SUBJECT)
        return subjects[0] if subjects else None

    @property
    def ambiguous(self) -> list[GroundedEntity]:
        return [e for e in self.entities if e.status == AMBIGUOUS]

    @property
    def not_found(self) -> list[GroundedEntity]:
        return [e for e in self.entities if e.status == NOT_FOUND]

    @property
    def mismatched(self) -> list[GroundedEntity]:
        return [e for e in self.entities if e.status == TYPE_MISMATCH]

    @property
    def is_fully_grounded(self) -> bool:
        return all(e.is_resolved for e in self.entities)

    @property
    def needs_clarification(self) -> bool:
        """Anything a human would have to settle before the query means
        one thing. NOT_FOUND is excluded on purpose — there is nothing to
        choose between, so it is an answer ("no such team"), not a
        question."""
        return bool(self.ambiguous or self.mismatched)

    def to_dict(self) -> dict:
        return {
            "is_fully_grounded": self.is_fully_grounded,
            "needs_clarification": self.needs_clarification,
            "entities": [e.to_dict() for e in self.entities],
        }


def _floor_for(level: str) -> float:
    """PERSON-valued levels use the stricter advisor floor.

    The team floor is routinely cleared by genuinely different people in
    this name population — "yasir ali" vs "asif ali" scores 0.82 — so
    matching a person at 0.80 invents an entity the query never named.
    hierarchy.match_kind_for owns which levels are person-valued.
    """
    return (advisor_resolver.PERSON_FLOOR
            if hierarchy.match_kind_for(level) == "advisor"
            else fuzzy_match.STRONG_FLOOR)


def _advisor_candidates(name: str, db: Session) -> list[Candidate]:
    """Advisors carry a real identifier, so they resolve through
    advisor_resolver rather than through the value list — that is what
    supplies the wid, and it is also what surfaces duplicate names as
    several candidates instead of one arbitrary person."""
    exact = advisor_resolver.resolve_by_name(name, db)
    if exact.candidates:
        return [Candidate(value=c.name, level="advisor", wid=c.wid, score=c.score)
                for c in exact.candidates]

    # No exact record: fall back to the same fuzzy tier the extractor
    # uses, then re-resolve the matched NAME so the wid still comes from
    # identity resolution rather than from a string comparison.
    known = entity_extractor.get_known_advisor_names(db)
    hit = fuzzy_match.best_match(name, known, kind="advisor",
                                 floor=advisor_resolver.PERSON_FLOOR)
    if not hit:
        return []
    matched, score = hit
    return [Candidate(value=c.name, level="advisor", wid=c.wid, score=score)
            for c in advisor_resolver.resolve_by_name(matched, db).candidates]


def _group_candidates(name: str, level: str, db: Session) -> list[Candidate]:
    """Exact (case-insensitive) first, then fuzzy — the existing tiers,
    in the existing order. Every value at or above the floor is kept:
    picking the best one here is what "do not arbitrarily choose" forbids.
    """
    values = _VALUE_SOURCES[level](db)
    target = name.strip().lower()

    exact = [v for v in values if v and v.lower() == target]
    if exact:
        return [Candidate(value=v, level=level, score=1.0) for v in exact]

    kind = hierarchy.match_kind_for(level)
    floor = _floor_for(level)
    scorer = fuzzy_match._scorer(kind)
    scored = [(v, round(scorer(target, v.lower()), 2)) for v in values if v]
    above = [(v, s) for v, s in scored if s >= floor]
    above.sort(key=lambda pair: pair[1], reverse=True)
    return [Candidate(value=v, level=level, score=s) for v, s in above]


def _candidates_at(name: str, level: str, db: Session) -> list[Candidate]:
    if level == "advisor":
        return _advisor_candidates(name, db)
    if level in _VALUE_SOURCES:
        return _group_candidates(name, level, db)
    return []


def _decide(candidates: list[Candidate]) -> str:
    """One record is RESOLVED; several too close to separate is AMBIGUOUS.

    A clear best match wins outright — that is what the margin is for.
    Two records with the SAME value at the same level (duplicate names,
    the eight Yasir Alis) can never be separated by score, so they are
    ambiguous however they scored.
    """
    if not candidates:
        return NOT_FOUND
    if len(candidates) == 1:
        return RESOLVED
    best, second = candidates[0], candidates[1]
    if best.value.lower() == second.value.lower():
        return AMBIGUOUS
    return RESOLVED if best.score - second.score > AMBIGUITY_MARGIN else AMBIGUOUS


def _other_levels_with(name: str, db: Session, exclude: str | None) -> tuple[str, ...]:
    """Which levels DO have this name. Used only to describe a mismatch."""
    return tuple(
        level for level in GROUNDABLE_LEVELS
        if level != exclude and _candidates_at(name, level, db)
    )


def ground_entity(ref: EntityRef, db: Session, *, role: str = SUBJECT) -> GroundedEntity:
    """Ground one named entity at the level the interpretation states.

    Three shapes, and the difference between them is entirely about who
    asserted the type:

      - a STATED level is verified as given. It is never traded for
        another level that happens to match, which is the whole contract.
      - NO level means the model did not commit to one ("connects of
        Faisal"), so every level is searched. Matching at exactly one is
        an answer; matching at several is AMBIGUOUS, not a vote.
      - an UNGROUNDABLE level (nothing stores values for it) can only be
        reported, not checked.
    """
    level = canonical_level(ref.level)

    if level is None:
        found: list[Candidate] = []
        for candidate_level in GROUNDABLE_LEVELS:
            found.extend(_candidates_at(ref.name, candidate_level, db))
        levels = {c.level for c in found}
        if not found:
            return GroundedEntity(ref=ref, role=role, status=NOT_FOUND)
        if len(levels) > 1:
            # The model declined to say which type this is and the name
            # is real at several. Answering with any one of them would be
            # inventing the interpretation the model withheld.
            return GroundedEntity(ref=ref, role=role, status=AMBIGUOUS,
                                  candidates=found, found_at=tuple(sorted(levels)))
        return GroundedEntity(ref=ref, role=role, status=_decide(found), candidates=found,
                              found_at=tuple(levels))

    if level not in _VALUE_SOURCES:
        return GroundedEntity(ref=ref, role=role, status=NOT_FOUND)

    candidates = _candidates_at(ref.name, level, db)
    if candidates:
        return GroundedEntity(ref=ref, role=role, status=_decide(candidates),
                              candidates=candidates, found_at=(level,))

    # Nothing at the stated level. Reporting WHERE it does exist is the
    # useful part — and reporting is all that happens. Applying it here
    # would answer about an advisor when a team was asked for.
    elsewhere = _other_levels_with(ref.name, db, exclude=level)
    if elsewhere:
        alternatives: list[Candidate] = []
        for other in elsewhere:
            alternatives.extend(_candidates_at(ref.name, other, db))
        return GroundedEntity(ref=ref, role=role, status=TYPE_MISMATCH,
                              candidates=alternatives, found_at=elsewhere)

    return GroundedEntity(ref=ref, role=role, status=NOT_FOUND)


def ground(model: SemanticModel, db: Session) -> Grounding:
    """Ground every entity one interpretation names.

    Order is subject, then scope, then comparison targets — the order a
    reader would check them in, and the order the pipeline reports them.
    The model is not modified: this returns a parallel structure, so the
    interpretation and what the database said about it stay separable.
    """
    grounded: list[GroundedEntity] = []

    if model.subject is not None:
        grounded.append(ground_entity(model.subject, db, role=SUBJECT))
    for ref in model.scope:
        grounded.append(ground_entity(ref, db, role=SCOPE))
    for ref in model.comparison_subjects:
        grounded.append(ground_entity(ref, db, role=COMPARISON))

    return Grounding(entities=grounded)
