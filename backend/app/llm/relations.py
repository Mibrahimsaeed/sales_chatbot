"""
Entity relationship declarations (M0 of the Relationship Inference Engine).

WHAT THIS IS. A declarative table of the relationships between entity
levels — "an advisor HAS a team", "an advisor HAS a unit head" — plus the
accessor that reaches the related value. It is data, not behaviour:
nothing here reads the database, parses text, or resolves anything. The
resolver (M1) and the reference parser (M1) are separate components that
consume these declarations.

WHY IT EXISTS. The architectural audit established that entity resolution
only grounds entities the user NAMED. "Waqar Haider's team" cannot
resolve, because nothing maps a resolved advisor to their team. That
mapping was in fact already present in the codebase — as
`hierarchy.MANAGER_COLUMNS` — but scoped by its own contract to a single
use ("which column holds this advisor's manager"), so it could not be
reused for the general case without either widening a contract silently
or copying the table. This module is that table, generalised, with
MANAGER_COLUMNS now derived FROM it rather than duplicating it (audit
debt item D2).

DESIGN RULES, deliberately narrow:

- A spec is DATA. `column` is a SQLAlchemy column; no callables, no
  branching, no query-shape awareness. A relation that needs conditional
  logic does not belong here.
- Registration is at import time, mirroring the existing idiom in
  entity_extractor.py (`entity_linker.register_entity_type(...)`), so
  adding a relationship stays a one-line declaration.
- `cached` records whether the value is ALREADY in advisor_resolver's
  in-memory identity projection. It is the difference between a
  relationship resolvable for free and one that costs a database read
  (audit debt item D3), and it is why M1 (team/company) is cheap while
  M3 (bm/zm/office/...) is not.
- `reverse_lookup` marks the relations that answer "who is X's <role>".
  It exists so MANAGER_COLUMNS can be derived with EXACTLY its previous
  membership — a relation being declared here must not silently make a
  new level reverse-lookupable, which would change routing.

M0 SCOPE: declarations only. No resolver, no parser, no behaviour change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.database.models import Advisor

# Cardinality of the TARGET side. Every relation declared today is "one"
# (an advisor has exactly one team); "many" is declared here because the
# inverse direction (a team has many advisors) is a planned extension and
# the resolver contract must not assume scalars.
ONE = "one"
MANY = "many"


@dataclass(frozen=True)
class RelationSpec:
    """One directed relationship: source_level --> target_level."""

    source_level: str
    target_level: str
    # The Advisor column holding the target's value for a given source row.
    column: Any
    cardinality: str = ONE
    # True when advisor_resolver's identity cache already carries this
    # value, so resolving it costs no database read.
    cached: bool = False
    # True when this relation backs a "who is X's <role>?" question.
    reverse_lookup: bool = True
    # M2: the words a user says to NAME this relation as somebody's role
    # — "bm", "unit head", "division head". Multi-word aliases are written
    # with single spaces; pattern building relaxes those to \s* so
    # "unithead" and "unit  head" match exactly as they did when this
    # vocabulary was two hand-written regexes.
    #
    # Empty is meaningful, not merely unset: `team` and `company` ARE
    # reverse-lookupable relations, but no phrasing may route a question
    # about them to reverse lookup — "X's team" asks about a group, not
    # about a person above X. Giving them aliases would silently reroute
    # every M1 query.
    role_aliases: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return (self.source_level, self.target_level)


class RelationRegistry:
    """Lookup over declared relations.

    Intentionally minimal: registration, point lookup, and enumeration.
    Any policy about WHETHER a relation should be inferred for a given
    query belongs to the caller, not here — this registry only answers
    what is declarable and how to reach it.
    """

    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], RelationSpec] = {}

    def register(self, spec: RelationSpec) -> RelationSpec:
        """Last registration wins, so a test or a future override can
        replace a declaration without mutating this module."""
        self._specs[spec.key] = spec
        return spec

    def resolve(self, source_level: str, target_level: str) -> RelationSpec | None:
        return self._specs.get((source_level, target_level))

    def has(self, source_level: str, target_level: str) -> bool:
        return (source_level, target_level) in self._specs

    def targets_for(self, source_level: str) -> list[str]:
        """Every level reachable FROM `source_level`, declaration order."""
        return [s.target_level for s in self._specs.values() if s.source_level == source_level]

    def specs_for(self, source_level: str) -> list[RelationSpec]:
        return [s for s in self._specs.values() if s.source_level == source_level]

    def all(self) -> list[RelationSpec]:
        return list(self._specs.values())


registry = RelationRegistry()


# ---------------------------------------------------------------------
# Declarations
#
# These reproduce, exactly, the advisor-rooted mappings that already
# existed in hierarchy.MANAGER_COLUMNS — same target names, same columns.
# Nothing new is declared here: M0 changes where the mapping LIVES, not
# what it says. `region` and `unit` are deliberately NOT declared yet;
# they have no hierarchy level, no gazetteer and no compiler binding, so
# declaring them would be configuration nothing can consume (they arrive
# with M5).
# ---------------------------------------------------------------------

def _advisor(target_level: str, column, *, cached: bool = False, reverse_lookup: bool = True,
             role_aliases: tuple[str, ...] = ()):
    return registry.register(
        RelationSpec(
            source_level="advisor",
            target_level=target_level,
            column=column,
            cardinality=ONE,
            cached=cached,
            reverse_lookup=reverse_lookup,
            role_aliases=role_aliases,
        )
    )


# cached=True: advisor_resolver.refresh_cache() already selects team and
# company onto every AdvisorIdentity, so these two resolve with no query.
# No role_aliases — see RelationSpec.role_aliases.
_advisor("team", Advisor.team, cached=True)
_advisor("company", Advisor.company, cached=True)

# PHASE 3 REBIND. The three chain levels above advisor, now pointing at
# the columns Phase 1 verified against production data. `unit_head` and
# `zonal_head` keep their keys — those were always the business's words,
# never data identifiers — and moved from Advisor.bm/Advisor.zm, which
# Phase 1 showed do not nest (75% of zm values spanned several bm values).
# `bcm` replaces `business_center`: the level directly above advisor is a
# PERSON (management lead), not a place.
#
# cached=True on all three: they are on the identity projection, so
# relationship inference resolves them with no extra query.
_advisor("bcm", Advisor.management_lead, cached=True,
         role_aliases=("bcm", "business center manager", "business centre manager",
                       "management lead"))
_advisor("zonal_head", Advisor.portfolio_lead, cached=True,
         role_aliases=("zonal head", "zone head", "portfolio lead"))
_advisor("unit_head", Advisor.rm, cached=True,
         role_aliases=("unit head", "division head", "rm", "regional manager",
                       "region manager", "regional head"))

# `office` is a groupable ATTRIBUTE, not a chain level (Phase 1: teams
# span offices, so offices do not contain teams). Inferable and
# filterable; never a "who is X's ..." answer, hence reverse_lookup=False.
# Aliases are longest-first at use, so "business center manager" reaches
# `bcm` while a bare "business center" reaches the PLACE. That split is
# the whole point of separating the manager from the office.
_advisor("office", Advisor.office, cached=True,
         role_aliases=("business center", "business centre", "branch"))

# Legacy manager columns. Phase 1 disproved bm/zm as CHAIN levels, but
# "who is X's BM" is still a truthful question about one column on one
# advisor's row — a strictly smaller capability than aggregating under
# them. Kept reverse-lookup-only, with their own keys so the answer names
# the column it actually read rather than borrowing a chain level's label.
_advisor("bm", Advisor.bm, role_aliases=("bm",))
_advisor("zm", Advisor.zm, role_aliases=("zm",))


def role_alias_pairs(source_level: str = "advisor") -> list[tuple[str, str]]:
    """Every (alias, target_level) pair, LONGEST ALIAS FIRST.

    Longest-first is what makes specificity ordering a property of the
    data instead of a hand-maintained list order: "regional manager" is
    tested before "manager", and "business center" before "branch",
    because they are longer — not because someone remembered to put them
    higher up. That ordering rule was previously a comment above a
    literal list, which is exactly the kind of instruction that rots.
    """
    pairs = [
        (alias, spec.target_level)
        for spec in registry.specs_for(source_level)
        for alias in spec.role_aliases
    ]
    return sorted(pairs, key=lambda pair: (-len(pair[0]), pair[0]))


def role_aliases(source_level: str = "advisor") -> list[str]:
    """Every role alias, longest first."""
    return [alias for alias, _level in role_alias_pairs(source_level)]


def manager_columns() -> dict[str, Any]:
    """The reverse-lookupable advisor relations, in the shape
    hierarchy.MANAGER_COLUMNS has always had.

    Derived rather than duplicated so there is one source of truth: a
    relation added below cannot drift from what reverse lookup supports,
    and cannot silently JOIN it either — that requires reverse_lookup=True
    at the declaration.
    """
    return {
        spec.target_level: spec.column
        for spec in registry.specs_for("advisor")
        if spec.reverse_lookup
    }
