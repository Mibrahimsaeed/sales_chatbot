"""
THE organizational hierarchy. One declaration; everything derives from it.

VERIFIED CHAIN (Phase 1, against production data — bottom to top):

    advisor -> bcm -> zonal_head -> unit_head -> team

    | level       | business label | Advisor column        | distinct |
    |-------------|----------------|-----------------------|----------|
    | advisor     | Advisor        | name                  | 711      |
    | bcm         | BCM            | management_lead       | 187      |
    | zonal_head  | Zonal Head     | portfolio_lead        |  90      |
    | unit_head   | Unit Head      | rm                    |  15      |
    | team        | Team           | team                  |  10      |

Cardinality decreases monotonically upward, and every edge passed a
containment test with at most 2 parents per node (data-quality noise on a
sound tree).

WHAT THIS REPLACED, AND WHY. The previous chain was
company -> region -> unit_head(bm) -> zonal_head(zm) -> unit -> business_
center(office) -> team -> advisor. Phase 1 disproved it outright: 100% of
teams spanned multiple offices (worst node: 17), 75% of zm values spanned
multiple bm values, and the "unit head" level had 138 members sitting
ABOVE a "zonal head" level of 59 — impossible in a containment tree.
Those columns are orthogonal attributes, not nested levels.

The level KEYS did not change for unit_head/zonal_head, because they were
never data identifiers — they are the words the business uses and the
words users type. What changed is the COLUMN each one reads. `bcm`
replaced `business_center` as the level directly above advisor.

LEVELS vs ATTRIBUTES. Only the five above form the chain. `company`,
`office` and `region` remain groupable and filterable — no capability is
lost — but they are NOT part of parent/child traversal, because the data
does not nest them. Keeping that distinction explicit is what stops a
breakdown from walking a chain that isn't there.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import and_, func, or_

from app.database.models import Advisor
from app.llm import relations

# =====================================================================
# THE declaration. Everything below is derived from these two tables.
# =====================================================================

# The verified chain, TOP to BOTTOM. Ordering here IS the hierarchy:
# PARENT_LEVEL, ancestors, descendants and breakdown nesting are all
# computed from this list, so there is no second place to keep in sync.
CHAIN: list[str] = ["team", "unit_head", "zonal_head", "bcm", "advisor"]

# Every level's backing column and business label. The chain levels plus
# the attributes that are groupable but do not nest.
_LEVEL_SPEC: dict[str, tuple[Any, str]] = {
    # ---- the verified chain ----
    "team":        (Advisor.team,            "Team"),
    "unit_head":   (Advisor.rm,              "Unit Head"),
    "zonal_head":  (Advisor.portfolio_lead,  "Zonal Head"),
    "bcm":         (Advisor.management_lead, "BCM"),
    "advisor":     (Advisor.name,            "Advisor"),
    # ---- attributes: groupable/filterable, outside the chain ----
    "company":     (Advisor.company,         "Company"),
    "office":      (Advisor.office,          "Office"),
    # `region` is mixed-semantics — see AMBIGUOUS_LEVELS below.
    "region":      (Advisor.region,          "Region"),
}

# Levels that can be filtered and grouped but are NOT part of the chain.
ATTRIBUTE_LEVELS: list[str] = [lvl for lvl in _LEVEL_SPEC if lvl not in CHAIN]

# ---------------------------------------------------------------------
# `region` holds TWO different concepts in one column (Phase 1):
#   master-sheet advisors -> a PERSON  ("Kaleem Satti", "Chairman", ...)
#   every other row       -> a PLACE   ("North", "Center", "South")
# because MasterSheet "Regional" and CCMC "Region" are both written to
# Advisor.region under don't-overwrite semantics.
#
# Splitting them needs an ETL change, which is out of Phase 3's scope, so
# the ambiguity is DECLARED here rather than silently carried. When the
# column is split, the change is confined to this file: add the second
# level to _LEVEL_SPEC and drop the entry below. No consumer needs
# touching, because every consumer already derives from these tables.
# ---------------------------------------------------------------------
AMBIGUOUS_LEVELS: dict[str, str] = {
    "region": "Advisor.region mixes a geographic region (non-master rows) "
              "with a regional head's name (master-sheet rows). Gazetteers "
              "filter to master-sheet rows, so today this level resolves "
              "PEOPLE. Split the ETL column before treating it as a place.",
}

# Old level names kept resolvable so a stored QueryIR, a ChatLog row or an
# API caller written before the rebind still lands somewhere sensible.
# `business_center` meant Advisor.office, which is now `office`.
LEVEL_ALIASES: dict[str, str] = {
    "business_center": "office",
}

# =====================================================================
# Derived tables — do not add entries here, add them above.
# =====================================================================

# EVERY level a query can address: the nesting chain plus the groupable
# attributes. This is the vocabulary the LLM is offered (llm_client's
# JSON schema enums, planner_prompt, prompt_builder) and the set
# query_ir.Level is derived from.
#
# F2. This was `list(CHAIN)` — the chain ONLY. Because it feeds the
# structured-output enum with strict:True, grammar-constrained decoding
# made it impossible for the model to emit `company`, `office` or
# `region` at all: "top companies by revenue" could not come back as a
# company ranking, it was forced into a chain level nobody asked about.
# The narrowing was invisible because the only test covering it asserted
# a SUBSET relation against query_ir.Level, which a narrowing passes.
#
# Chain first, so anything rendering this list reads top-down.
HIERARCHY_LEVELS: list[str] = list(CHAIN) + list(ATTRIBUTE_LEVELS)
LEVEL_COLUMNS: dict[str, Any] = {lvl: col for lvl, (col, _l) in _LEVEL_SPEC.items()}
LEVEL_LABELS: dict[str, str] = {lvl: label for lvl, (_c, label) in _LEVEL_SPEC.items()}

# Everything a query can group or filter by, advisor excluded (it is the
# leaf, addressed by name/wid rather than grouped).
GROUP_LEVELS: list[str] = [lvl for lvl in _LEVEL_SPEC if lvl != "advisor"]

# Levels reached through the compiler's generic advisor-column rollup
# rather than a per-metric ontology binding. team and company keep their
# own explicit bindings (TeamTarget among them) and must not be re-routed.
NEW_GROUP_LEVELS: list[str] = [lvl for lvl in GROUP_LEVELS if lvl not in ("team", "company")]

# Immediate parent per level, derived from CHAIN. Attributes have no
# parent — they do not nest, which is precisely why they are attributes.
PARENT_LEVEL: dict[str, str | None] = {
    **{level: (CHAIN[i - 1] if i > 0 else None) for i, level in enumerate(CHAIN)},
    **{level: None for level in ATTRIBUTE_LEVELS},
}

# Which fuzzy_match scorer grounds a level's values: PERSON-valued levels
# use the stricter advisor floor, org-unit names use the team floor.
_PERSON_VALUED = {"unit_head", "zonal_head", "bcm", "advisor", "region"}
LEVEL_MATCH_KIND: dict[str, str] = {
    lvl: ("advisor" if lvl in _PERSON_VALUED else ("company" if lvl == "company" else "team"))
    for lvl in _LEVEL_SPEC
}

# The plural key entity_extractor emits per level. Irregular plurals are
# spelled out; the rest are mechanical.
_IRREGULAR_PLURALS = {"company": "companies", "office": "offices"}
LEVEL_ENTITY_KEYS: dict[str, str] = {
    lvl: _IRREGULAR_PLURALS.get(lvl, f"{lvl}s") for lvl in GROUP_LEVELS
}

# Phrasing a user might type for each level. Deliberately NOT derived from
# the labels: "bcm" and "business center manager" both mean `bcm`, and no
# derivation produces that.
LEVEL_KEYWORDS: dict[str, list[str]] = {
    "team": ["team", "teams"],
    "unit_head": ["unit head", "unit heads", "unit-head", "unit-heads", "unit", "division"],
    "zonal_head": ["zonal head", "zonal heads", "zone head", "zone heads", "zonal-head", "zone"],
    "bcm": ["bcm", "bcms", "business center manager", "business centre manager",
            "management lead", "management leads"],
    "advisor": ["advisor", "advisors", "agent", "agents"],
    "company": ["company", "companies"],
    "office": ["office", "offices", "business center", "business centre",
               "business centers", "business centres", "center", "centre", "branch"],
    "region": ["region", "regions"],
}


# =====================================================================
# Traversal — generic, derived from CHAIN. No hardcoded pairs.
# =====================================================================

def canonical_level(level: str | None) -> str | None:
    """Resolve a legacy level name to its current one."""
    if level is None:
        return None
    return LEVEL_ALIASES.get(level, level)


def is_chain_level(level: str | None) -> bool:
    return canonical_level(level) in CHAIN


def parent_of(level: str) -> str | None:
    """The level immediately ABOVE `level`, or None at the top / for an
    attribute."""
    return PARENT_LEVEL.get(canonical_level(level))


def child_of(level: str) -> str | None:
    """The level immediately BELOW `level`, or None at the leaf / for an
    attribute. This is what a breakdown groups by."""
    level = canonical_level(level)
    if level not in CHAIN:
        return None
    index = CHAIN.index(level)
    return CHAIN[index + 1] if index + 1 < len(CHAIN) else None


def ancestors(level: str) -> list[str]:
    """Every level above `level`, nearest first."""
    level = canonical_level(level)
    if level not in CHAIN:
        return []
    return list(reversed(CHAIN[: CHAIN.index(level)]))


def descendants(level: str) -> list[str]:
    """Every level below `level`, nearest first."""
    level = canonical_level(level)
    if level not in CHAIN:
        return []
    return CHAIN[CHAIN.index(level) + 1:]


def depth(level: str) -> int | None:
    """0 at the top of the chain; None for an attribute."""
    level = canonical_level(level)
    return CHAIN.index(level) if level in CHAIN else None


def scope_filter(level: str, value: str):
    """The SQLAlchemy predicate selecting every advisor under `value` at
    `level`.

    THE one definition of "in scope". Filtering, breakdown, comparison
    and aggregation all call this, so a query cannot scope one way in a
    leaderboard and another in a comparison. Every level is a column on
    the advisor row, so scoping is a column match at any level — no join
    walk is required, and the chain's job is to say what NESTS inside
    what, not how to reach it.
    """
    column = column_for(level)
    if column is None:
        raise ValueError(f"'{level}' is not a known hierarchy level")
    return column.ilike(value)


def direct_scope_filter(level: str, value: str, target_level: str | None = None):
    """The predicate selecting only `value`'s IMMEDIATE reports.

    `scope_filter` above is the whole subtree, because one column match
    on a denormalised row IS every descendant: `rm ilike 'Faisal'`
    selects all 16 advisors beneath that Unit Head however many managers
    sit between. That is the right answer for "X's team" and the wrong
    one for "who reports DIRECTLY to X" — the word was not in any
    vocabulary, so both questions returned the same 16 people.

    Directness is read off the chain rather than declared: a row is an
    immediate report at `target_level` when the column of the level
    ABOVE it holds `value`. Advisors are the leaf, so their immediate
    manager is `bcm` (management_lead) — which is why the same Unit Head
    is `rm` to 16 people, `portfolio_lead` to 11 and `management_lead`
    to the 4 he actually manages himself.

    `target_level` defaults to the level immediately below `level`, so
    "who reports to a Unit Head" enumerates Zonal Heads. Naming it
    explicitly is what "how many ADVISORS report directly to X" needs:
    the target is the leaf, and the predicate becomes the leaf's own
    parent column regardless of how far above it `level` sits.

    SELF IS EXCLUDED. One person legitimately occupies several levels —
    Faisal Hussain Naqvi is his own Zonal Head for 11 advisors — so
    without this his direct reports include himself, and a headcount of
    his reports counts him as one of them. Excluding it here rather than
    at each caller keeps the count and the roster over one population.

    Returns None when the level has no reports to enumerate (an advisor
    is the leaf), which callers render as "not a question about this
    level" rather than as an empty result.
    """
    level = canonical_level(level)
    target = canonical_level(target_level) or child_of(level)
    if target is None:
        return None

    manager_level = parent_of(target)
    if manager_level is None:
        return None
    manager_column = column_for(manager_level)
    if manager_column is None:
        raise ValueError(f"'{manager_level}' is not a known hierarchy level")

    predicate = manager_column.ilike(value)

    own_column = column_for(target)
    if own_column is not None:
        predicate = and_(
            predicate,
            or_(own_column.is_(None),
                func.lower(own_column) != (value or "").lower()),
        )
    return predicate


# ---------------------------------------------------------------------
# Reverse hierarchy — "who is X's BM/ZM/RM?"
#
# A SUPERSET of the chain: it also covers manager columns that are NOT
# levels. Asking which column on one advisor's row holds their manager is
# a strictly smaller capability than aggregating everyone under that
# manager, and `bm`/`zm` earn the first without qualifying for the second
# — Phase 1 showed they do not nest.
#
# Derived from relations.py so declaring a relationship teaches reverse
# lookup and the registry at once.
# ---------------------------------------------------------------------
MANAGER_COLUMNS: dict[str, Any] = relations.manager_columns()

MANAGER_LABELS: dict[str, str] = {
    **LEVEL_LABELS,
    "bm": "BM",
    "zm": "ZM",
    "regional_manager": "Regional Manager",
    "portfolio_lead": "Portfolio Lead",
    "management_lead": "Management Lead",
}


def manager_column_for(level: str):
    """The Advisor column holding this advisor's manager at `level`, or
    None if `level` isn't a reverse-lookupable field."""
    return MANAGER_COLUMNS.get(canonical_level(level))


def is_manager_level(level: str) -> bool:
    return canonical_level(level) in MANAGER_COLUMNS


def column_for(level: str):
    """The Advisor column backing `level`, or None if `level` isn't part
    of this hierarchy (e.g. a metric key or "attendance_status")."""
    return LEVEL_COLUMNS.get(canonical_level(level))


def label_for(level: str | None) -> str:
    if not level:
        return "value"
    level = canonical_level(level)
    return MANAGER_LABELS.get(level, LEVEL_LABELS.get(level, level))


def is_valid_level(level: str) -> bool:
    return canonical_level(level) in LEVEL_COLUMNS


def match_kind_for(level: str) -> str:
    return LEVEL_MATCH_KIND.get(canonical_level(level), "team")
