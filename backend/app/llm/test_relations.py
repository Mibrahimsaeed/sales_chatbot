"""Relationship declarations (app/llm/relations.py) — M0.

M0 introduces no behaviour, so these tests defend exactly two things:

1. The registry says what the codebase already said. MANAGER_COLUMNS is
   now DERIVED from these declarations, so a drift between the two would
   silently change reverse-hierarchy routing ("who is X's BM") — the one
   advisor->X capability that works today.
2. The extension point is genuinely generic. Adding a relationship must
   be a declaration, never a code change, or the engine has failed its
   stated purpose before M1 begins.
"""

import pytest

from app.database.models import Advisor
from app.llm import hierarchy, relations


# The membership MANAGER_COLUMNS had before M0, written out literally.
# Deliberately NOT derived from the registry — a test that computed its
# expectation the same way as the code would pass no matter what both
# said.
# REBOUND BY PHASE 3 to the chain Phase 1 verified against production
# data. `unit_head` moved from Advisor.bm to Advisor.rm and `zonal_head`
# from Advisor.zm to Advisor.portfolio_lead; `bcm` replaced
# `business_center` as the level directly above advisor. `bm`/`zm` remain
# reverse-lookup-only under their own names, so "who is X's BM" still
# answers from the column it actually reads.
EXPECTED_MANAGER_COLUMNS = {
    "unit_head": Advisor.rm,
    "zonal_head": Advisor.portfolio_lead,
    "bcm": Advisor.management_lead,
    "bm": Advisor.bm,
    "zm": Advisor.zm,
    "team": Advisor.team,
    "company": Advisor.company,
}


def test_manager_columns_membership_is_unchanged_by_the_refactor():
    assert set(hierarchy.MANAGER_COLUMNS) == set(EXPECTED_MANAGER_COLUMNS) | {"office"}


def test_manager_columns_point_at_the_same_columns_as_before():
    for level, column in EXPECTED_MANAGER_COLUMNS.items():
        assert hierarchy.MANAGER_COLUMNS[level] is column


def test_manager_column_for_still_resolves_every_reverse_level():
    """The public accessor used by hierarchy_service.get_manager_of()."""
    for level, column in EXPECTED_MANAGER_COLUMNS.items():
        assert hierarchy.manager_column_for(level) is column
    assert hierarchy.manager_column_for("nonexistent_level") is None


def test_is_manager_level_did_not_gain_or_lose_levels():
    for level in EXPECTED_MANAGER_COLUMNS:
        assert hierarchy.is_manager_level(level)
    # region/unit are real Advisor columns but are NOT declared yet; if
    # they ever become manager levels it must be a deliberate decision,
    # not a side effect of declaring a relationship.
    assert not hierarchy.is_manager_level("region")
    assert not hierarchy.is_manager_level("nonexistent")


def test_every_declared_advisor_relation_maps_to_a_real_advisor_column():
    for spec in relations.registry.specs_for("advisor"):
        assert spec.column is not None
        assert spec.column.parent.class_ is Advisor


def test_registry_declares_the_verified_chain_plus_attributes():
    """The five chain levels, the office attribute, and the two legacy
    reverse-only manager columns."""
    targets = set(relations.registry.targets_for("advisor"))
    assert targets == set(EXPECTED_MANAGER_COLUMNS) | {"office"}


def test_lookup_by_source_and_target():
    spec = relations.registry.resolve("advisor", "team")
    assert spec is not None
    assert spec.column is Advisor.team
    assert spec.cardinality == relations.ONE
    assert relations.registry.resolve("advisor", "not_a_level") is None
    assert relations.registry.resolve("team", "advisor") is None  # inverse: not declared yet


def test_cached_flag_marks_only_what_the_identity_cache_actually_carries():
    """The `cached` flag is a claim about advisor_resolver's projection.
    UPDATED BY M3, which widened that projection from (team, company) to
    include the three group levels below; the flag is now what BUILDS the
    projection, so this set is the definition rather than a mirror of it.

    UPDATED BY PHASE 3: the chain was rebound, so the cached set is now
    the five chain levels above advisor plus the office attribute. `bm`
    and `zm` stay uncached — they are reverse-lookup-only columns with no
    group level to be inferred into, so caching them would spend memory
    on a value nothing can consume."""
    cached = {s.target_level for s in relations.registry.specs_for("advisor") if s.cached}
    assert cached == {"team", "company", "unit_head", "zonal_head", "bcm", "office"}


def test_registering_a_new_relation_requires_no_code_change():
    """The extensibility claim, exercised: declare, look up, derive."""
    scratch = relations.RelationRegistry()
    spec = scratch.register(
        relations.RelationSpec(
            source_level="advisor", target_level="region", column=Advisor.region
        )
    )
    assert scratch.resolve("advisor", "region") is spec
    assert scratch.targets_for("advisor") == ["region"]
    assert scratch.has("advisor", "region")


def test_reverse_lookup_false_keeps_a_relation_out_of_manager_columns():
    """Declaring a relationship must not silently make a new level
    reverse-lookupable — that would change routing for "who is X's ...".
    """
    scratch = relations.RelationRegistry()
    scratch.register(relations.RelationSpec(
        source_level="advisor", target_level="team", column=Advisor.team))
    scratch.register(relations.RelationSpec(
        source_level="advisor", target_level="region", column=Advisor.region,
        reverse_lookup=False))

    derived = {
        s.target_level: s.column
        for s in scratch.specs_for("advisor") if s.reverse_lookup
    }
    assert set(derived) == {"team"}


def test_registration_is_idempotent_on_the_same_key():
    scratch = relations.RelationRegistry()
    scratch.register(relations.RelationSpec("advisor", "team", Advisor.team))
    scratch.register(relations.RelationSpec("advisor", "team", Advisor.team, cached=True))
    assert len(scratch.all()) == 1
    assert scratch.resolve("advisor", "team").cached is True


def test_specs_are_immutable():
    """Specs are data shared process-wide; a consumer mutating one would
    change resolution for every later request."""
    spec = relations.registry.resolve("advisor", "team")
    with pytest.raises(Exception):
        spec.target_level = "company"
