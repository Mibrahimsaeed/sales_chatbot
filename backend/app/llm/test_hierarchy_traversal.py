"""Phase 3 — one hierarchy declaration, everything derived from it.

The properties that matter:

1. ONE declaration. PARENT_LEVEL, the level tables, the planner's
   ordering, the validator's gazetteers and the plural labels are all
   DERIVED from `hierarchy.CHAIN` + `_LEVEL_SPEC`. A second copy of the
   chain anywhere is what the previous design had, and it is how the
   chain came to be wrong in three different files at once.
2. GENERIC traversal. parent/child/ancestors/descendants read the chain;
   none of them names a level.
3. The chain matches what Phase 1 verified against production data.
"""

import pytest

from app.database.models import Advisor
from app.llm import hierarchy, relations


# ---------------------------------------------------------------------
# The verified chain
# ---------------------------------------------------------------------

def test_the_chain_is_the_one_phase_1_verified():
    assert hierarchy.CHAIN == ["team", "unit_head", "zonal_head", "bcm", "advisor"]


@pytest.mark.parametrize("level,column", [
    ("team", Advisor.team),
    ("unit_head", Advisor.rm),              # Unit Head IS the RM
    ("zonal_head", Advisor.portfolio_lead), # Zonal Head IS the Portfolio Lead
    ("bcm", Advisor.management_lead),       # BCM IS the Management Lead
    ("advisor", Advisor.name),
])
def test_every_chain_level_binds_to_its_verified_column(level, column):
    assert hierarchy.column_for(level) is column


def test_business_labels_match_what_the_business_calls_them():
    assert hierarchy.label_for("unit_head") == "Unit Head"
    assert hierarchy.label_for("zonal_head") == "Zonal Head"
    assert hierarchy.label_for("bcm") == "BCM"
    assert hierarchy.label_for("team") == "Team"


# ---------------------------------------------------------------------
# Traversal — generic, derived
# ---------------------------------------------------------------------

def test_parent_lookup():
    assert hierarchy.parent_of("advisor") == "bcm"
    assert hierarchy.parent_of("bcm") == "zonal_head"
    assert hierarchy.parent_of("zonal_head") == "unit_head"
    assert hierarchy.parent_of("unit_head") == "team"
    assert hierarchy.parent_of("team") is None


def test_child_lookup():
    assert hierarchy.child_of("team") == "unit_head"
    assert hierarchy.child_of("unit_head") == "zonal_head"
    assert hierarchy.child_of("zonal_head") == "bcm"
    assert hierarchy.child_of("bcm") == "advisor"
    assert hierarchy.child_of("advisor") is None


def test_ancestor_lookup():
    assert hierarchy.ancestors("advisor") == ["bcm", "zonal_head", "unit_head", "team"]
    assert hierarchy.ancestors("team") == []


def test_descendant_lookup():
    assert hierarchy.descendants("team") == ["unit_head", "zonal_head", "bcm", "advisor"]
    assert hierarchy.descendants("advisor") == []


def test_depth_is_the_chain_position():
    assert hierarchy.depth("team") == 0
    assert hierarchy.depth("advisor") == len(hierarchy.CHAIN) - 1
    assert hierarchy.depth("company") is None      # attribute


def test_traversal_is_symmetric():
    """parent(child(x)) == x for every level with a child."""
    for level in hierarchy.CHAIN:
        child = hierarchy.child_of(level)
        if child is not None:
            assert hierarchy.parent_of(child) == level


def test_attributes_do_not_participate_in_traversal():
    """company/office/region are groupable but do not nest — the data
    says teams span offices, so offices cannot contain teams."""
    for attribute in hierarchy.ATTRIBUTE_LEVELS:
        assert hierarchy.parent_of(attribute) is None
        assert hierarchy.child_of(attribute) is None
        assert hierarchy.ancestors(attribute) == []
        assert hierarchy.descendants(attribute) == []
        assert not hierarchy.is_chain_level(attribute)


# ---------------------------------------------------------------------
# Single source of truth
# ---------------------------------------------------------------------

def test_parent_table_is_derived_not_written():
    """PARENT_LEVEL must equal what the chain implies — a hand-written
    copy is exactly what let the previous chain disagree with itself."""
    expected = {
        level: (hierarchy.CHAIN[i - 1] if i > 0 else None)
        for i, level in enumerate(hierarchy.CHAIN)
    }
    for level, parent in expected.items():
        assert hierarchy.PARENT_LEVEL[level] == parent


def test_every_level_appears_in_every_derived_table():
    for level in hierarchy.CHAIN + hierarchy.ATTRIBUTE_LEVELS:
        assert level in hierarchy.LEVEL_COLUMNS, level
        assert level in hierarchy.LEVEL_LABELS, level
        assert level in hierarchy.LEVEL_KEYWORDS, level
        assert level in hierarchy.PARENT_LEVEL, level
        if level != "advisor":
            assert level in hierarchy.LEVEL_ENTITY_KEYS, level
            assert level in hierarchy.LEVEL_MATCH_KIND, level


def test_the_planner_ordering_is_derived_from_the_chain():
    from app.llm import intent_catalog as cat

    assert set(cat.GROUP_LEVEL_ORDER) == set(hierarchy.GROUP_LEVELS)
    # narrowest first: advisor's parent leads
    assert cat.GROUP_LEVEL_ORDER[0] == "bcm"


def test_the_validator_grounds_every_group_level():
    from app.llm.ir_validator import _SUBJECT_GAZETTEERS

    assert set(_SUBJECT_GAZETTEERS) == set(hierarchy.GROUP_LEVELS)


def test_the_registry_and_the_hierarchy_agree_on_columns():
    """relations.py and hierarchy.py both name the chain; a divergence
    would make inference read one column and filtering another."""
    for level in ("team", "unit_head", "zonal_head", "bcm"):
        spec = relations.registry.resolve("advisor", level)
        assert spec is not None, level
        assert spec.column is hierarchy.column_for(level), level


def test_narrative_can_pluralise_every_level():
    from app.llm.narrative import _LEVEL_PLURAL

    for level in hierarchy.LEVEL_LABELS:
        assert level in _LEVEL_PLURAL, level


# ---------------------------------------------------------------------
# Dead levels removed
# ---------------------------------------------------------------------

def test_the_dead_unit_level_is_gone():
    """Advisor.unit has zero production rows and the ETL never writes
    it — the level could not answer anything."""
    assert "unit" not in hierarchy.CHAIN
    assert "unit" not in hierarchy.ATTRIBUTE_LEVELS
    assert hierarchy.column_for("unit") is None


def test_office_is_the_single_canonical_key_for_that_column():
    """`office` and `business_center` were two entity types over one
    column, with two gazetteer caches that could disagree."""
    assert hierarchy.column_for("office") is Advisor.office
    assert hierarchy.canonical_level("business_center") == "office"
    assert "business_center" not in hierarchy.LEVEL_COLUMNS


def test_legacy_level_names_still_resolve():
    """A stored QueryIR or an older API client must not break."""
    assert hierarchy.column_for("business_center") is Advisor.office
    assert hierarchy.is_valid_level("business_center")
    assert hierarchy.label_for("business_center") == "Office"


# ---------------------------------------------------------------------
# Scope expansion — one definition for filtering/breakdown/comparison
# ---------------------------------------------------------------------

def test_scope_filter_exists_for_every_level():
    for level in hierarchy.CHAIN + hierarchy.ATTRIBUTE_LEVELS:
        assert hierarchy.scope_filter(level, "x") is not None


def test_scope_filter_rejects_an_unknown_level():
    with pytest.raises(ValueError):
        hierarchy.scope_filter("not_a_level", "x")


def test_scope_filter_targets_the_level_column(db_session):
    """The predicate must select on the level's OWN column — the single
    definition of "in scope" that Phase 4's aggregation will consume."""
    db_session.add(Advisor(wid=1, name="A", team="T1", rm="UH1", portfolio_lead="ZH1",
                           management_lead="BCM1"))
    db_session.add(Advisor(wid=2, name="B", team="T2", rm="UH2", portfolio_lead="ZH2",
                           management_lead="BCM2"))
    db_session.commit()

    for level, value, expected in (("team", "T1", "A"), ("unit_head", "UH2", "B"),
                                   ("zonal_head", "ZH1", "A"), ("bcm", "BCM2", "B")):
        rows = db_session.query(Advisor.name).filter(
            hierarchy.scope_filter(level, value)).all()
        assert [r.name for r in rows] == [expected], level


# ---------------------------------------------------------------------
# Region: mixed semantics, declared rather than silently carried
# ---------------------------------------------------------------------

def test_region_ambiguity_is_declared():
    """Phase 1 found Advisor.region holds a PERSON for master-sheet rows
    and a PLACE for the rest. Splitting it needs an ETL change, so the
    ambiguity is recorded where the level is declared — and the split is
    then confined to that one file."""
    assert "region" in hierarchy.AMBIGUOUS_LEVELS
    assert "region" in hierarchy.ATTRIBUTE_LEVELS
    note = hierarchy.AMBIGUOUS_LEVELS["region"]
    assert "geographic" in note.lower() or "place" in note.lower()


def test_region_is_not_a_chain_level():
    """It cannot nest while it means two things."""
    assert not hierarchy.is_chain_level("region")
