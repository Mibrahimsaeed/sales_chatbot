"""hierarchy.py is the single mapping every other layer (entity extraction,
query_planner, query_compiler, ir_validator, response_formatter) reads from
instead of hardcoding team/company/unit_head/zonal_head/business_center
branching — these tests lock in the mapping itself, since a mistake here
would silently corrupt every layer built on top of it."""

from app.database.models import Advisor
from app.llm import hierarchy


def test_every_group_level_has_a_column():
    for level in hierarchy.GROUP_LEVELS:
        assert hierarchy.column_for(level) is not None


def test_advisor_level_has_a_column_too():
    assert hierarchy.column_for("advisor") is not None


def test_unknown_level_returns_none():
    """UPDATED BY M5: "region" used to be the example of a NON-level and
    is now a real one, so the example moved to a field that genuinely
    isn't part of the hierarchy."""
    assert hierarchy.column_for("attendance_status") is None
    assert hierarchy.column_for("portfolio_lead") is None
    assert not hierarchy.is_valid_level("attendance_status")


def test_confirmed_column_mapping():
    """Locks in the confirmed mapping: unit_head -> bm, zonal_head -> zm,
    business_center -> office (asked and confirmed during implementation,
    not derivable from the code otherwise)."""
    # REBOUND BY PHASE 3 to the columns Phase 1 verified.
    assert hierarchy.column_for("unit_head") is Advisor.rm
    assert hierarchy.column_for("zonal_head") is Advisor.portfolio_lead
    assert hierarchy.column_for("bcm") is Advisor.management_lead
    # legacy name still resolves, to the attribute it always meant
    assert hierarchy.column_for("business_center") is Advisor.office
    assert hierarchy.column_for("team") is Advisor.team
    assert hierarchy.column_for("company") is Advisor.company


def test_new_group_levels_are_the_generically_bound_ones():
    """UPDATED BY M5, which added region and unit.

    NEW_GROUP_LEVELS is the set that gets the query compiler's generic
    advisor-column rollup instead of a per-metric binding — the reason a
    new level needs no ontology changes. team and company stay out
    because they have their own explicit, independently-tested bindings
    (TeamTarget among them) that must not be re-routed."""
    assert set(hierarchy.NEW_GROUP_LEVELS) == {
        "unit_head", "zonal_head", "bcm", "office", "region",
    }
    assert set(hierarchy.NEW_GROUP_LEVELS).issubset(hierarchy.GROUP_LEVELS)
    assert "team" not in hierarchy.NEW_GROUP_LEVELS
    assert "company" not in hierarchy.NEW_GROUP_LEVELS


def test_labels_are_human_readable():
    assert hierarchy.label_for("unit_head") == "Unit Head"
    assert hierarchy.label_for("zonal_head") == "Zonal Head"
    assert hierarchy.label_for("bcm") == "BCM"
    assert hierarchy.label_for("office") == "Office"
    assert hierarchy.label_for(None) == "value"


def test_match_kind_is_advisor_for_person_levels_and_team_for_group_levels():
    assert hierarchy.match_kind_for("unit_head") == "advisor"
    assert hierarchy.match_kind_for("zonal_head") == "advisor"
    assert hierarchy.match_kind_for("bcm") == "advisor"   # a BCM is a person
    assert hierarchy.match_kind_for("office") == "team"   # an office is a place
    assert hierarchy.match_kind_for("company") == "company"


def test_level_entity_keys_cover_every_group_level():
    assert set(hierarchy.LEVEL_ENTITY_KEYS.keys()) == set(hierarchy.GROUP_LEVELS)
    assert hierarchy.LEVEL_ENTITY_KEYS["company"] == "companies"   # irregular plural
    assert hierarchy.LEVEL_ENTITY_KEYS["unit_head"] == "unit_heads"


# ---- Phase 2: broadened synonyms + parent pointer ----

def test_new_level_synonyms_are_present():
    """UPDATED BY M5 — the alias split.

    "unit" and "region" no longer point at the manager levels; they name
    the levels of the same name. Everything else here is unchanged, which
    is the point: only the two genuinely ambiguous bare words moved.
    See test_region_unit_deprecation.py for the full before/after."""
    assert "division" in hierarchy.LEVEL_KEYWORDS["unit_head"]
    assert "zone" in hierarchy.LEVEL_KEYWORDS["zonal_head"]
    # PHASE 3: "business center"/"center"/"branch" name the PLACE, which
    # is the `office` attribute; the PERSON who runs it is `bcm`.
    assert "center" in hierarchy.LEVEL_KEYWORDS["office"]
    assert "branch" in hierarchy.LEVEL_KEYWORDS["office"]
    assert "business center" in hierarchy.LEVEL_KEYWORDS["office"]
    assert "unit head" in hierarchy.LEVEL_KEYWORDS["unit_head"]
    assert "bcm" in hierarchy.LEVEL_KEYWORDS["bcm"]
    assert "business center manager" in hierarchy.LEVEL_KEYWORDS["bcm"]
    # M5's alias split survives the rebind
    assert "region" in hierarchy.LEVEL_KEYWORDS["region"]
    assert "region" not in hierarchy.LEVEL_KEYWORDS["zonal_head"]


def test_synonyms_do_not_leak_into_unrelated_levels():
    """rm/portfolio_lead/management_lead and team/company/advisor stay
    untouched — only the 3 new levels got broader synonyms (per the
    explicit decision to keep those separate)."""
    assert hierarchy.LEVEL_KEYWORDS["team"] == ["team", "teams"]
    assert hierarchy.LEVEL_KEYWORDS["company"] == ["company", "companies"]
    assert hierarchy.LEVEL_KEYWORDS["advisor"] == ["advisor", "advisors", "agent", "agents"]


def test_parent_level_chain_matches_the_documented_hierarchy():
    """REBOUND BY PHASE 3 to the verified chain. PARENT_LEVEL is now
    DERIVED from hierarchy.CHAIN, so this asserts the chain itself."""
    assert hierarchy.PARENT_LEVEL["team"] is None
    assert hierarchy.PARENT_LEVEL["unit_head"] == "team"
    assert hierarchy.PARENT_LEVEL["zonal_head"] == "unit_head"
    assert hierarchy.PARENT_LEVEL["bcm"] == "zonal_head"
    assert hierarchy.PARENT_LEVEL["advisor"] == "bcm"
    # attributes do not nest
    for attribute in hierarchy.ATTRIBUTE_LEVELS:
        assert hierarchy.PARENT_LEVEL[attribute] is None



def test_parent_chain_is_a_single_walk_to_the_root():
    """Structural, so an inserted level cannot orphan another."""
    seen = set()
    level = hierarchy.CHAIN[-1]
    while level is not None:
        assert level not in seen, f"cycle at {level}"
        seen.add(level)
        level = hierarchy.PARENT_LEVEL[level]
    assert seen == set(hierarchy.CHAIN)


