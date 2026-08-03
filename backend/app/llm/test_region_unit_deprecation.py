"""The Region/Unit alias split — deprecation record (M5).

M5 could not be fully backward compatible, and this file is the record
of exactly what changed, expressed as executable assertions rather than
prose that can rot.

WHAT CHANGED
    "region"  stopped meaning zonal_head  and now means region
    "unit"    stopped meaning unit_head   and now means unit

WHY
    Those words name a PLACE and a GROUP. The levels they pointed at hold
    a PERSON'S NAME. So "advisors in North" could never work — the answer
    was in Advisor.region and nothing could reach it — while "his unit"
    silently answered about his unit head. One word cannot mean both a
    region and the person who runs one.

WHY IT IS NECESSARY RATHER THAN MERELY NICE
    Region and Unit cannot become first-class levels while their only
    names are taken. Any alternative — a new synonym like "geo region",
    or disambiguating by context — either leaves the columns unreachable
    by the words users actually type, or reintroduces the guessing this
    programme has been removing.

WHAT IS PRESERVED (requirement 4)
    Every EXPLICIT manager phrasing: "unit head", "zonal head", "zone
    head", "division", "zone", plus the "bm"/"zm" role aliases. Only the
    two bare, genuinely ambiguous words moved.

MIGRATION IMPACT
    A user who typed "his unit" meaning "his unit head" now gets his
    unit. Both readings are legitimate English; the new one matches the
    word. Users who want the manager say "unit head", which has always
    worked and still does.
"""

import pytest

from app.database.models import Advisor
from app.llm import advisor_resolver, entity_extractor, hierarchy


@pytest.fixture()
def db(db_session):
    db_session.add(Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana",
                           bm="Kaleem Ullah", rm="Kaleem Ullah", zm="Adeel Dogar", portfolio_lead="Adeel Dogar", office="Gulberg BC",
                           region="North", unit="Unit 1"))
    db_session.add(Advisor(wid=2, name="Sana Tariq", team="Blue Area", company="Graana",
                           bm="Kaleem Ullah", rm="Kaleem Ullah", zm="Adeel Dogar", portfolio_lead="Adeel Dogar", office="Gulberg BC",
                           region="North", unit="Unit 1"))
    db_session.add(Advisor(wid=3, name="Imran Butt", team="Downtown", company="Agency21",
                           bm="Nadia Rehman", rm="Nadia Rehman", zm="Faisal Iqbal", portfolio_lead="Faisal Iqbal", office="Saddar BC",
                           region="South", unit="Unit 2"))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


# ---------------------------------------------------------------------
# The split itself
# ---------------------------------------------------------------------

def test_region_no_longer_routes_to_zonal_head():
    assert "region" not in hierarchy.LEVEL_KEYWORDS["zonal_head"]
    assert "region" in hierarchy.LEVEL_KEYWORDS["region"]


def test_unit_no_longer_routes_to_unit_head():
    """UPDATED BY PHASE 3: the `unit` LEVEL was removed — Advisor.unit has
    zero production rows and the ETL never writes it. M5's alias split is
    still honoured in that "unit" no longer means the unit HEAD; it simply
    names nothing now."""
    assert "unit" not in hierarchy.HIERARCHY_LEVELS


def test_region_and_unit_are_first_class_levels():
    """PHASE 3 removed the `unit` level (zero production rows) and made
    `region` a groupable ATTRIBUTE rather than a chain level — the
    verified chain has five levels and region is not one of them. It
    remains filterable and groupable, which is the capability M5 added."""
    for level in ("region",):
        assert level in hierarchy.ATTRIBUTE_LEVELS
        assert level in hierarchy.GROUP_LEVELS
        assert hierarchy.is_valid_level(level)
        assert hierarchy.column_for(level) is not None


def test_they_map_to_their_own_columns():
    assert hierarchy.column_for("region") is Advisor.region


def test_labels_read_naturally():
    assert hierarchy.label_for("region") == "Region"


# ---------------------------------------------------------------------
# Preserved manager vocabulary (requirement 4)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("keyword,level", [
    ("unit head", "unit_head"), ("unit heads", "unit_head"),
    ("unit-head", "unit_head"), ("division", "unit_head"),
    ("zonal head", "zonal_head"), ("zone head", "zonal_head"),
    ("zone", "zonal_head"), ("zonal-head", "zonal_head"),
])
def test_explicit_manager_vocabulary_survives(keyword, level):
    assert keyword in hierarchy.LEVEL_KEYWORDS[level]


def test_bm_and_zm_role_aliases_survive():
    from app.llm import relations

    aliases = dict(relations.role_alias_pairs())
    assert aliases["bm"] == "bm"
    assert aliases["zm"] == "zm"
    assert aliases["unit head"] == "unit_head"
    assert aliases["zonal head"] == "zonal_head"


def test_region_and_unit_are_not_reverse_roles():
    """They name groups, so "who is X's region" is not a manager
    question — no role aliases, and none may be added without a
    deliberate decision."""
    from app.llm import relations

    for level in ("region",):
        spec = relations.registry.resolve("advisor", level)
        assert spec is None or spec.role_aliases == ()


# ---------------------------------------------------------------------
# Values now resolve (the capability the split buys)
# ---------------------------------------------------------------------

def test_a_region_value_resolves(db):
    entities = entity_extractor.extract_entities("advisors in North", db)
    assert entities["region"] == "North"
    assert entities["regions"] == ["North"]



def test_gazetteers_are_populated(db):
    entity_extractor._refresh_cache(db)
    assert sorted(entity_extractor._cache["regions"]) == ["North", "South"]



def test_every_level_appears_in_every_table():
    for level in hierarchy.HIERARCHY_LEVELS:
        assert level in hierarchy.LEVEL_COLUMNS, level
        assert level in hierarchy.LEVEL_LABELS, level
        assert level in hierarchy.LEVEL_KEYWORDS, level
        assert level in hierarchy.PARENT_LEVEL, level
        if level != "advisor":
            assert level in hierarchy.LEVEL_ENTITY_KEYS, level
            assert level in hierarchy.LEVEL_MATCH_KIND, level


def test_query_ir_level_literal_matches_the_hierarchy():
    from app.llm.query_ir import Level

    assert set(hierarchy.HIERARCHY_LEVELS) <= set(Level.__args__)


def test_narrative_can_pluralise_every_level():
    from app.llm.narrative import _LEVEL_PLURAL

    for level in hierarchy.HIERARCHY_LEVELS:
        assert level in _LEVEL_PLURAL, level


def test_every_group_level_is_reachable_by_the_planner():
    """A level absent from GROUP_LEVEL_ORDER can never be found by
    group_entity(), so a query naming it silently fails to scope."""
    from app.llm import intent_catalog as cat

    assert set(cat.GROUP_LEVEL_ORDER) == set(hierarchy.GROUP_LEVELS)


def test_every_group_level_has_a_subject_gazetteer():
    from app.llm.ir_validator import _SUBJECT_GAZETTEERS

    assert set(_SUBJECT_GAZETTEERS) == set(hierarchy.GROUP_LEVELS)
