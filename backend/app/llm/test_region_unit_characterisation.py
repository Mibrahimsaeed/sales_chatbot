"""Characterisation of Region and Unit handling (written BEFORE M5).

Records the state M5 changes: `Advisor.region` and `Advisor.unit` are
real columns that the chatbot cannot reach, while the WORDS "region" and
"unit" are wired as aliases for entirely different levels —
`zonal_head` and `unit_head`, which hold people's names.

So "advisors in North" cannot resolve (no gazetteer), and "his unit"
means "his unit head". Both are recorded here, because M5 deliberately
breaks the second one and the break must be visible rather than
discovered.
"""

import pytest

from app.database.models import Advisor
from app.llm import advisor_resolver, entity_extractor, hierarchy
from app.llm.query_ir import Level


@pytest.fixture()
def db(db_session):
    db_session.add(Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana",
                           bm="Kaleem Ullah", rm="Kaleem Ullah", zm="Adeel Dogar", portfolio_lead="Adeel Dogar", office="Gulberg BC",
                           region="North", unit="Unit 1"))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


# ---------------------------------------------------------------------
# RETIRED (M5 landed).
#
# This file also recorded the pre-M5 state: region/unit were not levels,
# had no gazetteer, could not resolve a value, and the WORDS were aliases
# for zonal_head and unit_head. Those six assertions were written to fail
# once the split landed, and they did. They now live in
# test_region_unit_deprecation.py, asserted from the other direction.
# Keeping both would mean asserting the limitation still exists.
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# What must SURVIVE M5 (requirement 4)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("keyword", ["zonal head", "zonal heads", "zone head", "zone heads"])
def test_explicit_zonal_head_vocabulary(keyword):
    assert keyword in hierarchy.LEVEL_KEYWORDS["zonal_head"]


@pytest.mark.parametrize("keyword", ["unit head", "unit heads", "unit-head", "unit-heads"])
def test_explicit_unit_head_vocabulary(keyword):
    assert keyword in hierarchy.LEVEL_KEYWORDS["unit_head"]


def test_zm_and_bm_reverse_roles_are_declared():
    from app.llm import relations

    aliases = dict(relations.role_alias_pairs())
    assert aliases["zm"] == "zm"
    assert aliases["bm"] == "bm"


# ---------------------------------------------------------------------
# The level enumerations M5 must extend together
# ---------------------------------------------------------------------

def test_level_tables_agree_with_each_other():
    """Every level must appear in every table, or a query naming it
    resolves in one layer and falls over in the next. This invariant is
    what makes adding a level a data change; it must still hold after
    M5."""
    for level in hierarchy.HIERARCHY_LEVELS:
        assert level in hierarchy.LEVEL_COLUMNS, level
        assert level in hierarchy.LEVEL_LABELS, level
        assert level in hierarchy.LEVEL_KEYWORDS, level
        if level != "advisor":
            assert level in hierarchy.LEVEL_ENTITY_KEYS, level
            assert level in hierarchy.LEVEL_MATCH_KIND, level


def test_query_ir_level_literal_matches_the_hierarchy():
    """query_ir.Level is hand-written; it must list exactly the levels
    hierarchy declares, or the IR cannot express one of them."""
    assert set(hierarchy.HIERARCHY_LEVELS) <= set(Level.__args__)
