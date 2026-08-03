"""Phase 3 — breakdowns walk the declared chain (task 7).

Before this, `get_level_breakdown` grouped EVERY level's advisors by
team: a fixed 2-hop (level -> team -> advisor) regardless of what sat
between. A Unit Head breakdown therefore skipped Zonal Head and BCM
entirely and showed teams, which under the verified chain are ABOVE the
Unit Head, not below.

Now the nesting level is the chain's child of whatever was asked about,
so each level breaks down into the thing it actually contains.
"""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import hierarchy
from app.services.hierarchy_service import (
    get_level_breakdown, get_level_flat_list, nesting_level,
)


@pytest.fixture()
def org(db_session):
    """One Team -> one Unit Head -> two Zonal Heads -> three BCMs ->
    four advisors, so every edge of the chain has something to nest."""
    people = [
        (1, "Adv One",   "Team North", "UH Ali",  "ZH North", "BCM A"),
        (2, "Adv Two",   "Team North", "UH Ali",  "ZH North", "BCM B"),
        (3, "Adv Three", "Team North", "UH Ali",  "ZH South", "BCM C"),
        (4, "Adv Four",  "Team North", "UH Ali",  "ZH South", "BCM C"),
    ]
    for wid, name, team, rm, pl, ml in people:
        db_session.add(Advisor(wid=wid, name=name, team=team, company="Graana",
                               rm=rm, portfolio_lead=pl, management_lead=ml))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=10, mtd_followup_connect=0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=1000, cleared=500))
    db_session.commit()
    return db_session


# ---------------------------------------------------------------------
# The nesting level comes from the chain
# ---------------------------------------------------------------------

@pytest.mark.parametrize("level,expected_child", [
    ("team", "unit_head"),
    ("unit_head", "zonal_head"),
    ("zonal_head", "bcm"),
    ("bcm", "advisor"),
])
def test_nesting_level_is_the_chains_child(level, expected_child):
    assert nesting_level(level) == expected_child


def test_an_attribute_nests_by_the_leaf():
    """company/office/region have no child in the chain, so grouping by
    the leaf is the only meaningful nesting."""
    for attribute in hierarchy.ATTRIBUTE_LEVELS:
        assert nesting_level(attribute) == "advisor"


# ---------------------------------------------------------------------
# Breakdowns end to end
# ---------------------------------------------------------------------

def test_a_unit_head_breaks_down_into_zonal_heads(org):
    breakdown = get_level_breakdown(org, "unit_head", "UH Ali")

    assert breakdown["nested_by"] == "zonal_head"
    assert breakdown["nested_by_label"] == "Zonal Head"
    assert {g["team"] for g in breakdown["teams"]} == {"ZH North", "ZH South"}
    assert breakdown["advisors"] == 4


def test_a_zonal_head_breaks_down_into_bcms(org):
    breakdown = get_level_breakdown(org, "zonal_head", "ZH South")

    assert breakdown["nested_by"] == "bcm"
    assert {g["team"] for g in breakdown["teams"]} == {"BCM C"}
    assert breakdown["advisors"] == 2


def test_a_bcm_breaks_down_into_advisors(org):
    breakdown = get_level_breakdown(org, "bcm", "BCM C")

    assert breakdown["nested_by"] == "advisor"
    assert {g["team"] for g in breakdown["teams"]} == {"Adv Three", "Adv Four"}


def test_a_team_breaks_down_into_unit_heads(org):
    breakdown = get_level_breakdown(org, "team", "Team North")

    assert breakdown["nested_by"] == "unit_head"
    assert {g["team"] for g in breakdown["teams"]} == {"UH Ali"}


def test_no_intermediate_level_is_skipped(org):
    """The defect this replaces: every level used to nest by team, which
    jumped over whatever sat between."""
    for level, value in (("team", "Team North"), ("unit_head", "UH Ali"),
                         ("zonal_head", "ZH North")):
        breakdown = get_level_breakdown(org, level, value)
        assert breakdown["nested_by"] == hierarchy.child_of(level), level


def test_group_counts_add_up_to_the_advisor_total(org):
    breakdown = get_level_breakdown(org, "unit_head", "UH Ali")
    counted = sum(g["advisor_count"] for g in breakdown["teams"])
    assert counted == breakdown["advisors"]


# ---------------------------------------------------------------------
# Flat opt-in still bypasses nesting
# ---------------------------------------------------------------------

def test_the_flat_list_is_ungrouped(org):
    flat = get_level_flat_list(org, "unit_head", "UH Ali")
    assert {a["name"] for a in flat["advisor_list"]} == {
        "Adv One", "Adv Two", "Adv Three", "Adv Four"
    }


# ---------------------------------------------------------------------
# Scope expansion is the same for every consumer
# ---------------------------------------------------------------------

def test_breakdown_scope_matches_scope_filter(org):
    """Breakdown, filtering and (Phase 4) aggregation must agree on who
    is in scope — they now share hierarchy.scope_filter."""
    for level, value in (("unit_head", "UH Ali"), ("zonal_head", "ZH North"),
                         ("bcm", "BCM C"), ("team", "Team North")):
        breakdown = get_level_breakdown(org, level, value)
        direct = org.query(Advisor).filter(
            hierarchy.scope_filter(level, value),
            Advisor.in_master_sheet.is_(True),
        ).count()
        assert breakdown["advisors"] == direct, level
