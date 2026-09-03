"""Per Capita: a group's figure spread over the people in it.

DERIVED, NOT MEASURED. Both operands are already on the row — the ranked
value and the Team Size column — so the division cannot disagree with the
two columns beside it, and nothing new is queried, calculated by the
model, or added to the ontology.

WHY THE LEVEL DECIDES, NOT JUST THE MEASURE. An advisor's own count over
their team's size is one person's work divided by their colleagues. That
is a real number and not a rate anybody asked for, so the column appears
for a TEAM, a Unit Head, a Zonal Head or a BCM, and never for an advisor.
"""

import pytest

from app.database.models import Advisor, Pipeline, SalesFunnel
from app.llm import conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import (
    BUNDLE_COLUMNS_KEY, _per_capita, handle_chat_message,
)

# Four advisors in one team, under one Unit Head / Zonal Head / BCM, so
# every group level counts the same four people and the arithmetic is
# checkable by hand.
PEOPLE = [(1, 4), (2, 3), (3, 2), (4, 1)]     # wid, meetings/CR/pipeline each


@pytest.fixture()
def org(db_session, monkeypatch):
    for wid, n in PEOPLE:
        db_session.add(Advisor(wid=wid, name=f"Person {wid}", team="Alpha",
                               company="Graana", rm="Uma", portfolio_lead="Zed",
                               management_lead="Bee", in_master_sheet=True))
        db_session.add(SalesFunnel(wid=wid, mtd_new_meeting=n, mtd_followup_meeting=0,
                                   mtd_cr=n, mtd_new_connect=n, mtd_followup_connect=0))
        db_session.add(Pipeline(wid=wid, pipeline=n * 10, overdue=0))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _cells(db, query):
    conversation_memory._store.clear()
    rows = handle_chat_message(db, query, session_id=None)["data"]
    return [row.get(BUNDLE_COLUMNS_KEY) or {} for row in rows]


# ---------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------

@pytest.mark.parametrize("total,size,expected", [
    (12, 8, 1.5),
    (16, 8, 2.0),
    (48, 8, 6.0),
    (0, 8, 0.0),
])
def test_the_division_is_the_stated_one(total, size, expected):
    assert _per_capita(total, size) == expected


@pytest.mark.parametrize("total,size", [(12, 0), (12, None), (None, 8), (None, None)])
def test_a_missing_or_zero_headcount_never_divides(total, size):
    """None, not zero: "no team to divide by" and "none per head" are
    different statements, and zero says the wrong one. Never raises."""
    assert _per_capita(total, size) is None


# ---------------------------------------------------------------------
# Where it appears
# ---------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "top teams by client registrations",
    "top teams by meetings",
    "top teams by pipeline",
])
def test_the_three_leaderboards_carry_it(query, org):
    cells = _cells(org, query)
    assert cells, f"no rows for {query!r}"
    for row in cells:
        assert "per_capita" in row, query
        assert row["per_capita"]["label"] == "Per Capita"


@pytest.mark.parametrize("level_query", [
    "top unit heads by meetings",
    "top zonal heads by meetings",
    "top bcms by meetings",
])
def test_every_supported_group_level_carries_it(level_query, org):
    cells = _cells(org, level_query)
    assert cells, f"no rows for {level_query!r}"
    assert all("per_capita" in row for row in cells), level_query


@pytest.mark.parametrize("query", [
    "top advisors by meetings",
    "top advisors by client registrations",
    "top advisors by pipeline",
])
def test_advisors_never_carry_it(query, org):
    """An advisor has no team size of their own to divide by."""
    for row in _cells(org, query):
        assert "per_capita" not in row, query


def test_an_unrelated_measure_does_not_gain_it(org):
    for row in _cells(org, "top teams by revenue"):
        assert "per_capita" not in row


# ---------------------------------------------------------------------
# The value on a real row
# ---------------------------------------------------------------------

def test_the_value_matches_the_two_columns_beside_it(org):
    """The point of deriving it from the row: it cannot disagree with the
    figures printed next to it."""
    for row in _cells(org, "top teams by meetings"):
        total = row["total_meetings"]["value"]
        size = row["team_size"]["value"]
        assert row["per_capita"]["value"] == pytest.approx(total / size)


def test_the_team_total_is_spread_over_its_four_people(org):
    """4 + 3 + 2 + 1 meetings over 4 advisors."""
    row = _cells(org, "top teams by meetings")[0]
    assert row["total_meetings"]["value"] == 10
    assert row["team_size"]["value"] == 4
    assert row["per_capita"]["value"] == pytest.approx(2.5)


def test_it_renders_to_two_decimals(org):
    row = _cells(org, "top teams by meetings")[0]
    assert row["per_capita"]["display"] == "2.50"


def test_pipeline_gains_a_headcount_only_where_it_is_divided(org):
    """Pipeline reports no Team Size of its own. It gains one at group
    level because Per Capita needs a denominator, and an ADVISOR pipeline
    ranking is left exactly as it was."""
    assert all("team_size" in row for row in _cells(org, "top teams by pipeline"))
    for row in _cells(org, "top advisors by pipeline"):
        assert "team_size" not in row and "per_capita" not in row
