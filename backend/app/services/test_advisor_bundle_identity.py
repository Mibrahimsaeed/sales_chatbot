"""Phase 36 — a name is not an identifier, and a bundle column proved it.

"connects of all advisors" RAISED:

    sqlalchemy.exc.MultipleResultsFound: Multiple rows were found when
    exactly one was required

The bundle enrichment looked each row's companions up by NAME —
`aggregation.metric_value(db, "advisor", row["name"], key)` — and
`scope_filter("advisor", …)` matches `Advisor.name`, which is not
unique. Five names are shared in the production master sheet, and
`metric_value` ends in `.scalar()`, which raises on the second row.

THE CRASH WAS THE LUCKY OUTCOME. Had the query returned one row, the
column would have been filled from whichever duplicate the database
happened to pick — one person's answered calls printed beside another
person's connects, on a line carrying a third person's name, with
nothing to show for it. That is what these tests exist to prevent, which
is why they assert per-wid values rather than merely that nothing
raises.

Group levels were never affected: a BCM row IS a distinct value of
`management_lead`, so the name addresses exactly one group. Only the
leaf level addresses a PERSON, and only a wid does that.

The fix reuses advisor_service.get_advisor_metric, whose own docstring
says it is "Keyed by WID, never by name" for this exact reason.
"""

import pytest

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import (
    advisor_resolver, conversation_memory, entity_extractor, narrative,
    semantic_parser,
)
from app.services import advisor_service
from app.services.chat_service import (
    BUNDLE_COLUMNS_KEY, handle_chat_message, handle_show_more,
)

# TWO PAIRS of same-named people, with different figures on purpose.
# "Sana Iqbal" appears twice and "Bilal Ahmed" three times, so a
# name-keyed lookup raises on the first and could silently pick any of
# three on the second.
#
# wid, name,          connects, answered
PEOPLE = [
    (1, "Sana Iqbal",   900, 400),
    (2, "Sana Iqbal",   800, 100),
    (3, "Bilal Ahmed",  700, 350),
    (4, "Bilal Ahmed",  600, 120),
    (5, "Bilal Ahmed",  500, 250),
    (6, "Unique Person", 400, 200),
]
BY_WID = {wid: (connects, answered) for wid, _n, connects, answered in PEOPLE}


@pytest.fixture()
def db(db_session, monkeypatch):
    for wid, name, connects, answered in PEOPLE:
        db_session.add(Advisor(wid=wid, name=name, team="Alpha", company="Graana",
                               rm="Unit One", portfolio_lead="Zonal One",
                               management_lead=f"Bcm {wid}", in_master_sheet=True))
        db_session.add(Calls(wid=wid, connects_mtd=connects, connects_daily=0,
                             answered_calls_mtd=answered, answered_calls_daily=0))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=connects,
                                   mtd_followup_connect=0, mtd_cr=1))
        db_session.add(Pipeline(wid=wid, pipeline=connects, overdue=0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=connects))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


def _ask(db, text, session="p36"):
    conversation_memory._store.pop(session, None)
    return handle_chat_message(db, text, session_id=session)


# ---------------------------------------------------------------------
# It answers at all
# ---------------------------------------------------------------------


def test_an_advisor_leaderboard_does_not_raise_on_duplicate_names(db):
    """The reported crash, as a test. Two people share a name, so the
    old name-keyed lookup raised MultipleResultsFound on the first row
    that hit one."""
    response = _ask(db, "connects of all advisors")
    assert response["type"] == "leaderboard"
    assert len(response["data"]) == len(PEOPLE)


def test_duplicate_names_remain_separate_rows(db):
    """Distinct people are distinct rows. Deduplicating by name would
    hide someone, which is a different wrong answer from the crash."""
    rows = _ask(db, "connects of all advisors")["data"]
    assert sorted(r["wid"] for r in rows) == sorted(BY_WID)
    assert [r["name"] for r in rows].count("Bilal Ahmed") == 3


# ---------------------------------------------------------------------
# Every column belongs to the person on that line
# ---------------------------------------------------------------------


def test_each_row_gets_its_own_wids_companion_values(db):
    """THE guarantee. Asserted per wid against the fixture, so a column
    filled from a same-named neighbour fails here even though it would
    look entirely plausible on screen."""
    for row in _ask(db, "connects of all advisors")["data"]:
        connects, answered = BY_WID[row["wid"]]
        assert row["value"] == connects
        assert row[BUNDLE_COLUMNS_KEY]["answered_calls"]["value"] == answered


def test_the_companion_matches_the_wid_keyed_service(db):
    """Same numbers as the single-person reply for that same wid — one
    owner, so a person's row and their own answer cannot disagree.

    TEAM SIZE IS EXEMPT, and deliberately so. The `team_size` metric read
    at advisor level is 1 — its advisor binding is literal(1), so summing
    one advisor row gives one. That is what the wid-keyed service returns
    and it is the wrong answer to "Team Size": the size meant is the
    person's TEAM. It is asserted against the team's headcount below
    instead, which is the same owner every other headcount comes from.
    """
    for row in _ask(db, "connects of all advisors")["data"]:
        for key, cell in row[BUNDLE_COLUMNS_KEY].items():
            if key == "team_size":
                continue
            assert cell["value"] == advisor_service.get_advisor_metric(
                db, row["wid"], key), f"wid={row['wid']} {key}"


def test_same_named_people_get_different_columns(db):
    """The two Sana Iqbals answered 400 and 100. If the lookup were still
    keyed by name, both rows would carry the same figure — which is what
    a reader would never be able to spot."""
    rows = {r["wid"]: r for r in _ask(db, "connects of all advisors")["data"]}
    assert rows[1][BUNDLE_COLUMNS_KEY]["answered_calls"]["value"] == 400
    assert rows[2][BUNDLE_COLUMNS_KEY]["answered_calls"]["value"] == 100


def test_the_displayed_name_is_unchanged(db):
    """Identity moved to the wid; the label the user reads did not."""
    names = {r["wid"]: r["name"] for r in _ask(db, "connects of all advisors")["data"]}
    assert names == {wid: name for wid, name, _c, _a in PEOPLE}


# ---------------------------------------------------------------------
# Group levels are untouched by the change
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query,level", [
    ("connects of all BCMs", "bcm"),
    ("connects of all zonal heads", "zonal_head"),
    ("connects of all unit heads", "unit_head"),
])
def test_group_levels_still_resolve_by_their_group_value(db, query, level):
    """A group row IS a distinct column value, so the name addresses it
    exactly — that branch was never at risk and must not change."""
    from app.llm import aggregation

    for row in _ask(db, query)["data"]:
        for key, cell in row[BUNDLE_COLUMNS_KEY].items():
            assert cell["value"] == aggregation.metric_value(
                db, level, row["name"], key)


def test_pagination_still_reaches_every_advisor_once(db):
    session = "walk"
    conversation_memory._store.pop(session, None)
    response = handle_chat_message(db, "connects of all advisors", session_id=session)
    wids = [r["wid"] for r in response["data"]]
    while response.get("has_more"):
        response = handle_show_more(db, session)
        wids += [r["wid"] for r in response["data"]]
    assert sorted(wids) == sorted(BY_WID)


def test_a_single_person_query_is_unchanged(db):
    """The wid-keyed service is the one this path already used, so the
    person reply is untouched."""
    response = _ask(db, "connects of Unique Person")
    assert response["type"] == "advisor_metric"
    assert "400" in response["reply"]


def test_team_size_is_the_advisors_team_not_one(db):
    """The whole reason team_size is exempt above: the per-advisor read
    is 1 for everybody, which renders as a plausible column of ones."""
    from app.llm import aggregation

    rows = _ask(db, "connects of all advisors")["data"]
    assert rows, "no rows to check"
    for row in rows:
        cell = row[BUNDLE_COLUMNS_KEY]["team_size"]["value"]
        assert cell == aggregation.headcount(db, "team", row["team"])
        assert cell != 1.0 or aggregation.headcount(db, "team", row["team"]) == 1
