"""Phase 8 — end-to-end semantic execution regressions.

Two defects found by tracing representative queries through every layer
(planner -> QueryIR -> SQL -> rows -> response). Both were invisible to
the intent layer, which was correct throughout: the planner chose
`group_metric`, and the query still answered about a different team.

The fixture deliberately contains a team whose name CONTAINS another
team's name. That is the condition under which the group-metric
compilation stops being equivalent, and no other fixture in the suite
has it — which is why 2,745 passing tests did not catch this.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.models import (
    Advisor, Attendance, Calls, Performance, PerformancePeriod, Pipeline,
    Portfolio, SalesFunnel, TeamTarget,
)
from app.database.session import Base
from app.llm import aggregation, entity_extractor, nlu_pipeline
from app.llm.query_compiler import compile_and_run
from app.services.chat_service import handle_chat_message

# "Blue Area" is a strict prefix of "Blue Area North". Both are real
# teams; a substring scope cannot tell them apart.
_PEOPLE = [
    (1, "Yasir Ali", "Blue Area", 900),
    (2, "Waqar Haider", "Blue Area", 800),
    (3, "Shehryar Abbasi", "Downtown", 600),
    (4, "North Person A", "Blue Area North", 5000),
    (5, "North Person B", "Blue Area North", 5000),
]

BLUE_AREA_TOTAL = 1700      # 900 + 800
BLUE_AREA_NORTH_TOTAL = 10000


@pytest.fixture(scope="module")
def _sem_engine():
    from conftest import _ADVISOR_PROFILE_VIEW

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    for wid, name, team, cleared in _PEOPLE:
        s.add(Advisor(wid=wid, name=name, team=team, company="Graana",
                      rm="Tariq Mehmood", portfolio_lead="Fawad Hafeez",
                      management_lead="Usman Ghani", office="Beverly Center",
                      region="North/KPK", unit="A", in_master_sheet=True))
        s.add(Performance(wid=wid, period=PerformancePeriod.MTD, target=1000,
                          cleared=cleared, pct=10.0))
        s.add(Performance(wid=wid, period=PerformancePeriod.YTD, target=10000,
                          cleared=cleared * 10, pct=10.0))
        s.add(SalesFunnel(wid=wid, mtd_new_connect=10, mtd_followup_connect=0,
                          mtd_cr=5, mtd_new_meeting=1, mtd_followup_meeting=0,
                          mtd_conversion=1, mtd_booking_stored=1))
        s.add(Pipeline(wid=wid, pipeline=1000, overdue=1))
        s.add(Portfolio(wid=wid, value=5000))
        s.add(Calls(wid=wid, answered_calls_mtd=20, connects_mtd=10))
        s.add(Attendance(wid=wid, biometric_mtd_ontime=15, biometric_mtd_late=5,
                         biometric_mtd_not_marked=0, login_mtd_ontime=18,
                         login_mtd_late=2, login_mtd_not_marked=0))
    for team in ("Blue Area", "Downtown", "Blue Area North"):
        s.add(TeamTarget(team=team, target=2000, achieved=1000, achievement_pct=50.0))
    s.commit()
    with engine.begin() as conn:
        conn.exec_driver_sql(_ADVISOR_PROFILE_VIEW)
    yield engine
    s.close()


@pytest.fixture()
def org(_sem_engine, monkeypatch):
    from app.core.config import settings
    from app.llm import llm_client, semantic_parser

    monkeypatch.setattr(settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(settings, "use_llm_planner", False, raising=False)
    monkeypatch.setattr(llm_client, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)

    s = sessionmaker(bind=_sem_engine)()
    entity_extractor._cache["loaded_at"] = 0
    yield s
    s.close()


# ---------------------------------------------------------------------
# F1 — the compiler scoped by SUBSTRING while every other layer
#      scoped by whole name
# ---------------------------------------------------------------------


def test_a_group_metric_answers_about_that_group_only(org):
    """The failure: "Blue Area revenue" compiled to
    `team LIKE '%Blue Area%' GROUP BY team`, which partitions into TWO
    groups. The ranking operators the group-metric compilation relies on
    being no-ops stopped being no-ops, and the reply read
    "Blue Area North has 10,000 ... ranking 1st of 2 teams"."""
    response = handle_chat_message(org, "Blue Area revenue", session_id=None)

    assert f"{BLUE_AREA_TOTAL:,}" in response["reply"]
    assert "Blue Area North" not in response["reply"]
    assert response["type"] == "metric_value"


def test_a_group_metric_produces_exactly_one_partition(org):
    """The IR-level invariant behind the equivalence: one named group
    must compile to one row, or ORDER BY and LIMIT stop being no-ops."""
    resolution = nlu_pipeline.resolve("Blue Area revenue", org, session_id=None)
    rows = compile_and_run(org, resolution.ir, offset=0)

    assert len(rows) == 1
    assert rows[0]["name"] == "Blue Area"
    assert rows[0]["value"] == BLUE_AREA_TOTAL


def test_a_scoped_leaderboard_excludes_the_lookalike_team(org):
    """Same root cause on the ranking path: the members of "Blue Area
    North" were ranked inside "Blue Area"."""
    resolution = nlu_pipeline.resolve("Top advisors in Blue Area by revenue",
                                      org, session_id=None)
    names = {r["name"] for r in compile_and_run(org, resolution.ir, offset=0)}

    assert names == {"Yasir Ali", "Waqar Haider"}


def test_the_compiler_and_the_aggregator_agree_on_scope(org):
    """hierarchy.scope_filter calls itself "THE one definition of 'in
    scope'". The compiler built its own, and the two disagreed."""
    resolution = nlu_pipeline.resolve("Blue Area revenue", org, session_id=None)
    compiled = compile_and_run(org, resolution.ir, offset=0)[0]["value"]
    aggregated = aggregation.metric_value(org, "team", "Blue Area", "mtd_cleared")

    assert compiled == aggregated == BLUE_AREA_TOTAL


def test_the_longer_team_is_still_reachable(org):
    """The fix must not make the containing name unaddressable."""
    assert aggregation.metric_value(org, "team", "Blue Area North",
                                    "mtd_cleared") == BLUE_AREA_NORTH_TOTAL


def test_an_ir_equality_filter_compiles_as_equality(org):
    """The contract this broke: the IR said operator="=" and the compiler
    emitted containment."""
    resolution = nlu_pipeline.resolve("Blue Area revenue", org, session_id=None)
    team_filters = [f for f in resolution.ir.filters if f.field == "team"]

    assert team_filters and all(f.operator == "=" for f in team_filters)
    assert len(compile_and_run(org, resolution.ir, offset=0)) == 1


# ---------------------------------------------------------------------
# F2 — a single-value reply did not state the scope it applied
# ---------------------------------------------------------------------


def test_a_narrowed_single_value_states_the_narrowing(org):
    """"Show Downtown pipeline" then "now only Graana" returned the same
    sentence twice. Both filters WERE applied; an identical reply is
    indistinguishable from a dropped filter."""
    handle_chat_message(org, "Show Downtown pipeline", session_id="narrow")
    response = handle_chat_message(org, "Now only Graana", session_id="narrow")

    assert "Graana" in response["reply"]


def test_an_unnarrowed_single_value_stays_clean(org):
    """The subject's own filter is not repeated back: "Blue Area has ...
    filtered by team = Blue Area" is noise."""
    response = handle_chat_message(org, "Blue Area revenue", session_id=None)
    assert "filtered by" not in response["reply"]


# ---------------------------------------------------------------------
# Semantics preserved across the other three families
# ---------------------------------------------------------------------


def test_an_advisor_metric_is_never_a_ranking(org):
    response = handle_chat_message(org, "Revenue of Yasir Ali", session_id=None)
    assert response["type"] == "advisor_metric"
    assert "ranking" not in response["reply"]
    assert "900" in response["reply"]


def test_a_comparison_keeps_both_subjects_and_their_values(org):
    resolution = nlu_pipeline.resolve("Compare Blue Area and Downtown on revenue",
                                      org, session_id=None)
    rows = {r["name"]: r["value"] for r in compile_and_run(org, resolution.ir, offset=0)}

    assert rows == {"Blue Area": BLUE_AREA_TOTAL, "Downtown": 600}


@pytest.mark.parametrize("query,direction,limit", [
    ("Top advisors by revenue", "desc", 10),
    ("Top 2 advisors by revenue", "desc", 2),
    ("Bottom 2 advisors by revenue", "asc", 2),
])
def test_a_leaderboard_keeps_its_direction_and_limit(org, query, direction, limit):
    ir = nlu_pipeline.resolve(query, org, session_id=None).ir
    assert ir.sort.direction == direction
    assert ir.limit == limit
    assert len(compile_and_run(org, ir, offset=0)) <= limit
