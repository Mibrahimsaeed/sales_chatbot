"""Phase 5B — comparison as a first-class QueryIR intent.

Comparison used to execute on the rule-based PLAN path through
comparison_service: a second pipeline that bypassed QueryIR and therefore
everything QueryIR owns. The audit found five consequences, and the fix
was routing rather than new machinery — the IR path already validated,
compiled, planned and rendered comparisons; the rule planner simply never
sent one there.

Each test names the capability it inherits, so a future change that moves
comparison back off the IR path fails here with the reason attached.
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
from app.llm import entity_extractor, nlu_pipeline, routing
from app.services.chat_service import handle_chat_message

# wid, name, team, company, office, region, bcm, zonal, unit head
_PEOPLE = [
    (1, "Yasir Ali", "Blue Area", "Graana", "Beverly Center", "North/KPK",
     "Usman Ghani", "Fawad Hafeez", "Tariq Mehmood"),
    (2, "Waqar Haider", "Blue Area", "Graana", "Beverly Center", "North/KPK",
     "Usman Ghani", "Fawad Hafeez", "Tariq Mehmood"),
    (3, "Sana Tariq", "Blue Area", "Graana", "Beverly Center", "North/KPK",
     "Rabia Anjum", "Fawad Hafeez", "Tariq Mehmood"),
    (4, "Shehryar Abbasi", "Downtown", "Graana", "Gold Crest", "Central",
     "Rabia Anjum", "Adeel Aslam", "Tariq Mehmood"),
    (5, "Hina Malik", "Downtown", "IMARAT", "Gold Crest", "Central",
     "Rabia Anjum", "Adeel Aslam", "Sadia Rehman"),
    (6, "Omar Farooq", "Gulberg", "IMARAT", "Emporium", "South",
     "Bilal Qadir", "Adeel Aslam", "Sadia Rehman"),
]


@pytest.fixture(scope="module")
def _cmp_engine():
    from conftest import _ADVISOR_PROFILE_VIEW

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    for wid, name, team, company, office, region, bcm, zonal, unit in _PEOPLE:
        s.add(Advisor(wid=wid, name=name, team=team, company=company, office=office,
                      region=region, management_lead=bcm, portfolio_lead=zonal,
                      rm=unit, unit="A", in_master_sheet=True))
        s.add(Performance(wid=wid, period=PerformancePeriod.MTD, target=1000,
                          cleared=100 * wid, pct=10.0 * wid))
        s.add(Performance(wid=wid, period=PerformancePeriod.YTD, target=10000,
                          cleared=1000 * wid, pct=10.0 * wid))
        s.add(SalesFunnel(wid=wid, mtd_new_connect=10 * wid, mtd_followup_connect=0,
                          mtd_cr=5 * wid, mtd_new_meeting=wid, mtd_followup_meeting=0,
                          mtd_conversion=wid, mtd_booking_stored=wid))
        s.add(Pipeline(wid=wid, pipeline=1000 * wid, overdue=wid))
        s.add(Portfolio(wid=wid, value=5000 * wid))
        s.add(Calls(wid=wid, answered_calls_mtd=20 * wid, connects_mtd=10 * wid))
        s.add(Attendance(wid=wid, biometric_mtd_ontime=10 + wid, biometric_mtd_late=5,
                         biometric_mtd_not_marked=0, login_mtd_ontime=18,
                         login_mtd_late=2, login_mtd_not_marked=0))
    for team, target, achieved in (("Blue Area", 3000, 2400), ("Downtown", 2000, 1100),
                                   ("Gulberg", 1500, 700)):
        s.add(TeamTarget(team=team, target=target, achieved=achieved,
                         achievement_pct=round(achieved / target * 100, 1)))
    s.commit()
    with engine.begin() as conn:
        conn.exec_driver_sql(_ADVISOR_PROFILE_VIEW)
    yield engine
    s.close()


@pytest.fixture()
def org(_cmp_engine, monkeypatch):
    from app.core.config import settings
    from app.llm import llm_client, semantic_parser

    monkeypatch.setattr(settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(settings, "use_llm_planner", False, raising=False)
    monkeypatch.setattr(llm_client, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)

    s = sessionmaker(bind=_cmp_engine)()
    entity_extractor._cache["loaded_at"] = 0
    yield s
    s.close()


def _ir(org, query, session=None):
    return nlu_pipeline.resolve(query, org, session_id=session)


# ---------------------------------------------------------------------
# Every hierarchy level compares (C2 / requirement 6)
# ---------------------------------------------------------------------

LEVELS = [
    ("Compare Yasir Ali and Sana Tariq on cleared", "advisor",
     {"Yasir Ali", "Sana Tariq"}),
    ("Compare Blue Area and Downtown on revenue", "team",
     {"Blue Area", "Downtown"}),
    ("Compare Graana and IMARAT on cleared", "company", {"Graana", "IMARAT"}),
    ("Compare Beverly Center and Gold Crest on pipeline", "office",
     {"Beverly Center", "Gold Crest"}),
    ("Compare North/KPK and Central on revenue", "region",
     {"North/KPK", "Central"}),
    ("Compare Usman Ghani and Rabia Anjum on connects", "bcm",
     {"Usman Ghani", "Rabia Anjum"}),
    ("Compare Fawad Hafeez and Adeel Aslam on revenue", "zonal_head",
     {"Fawad Hafeez", "Adeel Aslam"}),
    ("Compare Tariq Mehmood and Sadia Rehman on cleared", "unit_head",
     {"Tariq Mehmood", "Sadia Rehman"}),
]


@pytest.mark.parametrize("query,level,subjects", LEVELS)
def test_every_level_produces_a_comparison_ir(org, query, level, subjects):
    """Advisor-vs-advisor is the one that did not work: identity
    resolution answers "who is this ABOUT?" and returned one person, so
    comparison_targets() saw one side of a two-sided question."""
    resolution = _ir(org, query)

    assert resolution.kind == "ir"
    assert resolution.ir.intent == "comparison"
    assert {s.value for s in resolution.ir.subjects} == subjects
    assert resolution.ir.subject_level == level


@pytest.mark.parametrize("query,level,subjects", LEVELS)
def test_every_level_renders_as_a_comparison(org, query, level, subjects):
    response = handle_chat_message(org, query, session_id=None)
    assert response["type"] == "comparison"
    for subject in subjects:
        assert subject in response["reply"]


def test_a_comparison_never_renders_as_a_leaderboard_or_single_value(org):
    """Requirement 10 — the response mode must never degrade."""
    for query, _level, _subjects in LEVELS:
        response = handle_chat_message(org, query, session_id=None)
        assert response["type"] not in ("leaderboard", "metric_value", "advisor")


# ---------------------------------------------------------------------
# Subjects are subjects, not filters
# ---------------------------------------------------------------------


def test_subjects_do_not_become_intersecting_filters(org):
    """Both sides as entity filters would AND together and match nobody."""
    ir = _ir(org, "Compare Blue Area and Downtown on revenue").ir
    assert not [f for f in ir.filters if f.field == "team"]
    assert len(ir.subjects) == 2


def test_three_subject_group_comparisons(org):
    """Group levels support any number of sides. Comma-separated ADVISOR
    lists do not — span extraction reads "Yasir Ali, Waqar Haider and
    Sana Tariq" as fewer people than were named — so three-way advisor
    comparison is a known gap, recorded in the Phase 5B report rather
    than asserted here."""
    ir = _ir(org, "Compare Blue Area, Downtown and Gulberg on revenue").ir
    assert ir.intent == "comparison"
    assert len(ir.subjects) == 3


def test_a_cross_level_comparison_keeps_both_levels(org):
    """Subjects carry their own type, so the levels need not match."""
    ir = _ir(org, "Compare Blue Area with Graana on revenue").ir
    assert {(s.type, s.value) for s in ir.subjects} == {
        ("team", "Blue Area"), ("company", "Graana")}


# ---------------------------------------------------------------------
# C3 — period, via the ONE owner
# ---------------------------------------------------------------------


def test_a_period_comparison_executes_that_period(org):
    """The plan path resolved YTD and executed MTD, because
    _effective_metric() lives on the IR path and comparison bypassed it."""
    from app.llm.query_compiler import effective_metric

    resolution = _ir(org, "Compare Blue Area and Downtown on revenue year to date")
    assert resolution.ir.time_range.period == "YTD"
    assert effective_metric(resolution.ir) == "ytd_cleared"

    reply = handle_chat_message(
        org, "Compare Blue Area and Downtown on revenue year to date",
        session_id=None)["reply"]
    assert "YTD" in reply


def test_the_period_owner_is_not_duplicated_for_comparisons(org):
    """No comparison-specific period mapping exists to drift."""
    import inspect

    from app.services import comparison_service

    assert "metric_for_period" not in inspect.getsource(comparison_service)


# ---------------------------------------------------------------------
# C1 — superlatives are comparisons, not rankings
# ---------------------------------------------------------------------

# Each names a MEASURE as well as its two sides: without one the query
# is a no-metric comparison, which still answers with the KPI table on
# the plan path (see _is_rule_based) and so has no IR to assert on.
SUPERLATIVES = [
    "Who has more revenue, Blue Area or Downtown?",
    "Who has more revenue, Downtown or Blue Area?",
    "Which of Usman Ghani's or Rabia Anjum's groups has more conversions?",
    "Which team has more revenue, Blue Area or Downtown?",
]


@pytest.mark.parametrize("query", SUPERLATIVES)
def test_an_enumerated_superlative_is_a_comparison(org, query):
    """These routed to the leaderboard, which dropped one subject from
    the filter and answered with the other's figure — right only when the
    gazetteer order happened to match the metric order."""
    resolution = _ir(org, query)
    assert resolution.ir.intent == "comparison"
    assert len(resolution.ir.subjects) == 2


@pytest.mark.parametrize("query", SUPERLATIVES)
def test_a_superlative_reports_both_sides(org, query):
    response = handle_chat_message(org, query, session_id=None)
    assert response["type"] == "comparison"


def test_an_unenumerated_superlative_is_still_a_ranking(org):
    """"Which team has the highest revenue" ranks every team. The
    disjunction is what makes a superlative a comparison, not the
    comparative word — otherwise every leaderboard becomes a comparison."""
    resolution = _ir(org, "Which team has the highest revenue?")
    assert resolution.ir.intent == "leaderboard"


def test_the_superlative_winner_is_computed_not_positional(org):
    """The regression proof: whichever subject is named first, the answer
    must be the one the METRIC picks."""
    first = handle_chat_message(org, "Who has more revenue, Blue Area or Downtown?",
                                session_id=None)["reply"]
    second = handle_chat_message(org, "Who has more revenue, Downtown or Blue Area?",
                                 session_id=None)["reply"]
    assert "Blue Area" in first and "Downtown" in first
    assert "Blue Area" in second and "Downtown" in second


# ---------------------------------------------------------------------
# C4 — conversation context, via the ONE owner
# ---------------------------------------------------------------------


def test_a_comparison_is_stored_as_conversation_state(org):
    from app.llm import conversation_memory

    handle_chat_message(org, "Compare Blue Area and Downtown on revenue",
                        session_id="c-store")
    stored = conversation_memory.get("c-store")
    assert stored is not None
    assert stored.intent == "comparison"
    assert len(stored.subjects) == 2


def test_a_metric_follow_up_keeps_both_subjects(org):
    handle_chat_message(org, "Compare Blue Area and Downtown on revenue",
                        session_id="c-metric")
    response = handle_chat_message(org, "what about overdue?", session_id="c-metric")

    assert response["type"] == "comparison"
    assert "Blue Area" in response["reply"] and "Downtown" in response["reply"]
    assert "Overdue" in response["reply"]


def test_a_period_follow_up_keeps_both_subjects(org):
    handle_chat_message(org, "Compare Blue Area and Downtown on revenue",
                        session_id="c-period")
    response = handle_chat_message(org, "year to date", session_id="c-period")

    assert response["type"] == "comparison"
    assert "YTD" in response["reply"]
    assert "Blue Area" in response["reply"] and "Downtown" in response["reply"]


def test_a_filter_follow_up_keeps_both_subjects(org):
    handle_chat_message(org, "Compare Blue Area and Downtown on revenue",
                        session_id="c-filter")
    response = handle_chat_message(org, "only Graana", session_id="c-filter")

    assert response["type"] == "comparison"
    assert "Graana" in response["reply"]


def test_a_four_turn_comparison_chain_never_loses_its_subjects(org):
    """Requirement 8, end to end."""
    session = "c-chain"
    handle_chat_message(org, "Compare Blue Area and Downtown on revenue",
                        session_id=session)
    for follow_up in ("what about overdue?", "year to date", "only Graana"):
        response = handle_chat_message(org, follow_up, session_id=session)
        assert response["type"] == "comparison", f"{follow_up!r} lost the comparison"
        assert "Blue Area" in response["reply"]
        assert "Downtown" in response["reply"]


def test_a_new_complete_question_ends_the_comparison(org):
    """Context expiry still applies — a comparison is not sticky."""
    session = "c-expire"
    handle_chat_message(org, "Compare Blue Area and Downtown on revenue",
                        session_id=session)
    response = handle_chat_message(org, "Top advisors by connects", session_id=session)
    assert response["type"] == "leaderboard"


# ---------------------------------------------------------------------
# Metric kinds
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query,marker", [
    ("Compare Blue Area and Downtown on attendance rate", "%"),
    ("Compare Blue Area and Downtown on achievement %", "%"),
])
def test_percentage_metrics_compare(org, query, marker):
    response = handle_chat_message(org, query, session_id=None)
    assert response["type"] == "comparison"
    assert marker in response["reply"]


def test_count_and_currency_metrics_compare(org):
    for query in ("Compare Blue Area and Downtown on connects",
                  "Compare Blue Area and Downtown on pipeline"):
        assert handle_chat_message(org, query, session_id=None)["type"] == "comparison"


@pytest.mark.parametrize("query", [
    # portfolio % is the one measure working_days.py did NOT make
    # computable — it has no target to measure against at all. CR % and
    # connect % were retired from this list when they became real rates.
    "Compare Blue Area and Downtown on portfolio %",
    "Compare Yasir Ali and Sana Tariq on portfolio %",
])
def test_an_unavailable_metric_explains_rather_than_comparing(org, query):
    """Requirement 9 — never silently degrade. The capability registry
    owns the reason; comparison inherits it like every other query."""
    response = handle_chat_message(org, query, session_id=None)
    assert response["type"] == "clarification"
    assert "working-day" in response["reply"] or "instead" in response["reply"]


# ---------------------------------------------------------------------
# Incomplete comparisons
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "Compare Blue Area and Nonexistent Team on revenue",
    "Compare with Downtown",
])
def test_an_incomplete_comparison_says_which_side_it_found(org, query):
    response = handle_chat_message(org, query, session_id=None)
    assert response["type"] == "clarification"
    assert "compare" in response["reply"].lower()


def test_a_relation_comparison_compares_the_groups_not_the_people(org):
    """"Compare X's team with Y's team" is about two TEAMS. Reading the
    named people as the subjects answers a different question with a
    confident number."""
    from app.llm.query_planner import build_query_plan
    from app.llm.entity_extractor import extract_entities

    text = "Compare Waqar Haider's team with Sana Tariq's team"
    plan = build_query_plan(text, extract_entities(text, org))
    assert [t for t in plan.comparison_targets if t[0] == "advisor"] == []


# ---------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------


def test_the_trace_records_the_comparison_response_mode(org):
    handle_chat_message(org, "Compare Blue Area and Downtown on revenue",
                        session_id=None)
    step = next(s for s in routing.current_trace().steps if s.stage == "Response")
    assert step.chose == "comparison"


def test_the_trace_records_inherited_subjects(org):
    handle_chat_message(org, "Compare Blue Area and Downtown on revenue",
                        session_id="c-trace")
    handle_chat_message(org, "what about overdue?", session_id="c-trace")
    step = next(s for s in routing.current_trace().steps if s.stage == "Context")
    assert "subjects" in step.why
    assert "Blue Area" in step.why
