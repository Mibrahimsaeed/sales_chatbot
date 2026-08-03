"""Phase 2 — subject-level precedence regression tests.

The defect these lock down was invisible to every other kind of test.
Entity resolution was right, the metric was right, the period was right,
the SQL was right, and the number shown was a real number from the right
scope — the answer was simply about a different subject than the one the
user named. Only an assertion on the LEVEL catches that.

    "What is Downtown's pipeline value this month?"
      before: "Shehryar Abbasi has 3,500 ... 1st of 2 advisors shown"
      after:  Downtown's figure

Two layers are tested: `subject_level.decide()` in isolation (pure, so
the precedence itself is pinned without a database), and the whole
pipeline (so a future caller cannot bypass the owner and reintroduce the
defect one layer down).
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
from app.llm import entity_extractor, nlu_pipeline, routing, subject_level


# ---------------------------------------------------------------------
# The precedence itself — pure, no database
# ---------------------------------------------------------------------


def test_an_explicit_level_word_beats_everything():
    d = subject_level.decide(
        level_word="team", entity_level="company", entity_value="Graana",
        relation_level="advisor", metric_default="advisor",
    )
    assert d.level == "team"
    assert d.source == "level_word"
    # every loser is recorded, which is what makes a mis-route diagnosable
    assert len(d.rejected) == 3


def test_a_grounded_subject_beats_the_metric_default():
    """THE Phase 2 fix. This assertion inverted: `primary_level` used to
    win here, which is how "Downtown's pipeline" became an advisor list."""
    d = subject_level.decide(
        level_word=None, entity_level="team", entity_value="Downtown",
        metric_default="advisor",
    )
    assert d.level == "team"
    assert d.source == "entity"
    assert any("metric_default" in src for src, _ in d.rejected)


def test_a_ranking_makes_the_named_group_a_scope_not_the_subject():
    """"Top 5 in Blue Area by revenue" enumerates its members. Ranking a
    single team against itself is not a question anyone asks."""
    d = subject_level.decide(
        level_word=None, entity_level="team", entity_value="Blue Area",
        metric_default="advisor", has_ranking=True,
    )
    assert d.level == "advisor"
    assert d.source == "metric_default"
    reason = next(r for src, r in d.rejected if src == "entity=team")
    assert "SCOPE" in reason


def test_a_relation_beats_the_metric_default():
    d = subject_level.decide(
        level_word=None, entity_level=None, relation_level="team",
        metric_default="advisor",
    )
    assert d.level == "team"
    assert d.source == "relation"


def test_the_metric_default_still_applies_with_no_subject():
    """Requirement 4 — "What is revenue?" must not change behaviour."""
    d = subject_level.decide(
        level_word=None, entity_level=None, metric_default="advisor",
    )
    assert d.level == "advisor"
    assert d.source == "metric_default"
    assert d.rejected == ()


def test_the_decision_is_pure():
    """Same inputs, same decision — routing must be reproducible."""
    kwargs = dict(level_word=None, entity_level="team", entity_value="Downtown",
                  metric_default="advisor")
    assert subject_level.decide(**kwargs) == subject_level.decide(**kwargs)


def test_every_decision_explains_itself():
    for kwargs in (
        dict(level_word="team"),
        dict(entity_level="region", entity_value="South"),
        dict(relation_level="team"),
        dict(metric_default="advisor"),
        dict(),
    ):
        d = subject_level.decide(**kwargs)
        assert d.why, f"{kwargs} produced no reason"
        assert d.level


# ---------------------------------------------------------------------
# End to end, every hierarchy level
# ---------------------------------------------------------------------

PEOPLE = [
    (1, "Yasir Ali", "Blue Area", "Graana", "Tariq Mehmood", "Fawad Hafeez",
     "Usman Ghani", "Beverly Center", "North/KPK"),
    (2, "Waqar Haider", "Blue Area", "Graana", "Tariq Mehmood", "Fawad Hafeez",
     "Usman Ghani", "Beverly Center", "North/KPK"),
    (3, "Shehryar Abbasi", "Downtown", "Graana", "Tariq Mehmood", "Fawad Hafeez",
     "Rabia Anjum", "Gold Crest", "Central"),
    (4, "Hina Malik", "Downtown", "Graana", "Tariq Mehmood", "Adeel Aslam",
     "Rabia Anjum", "Gold Crest", "Central"),
    (5, "Nadia Sheikh", "Gulberg", "IMARAT", "Sadia Rehman", "Adeel Aslam",
     "Kamran Shah", "Emporium", "South"),
    (6, "Omar Farooq", "GCC", "IMARAT", "Sadia Rehman", "Adeel Aslam",
     "Bilal Qadir", "Emporium", "South"),
]


@pytest.fixture(scope="module")
def _engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    for wid, name, team, company, unit_head, zonal, bcm, office, region in PEOPLE:
        s.add(Advisor(wid=wid, name=name, team=team, company=company, rm=unit_head,
                      portfolio_lead=zonal, management_lead=bcm, office=office,
                      region=region, unit="A", in_master_sheet=True))
        s.add(Performance(wid=wid, period=PerformancePeriod.MTD, target=1000,
                          cleared=100 * wid, pct=10.0 * wid))
        s.add(Performance(wid=wid, period=PerformancePeriod.YTD, target=10000,
                          cleared=1000 * wid, pct=10.0 * wid))
        s.add(SalesFunnel(wid=wid, mtd_new_connect=10 * wid, mtd_followup_connect=0,
                          mtd_cr=5 * wid, mtd_new_meeting=wid, mtd_followup_meeting=0,
                          mtd_conversion=wid, mtd_booking_stored=wid,
                          mtd_meetings_planned=wid + 2, mtd_meetings_conducted=wid))
        s.add(Pipeline(wid=wid, pipeline=1000 * wid, overdue=wid))
        s.add(Portfolio(wid=wid, value=5000 * wid))
        s.add(Calls(wid=wid, answered_calls_mtd=20 * wid, connects_mtd=10 * wid))
        s.add(Attendance(wid=wid, biometric_mtd_ontime=15, biometric_mtd_late=5,
                         biometric_mtd_not_marked=0, login_mtd_ontime=18,
                         login_mtd_late=2, login_mtd_not_marked=0))
    for team, target, achieved in (("Blue Area", 3000, 2400), ("Downtown", 2000, 1100),
                                   ("Gulberg", 1500, 700), ("GCC", 1200, 450)):
        s.add(TeamTarget(team=team, target=target, achieved=achieved,
                         achievement_pct=round(achieved / target * 100, 1)))
    s.commit()
    yield engine
    s.close()


@pytest.fixture()
def org(_engine, monkeypatch):
    from app.core.config import settings
    from app.llm import llm_client, semantic_parser

    monkeypatch.setattr(settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(settings, "use_llm_planner", False, raising=False)
    monkeypatch.setattr(llm_client, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)

    s = sessionmaker(bind=_engine)()
    entity_extractor._cache["loaded_at"] = 0
    yield s
    s.close()


def _level(resolution):
    if resolution.kind == "ir":
        return resolution.ir.subject_level
    plan = getattr(resolution, "plan", None)
    return plan.level if plan else None


# The core matrix. Every one of these named a subject and every one of
# them used to answer at `metric.primary_level` instead.
SUBJECT_QUERIES = [
    # ---- team ----
    ("What is Downtown's pipeline value this month?", "team"),
    ("What is Downtown's revenue?", "team"),
    ("What is Downtown's overdue?", "team"),
    ("What is Blue Area's portfolio?", "team"),
    ("Total connects for Blue Area", "team"),
    ("Downtown revenue", "team"),
    ("What is Blue Area's revenue year to date?", "team"),
    ("What is Blue Area's achievement %?", "team"),
    # ---- company ----
    ("What is Graana's cleared?", "company"),
    ("What is IMARAT's pipeline value?", "company"),
    # ---- office (business center) ----
    ("What is Beverly Center's revenue?", "office"),
    ("What is Gold Crest's overdue?", "office"),
    # ---- region ----
    ("What is Central Region's revenue?", "region"),
    ("What is South's pipeline value?", "region"),
    # ---- bcm ----
    ("What is BCM Usman Ghani's group overdue count?", "bcm"),
    ("What is Rabia Anjum's pipeline value?", "bcm"),
    # ---- zonal head ----
    ("What is Fawad Hafeez's revenue?", "zonal_head"),
    ("What is Adeel Aslam's overdue?", "zonal_head"),
    # ---- unit head ----
    ("What is Tariq Mehmood's pipeline value?", "unit_head"),
    ("What is Sadia Rehman's revenue?", "unit_head"),
]


@pytest.mark.parametrize("query,expected", SUBJECT_QUERIES)
def test_the_named_subject_owns_the_level(org, query, expected):
    resolution = nlu_pipeline.resolve(query, org, session_id=None)
    assert _level(resolution) == expected, (
        f"{query!r} answered at {_level(resolution)!r}, not about the "
        f"{expected} the user named"
    )


@pytest.mark.parametrize("query,expected", SUBJECT_QUERIES)
def test_no_named_subject_query_becomes_an_advisor_leaderboard(org, query, expected):
    """The success criterion, stated as its own assertion: no query
    silently becomes an advisor list unless advisors were asked for."""
    if expected == "advisor":
        return
    assert _level(nlu_pipeline.resolve(query, org, session_id=None)) != "advisor"


# Explicit level words must still win — including when they disagree
# with the grounded entity.
EXPLICIT_LEVEL_QUERIES = [
    ("Top 5 advisors in Blue Area by revenue", "advisor"),
    ("Which team has the highest overdue?", "team"),
    ("top 3 teams by portfolio value", "team"),
    ("Show me revenue by company", "company"),
    ("Top advisors by revenue", "advisor"),
    ("overdue count by team", "team"),
]


@pytest.mark.parametrize("query,expected", EXPLICIT_LEVEL_QUERIES)
def test_an_explicit_level_word_wins_end_to_end(org, query, expected):
    assert _level(nlu_pipeline.resolve(query, org, session_id=None)) == expected


# Requirement 4: unscoped queries keep the metric default.
NO_SUBJECT_QUERIES = [
    ("What is revenue?", "advisor"),
    ("Show me the revenue leaderboard", "advisor"),
    ("What is the achievement %?", "team"),      # primary_level=team
    ("What is the 1 unit ratio?", "team"),       # primary_level=team
]


@pytest.mark.parametrize("query,expected", NO_SUBJECT_QUERIES)
def test_the_metric_default_still_applies_when_nothing_is_named(org, query, expected):
    assert _level(nlu_pipeline.resolve(query, org, session_id=None)) == expected


def test_a_ranking_scopes_to_the_group_rather_than_ranking_it(org):
    """"Top 5 in Blue Area by revenue" lists people IN Blue Area, and
    keeps Blue Area as a filter rather than promoting it to the subject."""
    resolution = nlu_pipeline.resolve("Top 5 in Blue Area by revenue", org,
                                      session_id=None)
    assert _level(resolution) == "advisor"
    if resolution.kind == "ir":
        assert any(f.field == "team" and f.value == "Blue Area"
                   for f in resolution.ir.filters)


# ---------------------------------------------------------------------
# Single ownership
# ---------------------------------------------------------------------


def test_the_validator_does_not_override_a_chosen_level(org):
    """ir_validator used to re-decide the level from `primary_level`,
    silently undoing a correct entity level on the way to the compiler.
    It may now only DEGRADE an unanswerable one."""
    from app.llm.ir_validator import validate_ir
    from app.llm.query_ir import Filter, MetricRef, QueryIR, Sort

    ir = QueryIR(intent="leaderboard", subject_level="team",
                 metric=MetricRef(key="pipeline_value"),
                 sort=Sort(metric="pipeline_value"),
                 filters=[Filter(field="team", operator="=", value="Downtown")],
                 limit=10)
    result = validate_ir(ir, org)
    assert result.ir.subject_level == "team"


def test_the_level_decision_is_traced_with_its_losers(org):
    """Requirement 7 — a future audit must see WHY a level was chosen and
    what it displaced, without re-deriving it."""
    nlu_pipeline.resolve("What is Downtown's pipeline value this month?", org,
                         session_id=None)
    step = next(s for s in routing.current_trace().steps if s.stage == "Level")

    assert step.chose == "team"
    assert "Downtown" in step.why          # the winner's evidence
    assert "metric_default" in step.why    # the loser
    assert "lost" in step.why              # and why it lost


def test_the_trace_does_not_repeat_an_unchanged_decision(org):
    nlu_pipeline.resolve("What is Downtown's revenue?", org, session_id=None)
    levels = [s for s in routing.current_trace().steps if s.stage == "Level"]
    assert len(levels) == 1


# ---------------------------------------------------------------------
# Relations and people are unaffected
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "Who does Yasir Ali report to?",
    "Who is Nadia Sheikh's unit head?",
    "List all advisors under BCM Rabia Anjum.",
    "Show me Fawad Hafeez's team",
])
def test_relation_queries_are_unaffected(org, query):
    resolution = nlu_pipeline.resolve(query, org, session_id=None)
    assert resolution.kind != "clarify"


@pytest.mark.parametrize("query", [
    "What is Yasir Ali's revenue?",
    "What is Nadia Sheikh's overdue?",
])
def test_a_named_advisor_still_answers_about_that_advisor(org, query):
    resolution = nlu_pipeline.resolve(query, org, session_id=None)
    assert _level(resolution) == "advisor"


# ---------------------------------------------------------------------
# R1 — the Phase 1 possessive regression this phase also removed
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "What is BCM Usman Ghani's group overdue count?",
    "How many total connects does BCM Rabia Anjum's group have today?",
    "What is Central Region's revenue this month?",
    "Compare Fawad Hafeez's group and Adeel Aslam's group on cleared",
    "Which of BCM Usman Ghani's or BCM Rabia Anjum's groups has more conversions?",
])
def test_role_prefixed_and_group_possessives_are_not_read_as_unknown_people(org, query):
    """The P3 detector walked backwards over every capitalised token, so
    "BCM Usman Ghani's", "Compare Fawad Hafeez's" and "Central Region's"
    all became names of people who do not exist."""
    resolution = nlu_pipeline.resolve(query, org, session_id=None)
    assert "couldn't find anyone" not in (resolution.clarify_message or "")


@pytest.mark.parametrize("query,name", [
    ("Show me Ahmed's performance", "Ahmed"),
    ("What is Zainab's CR?", "Zainab"),
])
def test_a_genuinely_unknown_person_is_still_refused(org, query, name):
    """The R1 fix must not overshoot: P3 exists because answering about
    someone else is worse than asking who was meant."""
    resolution = nlu_pipeline.resolve(query, org, session_id=None)
    assert "couldn't find anyone" in (resolution.clarify_message or "")
    assert name in resolution.clarify_message


def test_trim_span_strips_only_what_is_not_a_name():
    assert subject_level is not None  # module import guard
    from app.llm.routing import _trim_span

    assert _trim_span("BCM Usman Ghani") == "Usman Ghani"
    assert _trim_span("Unit Head Tariq Mehmood") == "Tariq Mehmood"
    assert _trim_span("Compare Fawad Hafeez") == "Fawad Hafeez"
    # a bare name is untouched
    assert _trim_span("Yasir Ali") == "Yasir Ali"
    assert _trim_span("Ahmed") == "Ahmed"
