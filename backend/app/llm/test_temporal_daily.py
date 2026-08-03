"""Temporal understanding — "today" is a period, not a synonym for MTD.

`today`, `right now` and `currently` matched NOTHING: not an equivalent
period, not an unsupported one. So they fell through to the MTD default
and a question about today was answered with month-to-date figures,
silently. "today" is the single most common period word in the KPI
spec's own example questions — 34 of its 89 use it.

They are now recognised vocabulary:

    today / right now / as of now / so far today  -> DAILY
    currently / at the moment                     -> MTD

DAILY IS RECOGNISED, NOT ANSWERABLE. PerformancePeriod holds MTD/YTD/3M
and the SalesFunnel columns are MTD-only, so no metric has daily data —
and metric bindings were explicitly out of scope here. That is the whole
improvement: metric_for_period() returns None for a period a measure has
no data at, its contract says "callers must degrade, not substitute", and
a daily question now SAYS SO instead of quietly answering a different
one. When daily data lands, DAILY is already its name and only the
bindings change.

"currently" is deliberately MTD, not DAILY: it means "as things stand",
not "on this specific day", and MTD is the current-state figure this
system actually holds.
"""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import entity_extractor, temporal_parser
from app.llm.entity_extractor import extract_entities
from app.llm.preprocessing import normalize
from app.llm.query_compiler import compile_and_run, effective_metric
from app.llm.query_ir import TimeRange, plan_to_ir
from app.llm.query_planner import build_query_plan
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def org(db_session):
    for wid, name in ((1, "Adv One"), (2, "Adv Two")):
        db_session.add(Advisor(wid=wid, name=name, team="Blue Area",
                               company="Graana", in_master_sheet=True))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=100 * wid, pct=50))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.YTD,
                                   target=1000, cleared=1000 * wid, pct=50))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=10 * wid,
                                   mtd_followup_connect=0))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    return db_session


def _ir(text, db):
    cleaned = normalize(text)
    entities = extract_entities(cleaned, db)
    return plan_to_ir(build_query_plan(cleaned, entities), entities)


# ---------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------

@pytest.mark.parametrize("phrase", [
    "today", "revenue today", "today's revenue", "right now",
    "as of now", "as of today", "so far today",
])
def test_daily_vocabulary(phrase):
    match = temporal_parser.parse_period(phrase)
    assert match is not None, phrase
    assert match.kind == "equivalent", phrase
    assert match.period == "DAILY", phrase


@pytest.mark.parametrize("phrase", ["currently", "at the moment"])
def test_currently_means_month_to_date(phrase):
    """"As things stand" is the current-state figure this system holds —
    it is not naming a day."""
    match = temporal_parser.parse_period(phrase)
    assert match.kind == "equivalent"
    assert match.period == "MTD"


@pytest.mark.parametrize("phrase,expected", [
    ("revenue this month", "MTD"),
    ("month to date", "MTD"),
    ("mtd", "MTD"),
    ("revenue year to date", "YTD"),
    ("this year", "YTD"),
    ("ytd", "YTD"),
    ("this quarter", "3M"),
    ("last 3 months", "3M"),
])
def test_the_existing_vocabulary_is_unchanged(phrase, expected):
    assert temporal_parser.match_period(phrase) == expected


@pytest.mark.parametrize("phrase", [
    "yesterday", "last month", "previous month", "this week",
    "last week", "past 7 days", "last quarter",
])
def test_genuinely_unsupported_windows_still_refuse(phrase):
    """"today" joining the vocabulary must not turn a real calendar
    window into a silent guess. These still carry an explanation."""
    match = temporal_parser.parse_period(phrase)
    assert match.kind == "unsupported", phrase
    assert match.reason
    assert temporal_parser.match_period(phrase) is None


def test_today_is_not_confused_with_yesterday():
    """"yesterday" contains no "today", but the check is cheap and the
    failure would be silent."""
    assert temporal_parser.parse_period("yesterday").kind == "unsupported"
    assert temporal_parser.parse_period("today").period == "DAILY"


# ---------------------------------------------------------------------
# The explicit period always wins
# ---------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Revenue today", "DAILY"),
    ("Revenue this month", "MTD"),
    ("Revenue year to date", "YTD"),
    ("revenue this quarter", "3M"),
    ("revenue right now", "DAILY"),
    ("revenue currently", "MTD"),
])
def test_the_stated_period_reaches_the_ir(org, text, expected):
    """The metric "revenue" resolves to an MTD-flavoured key. The user's
    words must still decide the window — this is the F4 rule, now
    covering DAILY too."""
    assert _ir(text, org).time_range.period == expected


def test_a_query_naming_no_period_is_unchanged(org):
    assert _ir("top advisors by revenue", org).time_range.period == "MTD"


def test_the_ir_can_carry_daily():
    """If TimeRange rejected DAILY, "revenue today" would be a pydantic
    error and degrade to exactly the silent MTD default this replaces."""
    assert TimeRange(period="DAILY").period == "DAILY"
    assert set(temporal_parser.PERIODS) == set(TimeRange.model_fields["period"].annotation.__args__)


# ---------------------------------------------------------------------
# DAILY refuses rather than substituting
# ---------------------------------------------------------------------

def test_a_daily_metric_query_has_no_answer(org):
    """No metric has daily data, and bindings were out of scope. The
    correct outcome is no rows — not MTD rows under a daily question."""
    ir = _ir("top advisors by revenue today", org)

    assert ir.time_range.period == "DAILY"
    assert effective_metric(ir) is None
    assert compile_and_run(org, ir) is None


def test_the_daily_refusal_says_why(org):
    """The old wording blamed the metric and quoted its raw key
    ("mtd_cleared"), reading as though revenue itself were unsupported.
    The period is the actionable part."""
    response = handle_chat_message(org, "Revenue today", session_id=None)

    assert response["type"] == "unknown"
    assert "daily" in response["reply"].lower()
    assert "MTD" in response["reply"]          # what IS available
    assert "mtd_cleared" not in response["reply"]


def test_daily_does_not_silently_return_month_to_date(org):
    """The defect itself. Adv Two has 200 MTD; a daily question must not
    produce that number."""
    response = handle_chat_message(org, "top advisors by revenue today", session_id=None)
    assert "200" not in response["reply"]
    assert response["data"] is None


@pytest.mark.parametrize("text,expected_period", [
    ("Revenue this month", "MTD"),
    ("Revenue year to date", "YTD"),
])
def test_the_answerable_periods_still_answer(org, text, expected_period):
    response = handle_chat_message(org, text, session_id=None)
    assert response["type"] == "leaderboard"
    assert response["data"]


def test_ytd_returns_ytd_numbers(org):
    response = handle_chat_message(org, "top advisors by revenue year to date", session_id=None)
    assert [r["value"] for r in response["data"]] == [2000, 1000]


# ---------------------------------------------------------------------
# The label must name the metric that was actually computed
# ---------------------------------------------------------------------

def test_the_reply_labels_the_period_it_actually_used(org):
    """Found while implementing this step, and its own kind of silently
    wrong answer: once a stated period could reach the compiler, the
    VALUE came from ytd_cleared while the header still read "MTD Revenue
    Cleared" — the right figure under the wrong name. Everything that
    names the metric now resolves it the same way the value was
    computed."""
    response = handle_chat_message(org, "top advisors by revenue year to date", session_id=None)

    assert "YTD Revenue Cleared" in response["reply"]
    assert "MTD Revenue Cleared" not in response["reply"]


def test_the_mtd_label_is_still_right_when_mtd_was_used(org):
    response = handle_chat_message(org, "top advisors by revenue this month", session_id=None)
    assert "MTD Revenue Cleared" in response["reply"]
