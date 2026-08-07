"""Phase 11 — an explicit period must never be silently discarded.

THE DEFECT. `advisor_metric` (one measure, one named person) is served
off the rule-based PLAN path, and that branch called the advisor service
with `plan.metric` alone. The period the planner had already resolved and
carried on `plan.period` was read by NOTHING, so the metric answered at
its own declared window:

    "Zainab's CR today"      -> 7 MTD client registrations
    "Zainab's CR this year"  -> 7 MTD client registrations

Nothing substituted MTD. The period was simply never applied, which is
why the reply looked internally consistent — a real number, correctly
labelled "MTD", answering a question nobody asked.

WHY IT LOOKED LIKE A DAILY BUG. "Zainab's YTD CR" was right, so the
failure read as DAILY-specific. It was not: metric_aliases has a literal
"ytd cr" phrase that resolves straight to `ytd_client_registrations`, so
the period was baked into the METRIC KEY and `plan.period` was never
needed. Rephrase it as "CR this year" and YTD failed the same way. Both
phrasings are pinned below for that reason.

THE SECOND DEFECT was vocabulary: temporal_parser mapped "today", "right
now", "this morning" and "tonight" to DAILY but not the literal word
"daily", so "daily CR" named no period at all.

The two paths that can answer a measure — the IR path (leaderboards,
group metrics) and this plan path — must reach the SAME period verdict
through the SAME authority (query_compiler.resolve_metric_for_period).
The last test here asserts exactly that, because the defect was precisely
that they disagreed.
"""

import pytest

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import (
    advisor_resolver, conversation_memory, entity_extractor, narrative,
    nlu_pipeline, semantic_parser, temporal_parser,
)
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def db(db_session, monkeypatch):
    """MTD and YTD CR differ by an order of magnitude, so a reply that
    answered at the wrong window cannot pass by coincidence."""
    db_session.add(Advisor(wid=1, name="Zainab Riaz", team="Blue Area",
                           company="Graana", bm="Kaleem Ullah", rm="Kaleem Ullah",
                           in_master_sheet=True))
    db_session.add(SalesFunnel(wid=1, mtd_cr=7, ytd_cr=88, mtd_new_connect=2,
                               mtd_followup_connect=0, mtd_new_meeting=3,
                               mtd_followup_meeting=1, mtd_booking_stored=4))
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, target=1000, cleared=500))
    db_session.add(Performance(wid=1, period=PerformancePeriod.YTD, target=10000, cleared=5000))
    db_session.add(Pipeline(wid=1, pipeline=7500, overdue=2))
    db_session.add(Calls(wid=1, answered_calls_mtd=20, connects_mtd=10))

    db_session.add(Advisor(wid=2, name="Waqar Haider", team="Blue Area",
                           company="Graana", in_master_sheet=True))
    db_session.add(SalesFunnel(wid=2, mtd_cr=14, ytd_cr=176))
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


def _reply(db, text, session_id=None):
    return handle_chat_message(db, text, session_id=session_id)


# ---------------------------------------------------------------------
# The vocabulary gap
# ---------------------------------------------------------------------


@pytest.mark.parametrize("phrase", ["daily", "daily CR", "Zainab's Daily CR"])
def test_the_literal_word_daily_names_the_daily_window(phrase):
    """"today" was DAILY and "daily" was nothing — so the most direct way
    of asking for the daily figure was the one phrasing that fell through
    to the MTD default."""
    match = temporal_parser.parse_period(phrase)
    assert match is not None, phrase
    assert match.kind == "equivalent", phrase
    assert match.period == "DAILY", phrase


def test_today_still_names_the_daily_window():
    assert temporal_parser.parse_period("CR today").period == "DAILY"


@pytest.mark.parametrize("phrase,expected", [
    ("MTD CR", "MTD"), ("CR this month", "MTD"),
    ("YTD CR", "YTD"), ("CR this year", "YTD"), ("CR year to date", "YTD"),
    ("CR this quarter", "3M"),
])
def test_adding_daily_did_not_disturb_the_other_windows(phrase, expected):
    assert temporal_parser.parse_period(phrase).period == expected


def test_a_query_naming_no_window_names_no_period():
    """The MTD default must stay a DEFAULT — inferred from the metric,
    never asserted by the parser."""
    assert temporal_parser.parse_period("Zainab's CR") is None


# ---------------------------------------------------------------------
# The advisor_metric path — one person, one measure, the stated window
# ---------------------------------------------------------------------


def test_no_stated_window_answers_at_the_metrics_own_period(db):
    reply = _reply(db, "What is Zainab Riaz's CR?")["reply"]
    assert "7" in reply and "MTD" in reply


def test_an_explicit_mtd_request_answers_mtd(db):
    reply = _reply(db, "What is Zainab Riaz's MTD CR?")["reply"]
    assert "7" in reply and "MTD" in reply


@pytest.mark.parametrize("text", [
    "What is Zainab Riaz's YTD CR?",        # the metric_aliases phrase
    "What is Zainab Riaz's CR this year?",  # period word only — this one FAILED
    "What is Zainab Riaz's CR year to date?",
])
def test_an_explicit_ytd_request_answers_ytd_however_it_is_phrased(db, text):
    """The alias phrasing worked because "ytd cr" resolves to the YTD KEY;
    the period-word phrasings reached the same branch with the MTD key and
    `plan.period='YTD'` that nothing read."""
    reply = _reply(db, text)["reply"]
    assert "88" in reply, f"{text!r} did not answer at YTD"
    assert "YTD" in reply
    assert "7 MTD" not in reply


@pytest.mark.parametrize("text", [
    "What is Zainab Riaz's Daily CR?",
    "What is Zainab Riaz's CR today?",
])
def test_a_daily_request_reports_that_daily_is_unavailable(db, text):
    """SalesFunnel holds mtd_cr and ytd_cr — there is no daily CR column,
    so the correct answer is not a number."""
    response = _reply(db, text)
    reply = response["reply"]
    assert "daily" in reply.lower()
    assert "MTD, YTD" in reply, "the reply must say which windows DO exist"
    assert response["data"] is None


@pytest.mark.parametrize("text", [
    "What is Zainab Riaz's Daily CR?",
    "What is Zainab Riaz's CR today?",
])
def test_a_daily_request_never_returns_the_mtd_number(db, text):
    """THE production invariant. This is the exact string the defect
    produced, and no phrasing of a daily question may produce it."""
    reply = _reply(db, text)["reply"]
    assert "7 MTD client registrations" not in reply
    assert "has 7" not in reply


def test_an_unavailable_window_is_reported_for_any_measure_not_just_cr(db):
    """The fix is the period resolution, not a CR branch — so a different
    MTD-only measure must refuse the same way."""
    reply = _reply(db, "What is Zainab Riaz's pipeline today?")["reply"]
    assert "daily" in reply.lower()


# ---------------------------------------------------------------------
# Context: an explicit period wins, an inherited one never overrides it
# ---------------------------------------------------------------------


def test_an_explicit_period_overrides_an_inherited_one(db):
    """"MTD CR" then "daily" — the inherited MTD must not survive a turn
    that names its own window."""
    _reply(db, "What is Zainab Riaz's MTD CR?", session_id="p1")
    reply = _reply(db, "What is Zainab Riaz's CR today?", session_id="p1")["reply"]
    assert "daily" in reply.lower()
    assert "7" not in reply


def test_an_inherited_period_cannot_override_an_explicit_daily(db):
    """CR -> YTD -> Daily. Neither the MTD default of turn 1 nor the YTD
    of turn 2 may reach turn 3."""
    _reply(db, "What is Zainab Riaz's CR?", session_id="p2")
    ytd = _reply(db, "What is Zainab Riaz's YTD CR?", session_id="p2")["reply"]
    assert "88" in ytd
    daily = _reply(db, "What is Zainab Riaz's Daily CR?", session_id="p2")["reply"]
    assert "daily" in daily.lower()
    assert "88" not in daily and "7" not in daily


def test_a_follow_up_naming_no_window_keeps_the_inherited_one(db):
    """The other half of the precedence rule: silence inherits. Phase 10
    owns this merge; asserted here so the period fix cannot quietly
    weaken it."""
    _reply(db, "Top advisors by CR year to date", session_id="p3")
    resolution = nlu_pipeline.resolve("top 5", db, session_id="p3")
    assert resolution.ir.time_range.period == "YTD"
    assert resolution.ir.metric.key == "ytd_client_registrations"


# ---------------------------------------------------------------------
# The two paths must agree
# ---------------------------------------------------------------------


@pytest.mark.parametrize("person,group", [
    ("What is Zainab Riaz's CR today?", "Top advisors by CR today"),
    ("What is Zainab Riaz's Daily CR?", "Top advisors by daily CR"),
])
def test_both_answer_paths_reach_the_same_period_verdict(db, person, group):
    """The defect WAS the disagreement: the IR path already refused a
    daily question honestly while the plan path answered it with MTD.
    Both now resolve through query_compiler.resolve_metric_for_period, so
    one question cannot have two answers."""
    assert _reply(db, person)["reply"] == _reply(db, group)["reply"]


def test_the_group_path_still_answers_the_windows_it_has(db):
    """Regression guard on the path that was already correct."""
    assert "MTD" in _reply(db, "Top advisors by CR")["reply"]
    assert "YTD" in _reply(db, "Top advisors by CR year to date")["reply"]


# ---------------------------------------------------------------------
# The unavailable reply names the measure, not the key's own window
# ---------------------------------------------------------------------


@pytest.mark.parametrize("text,measure", [
    ("Daily Connects", "Total Connects"),
    ("Connects today", "Total Connects"),
    ("Daily CR", "Client Registrations"),
    ("CR today", "Client Registrations"),
])
def test_an_unavailable_reply_names_the_measure_period_neutrally(db, text, measure):
    """"Total MTD Connects" put a window in the measure's name that the
    user never asked for — it is only which key the alias table resolved
    — leaving a sentence with two periods in it. See
    app/llm/test_measure_label.py for the derivation this rests on."""
    reply = _reply(db, text)["reply"]
    assert f"for {measure} yet" in reply
    assert "MTD Connects" not in reply
    assert "MTD Client Registrations" not in reply


@pytest.mark.parametrize("text,window", [
    ("Daily Connects", "daily"),
    ("Daily CR", "daily"),
    ("Connects this quarter", "3-month"),
])
def test_the_requested_window_survives_the_neutral_label(db, text, window):
    """The label going neutral must not make the SENTENCE vague — which
    window was asked for is the part the user can act on."""
    reply = _reply(db, text)["reply"]
    assert f"I don't have {window} figures" in reply
    assert "I hold MTD, YTD totals" in reply


def test_an_available_window_still_captions_its_answer_with_that_window(db):
    """Only the unavailable sentence goes neutral. A real answer must
    still say which window it computed, or a YTD figure would ship under
    an unqualified name."""
    assert "YTD" in _reply(db, "Top advisors by connects year to date")["reply"]
    assert "MTD" in _reply(db, "Top advisors by connects")["reply"]
