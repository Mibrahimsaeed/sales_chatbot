"""Phase 13B end to end — every measure asked for is accounted for.

THE INVARIANT these tests exist to hold:

    If the user names N measures, the reply either answers all N or says
    which ones it could not.

Silently answering one of two is the failure that motivated the phase,
and it is invisible in a reply — "Zainab has 200 answered calls" is a
correct sentence, correctly labelled, about half the question. So every
assertion below is on the VALUES, with the fixture giving each measure a
different number: a test that checked labels would pass while reading
the wrong column.

SHAPES, and why they are answered where they are:

  one subject, several measures   -> the advisor_metric dispatch, which
                                     already resolved identity and
                                     (measure, period) and now loops
  several subjects, any measures  -> comparison_service, which has taken
                                     a tuple of KPI keys since it was
                                     written and needed only to be given
                                     more than one
  different measure per subject   -> REFUSED. A plan carries one subject
                                     and a flat measure list, so there is
                                     nowhere to record that connects
                                     belongs to one person and answered
                                     calls to another. Answering anyway
                                     means attaching both to whoever
                                     resolved, which is not a partial
                                     answer but a wrong one.
"""

import pytest

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import (
    advisor_resolver, conversation_memory, entity_extractor, narrative,
    semantic_parser,
)
from app.services.chat_service import handle_chat_message

# Every measure a different number, and different between the two people,
# so no assertion can pass by reading the wrong column or the wrong row.
#        wid  name             connects  answered  cr   ytd_connects
PEOPLE = [(1, "Zainab Tariq",  500,      200,      7,   5000),
          (2, "Awais Ali",     700,      300,      11,  7000)]


@pytest.fixture()
def db(db_session, monkeypatch):
    for wid, name, connects, answered, cr, ytd_connects in PEOPLE:
        db_session.add(Advisor(wid=wid, name=name, team="Blue Area",
                               company="Graana", in_master_sheet=True))
        db_session.add(SalesFunnel(
            wid=wid, mtd_new_connect=connects, mtd_followup_connect=0, mtd_cr=cr,
            ytd_new_connect=ytd_connects, ytd_followup_connect=0, ytd_cr=cr * 10,
            mtd_new_meeting=3, mtd_followup_meeting=1, mtd_conversion=2))
        db_session.add(Calls(wid=wid, answered_calls_mtd=answered, connects_mtd=connects))
        db_session.add(Pipeline(wid=wid, pipeline=1000 * wid, overdue=wid))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=1000, cleared=400 * wid))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.YTD,
                                   target=10000, cleared=4000 * wid))
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


def _reply(db, text):
    return handle_chat_message(db, text, session_id=None)


def _numbers(text):
    import re
    out = set()
    for token in re.findall(r"\d[\d,]*", text):
        try:
            out.add(int(token.replace(",", "")))
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------------
# Single measure — unchanged
# ---------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("What are Zainab Tariq's connects?", 500),
    ("What are Zainab Tariq's answered calls?", 200),
    ("What are Zainab Tariq's client registrations?", 7),
])
def test_a_single_measure_answers_exactly_as_before(db, text, expected):
    assert expected in _numbers(_reply(db, text)["reply"])


# ---------------------------------------------------------------------
# Shape A — one subject, several measures
# ---------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("What are Zainab Tariq's connects and answered calls?", {500, 200}),
    ("Give me Zainab Tariq's connects and answered calls.", {500, 200}),
    ("What are Zainab Tariq's connects and client registrations?", {500, 7}),
    ("What are Zainab Tariq's connects, answered calls and client registrations?",
     {500, 200, 7}),
])
def test_every_requested_measure_is_answered(db, text, expected):
    """THE reported defect. Before this phase each of these returned
    whichever measure had the longest alias string and said nothing about
    the others."""
    assert expected <= _numbers(_reply(db, text)["reply"])


def test_the_dropped_measure_is_the_one_that_used_to_survive(db):
    """"answered calls" is the longer alias, so it was the survivor and
    connects was lost. Pinned by value so a regression is unmistakable."""
    reply = _reply(db, "What are Zainab Tariq's connects and answered calls?")["reply"]
    assert 500 in _numbers(reply), "connects — the measure that used to vanish"
    assert 200 in _numbers(reply), "answered calls — the measure that used to survive"


def test_the_structured_payload_carries_every_measure(db):
    data = _reply(db, "What are Zainab Tariq's connects and answered calls?")["data"]
    assert data["metrics"] == [
        {"metric": "total_connects", "value": 500.0},
        {"metric": "answered_calls", "value": 200.0},
    ]


def test_measures_are_reported_in_the_order_they_were_named(db):
    reply = _reply(db, "What are Zainab Tariq's connects and answered calls?")["reply"]
    assert reply.index("500") < reply.index("200")
    reply = _reply(db, "What are Zainab Tariq's answered calls and connects?")["reply"]
    assert reply.index("200") < reply.index("500")


# ---------------------------------------------------------------------
# Shape B — several subjects, one measure (must not regress)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "What are Zainab Tariq's connects and Awais Ali's connects?",
    "Compare Zainab Tariq and Awais Ali on connects.",
])
def test_two_subjects_on_one_measure_still_answer_both(db, text):
    got = _numbers(_reply(db, text)["reply"])
    assert 500 in got and 700 in got


def test_two_subjects_on_one_measure_still_take_the_single_metric_path(db):
    """One measure keeps the IR path, and everything that depends on it —
    _effective_metric, conversation memory, the response planner."""
    assert _reply(db, "Compare Zainab Tariq and Awais Ali on connects.")["type"] == "comparison"


# ---------------------------------------------------------------------
# Shape C — several subjects, several measures
# ---------------------------------------------------------------------


def test_two_subjects_on_two_measures_answer_all_four_values(db):
    reply = _reply(db, "Compare Zainab Tariq and Awais Ali on connects and answered calls.")["reply"]
    assert {500, 200, 700, 300} <= _numbers(reply)


def test_the_multi_measure_comparison_labels_every_row(db):
    reply = _reply(db, "Compare Zainab Tariq and Awais Ali on connects and answered calls.")["reply"]
    assert "Total MTD Connects" in reply
    assert "Answered Calls (MTD)" in reply


def test_the_multi_measure_comparison_header_does_not_leak_a_key_list(db):
    """The header names ONE measure or none; with several, the rows carry
    the labels. Rendering the key list into the sentence put
    "['total_connects', 'answered_calls']" in front of the user."""
    reply = _reply(db, "Compare Zainab Tariq and Awais Ali on connects and answered calls.")["reply"]
    assert "total_connects" not in reply
    assert "[" not in reply.split("\n")[0]


# ---------------------------------------------------------------------
# Shape D — a measure each. Refused, never misattributed.
# ---------------------------------------------------------------------


def test_a_measure_per_subject_is_refused_rather_than_misattributed(db):
    """THE safety requirement. Attaching both measures to whichever
    person resolved is not a partial answer, it is the wrong person's
    number under the right label."""
    response = _reply(db, "What are Zainab Tariq's connects and Awais Ali's answered calls?")
    assert response["type"] == "clarification"
    reply = response["reply"]
    assert 300 not in _numbers(reply), "Awais's answered calls attributed to Zainab"
    assert 200 not in _numbers(reply), "Zainab's answered calls offered as an answer"


def test_the_refusal_names_both_measures_and_offers_a_way_forward(db):
    reply = _reply(db, "What are Zainab Tariq's connects and Awais Ali's answered calls?")["reply"]
    assert "Total Connects" in reply and "Answered Calls" in reply
    assert "compare" in reply.lower()


def test_one_possessive_with_two_measures_is_NOT_refused(db):
    """The guard keys on measures distributed across subjects. A single
    subject with two measures must still be answered."""
    got = _numbers(_reply(db, "What are Zainab Tariq's connects and answered calls?")["reply"])
    assert {500, 200} <= got


# ---------------------------------------------------------------------
# Periods — every measure resolved at the requested window
# ---------------------------------------------------------------------


def test_a_ytd_multi_measure_query_uses_the_ytd_column_for_what_has_one(db):
    """connects has a YTD sibling and answered calls does not, so this is
    the mixed case: one answered at YTD, the other reported unavailable —
    and never silently answered at MTD."""
    reply = _reply(db, "What are Zainab Tariq's connects and answered calls year to date?")["reply"]
    got = _numbers(reply)
    assert 5000 in got, "YTD connects"
    assert 200 not in got, "MTD answered calls silently substituted for YTD"
    assert "year-to-date" in reply and "Answered Calls" in reply


def test_an_mtd_multi_measure_query_uses_the_mtd_columns(db):
    got = _numbers(_reply(db, "What are Zainab Tariq's connects and answered calls this month?")["reply"])
    assert {500, 200} <= got
    assert 5000 not in got


@pytest.mark.parametrize("text", [
    "What are Zainab Tariq's connects and answered calls today?",
    "What are Zainab Tariq's daily connects and answered calls?",
])
def test_a_daily_multi_measure_query_never_answers_with_another_window(db, text):
    """Whatever daily data exists, the one thing that must never happen
    is a daily question answered with the month. Asserted on the MTD and
    YTD values, so this holds whether or not a daily binding exists."""
    reply = _reply(db, text)["reply"]
    got = _numbers(reply)
    assert 500 not in got and 200 not in got, "an MTD figure answered a daily question"
    assert 5000 not in got, "a YTD figure answered a daily question"


# ---------------------------------------------------------------------
# The no-silent-partial invariant
# ---------------------------------------------------------------------


@pytest.mark.parametrize("text,requested", [
    ("What are Zainab Tariq's connects and answered calls?", ["Connects", "Answered Calls"]),
    ("What are Zainab Tariq's connects and answered calls year to date?",
     ["Connects", "Answered Calls"]),
    ("What are Zainab Tariq's connects and answered calls today?",
     ["Connects", "Answered Calls"]),
])
def test_every_requested_measure_is_either_answered_or_named_as_unavailable(
        db, text, requested):
    """The invariant, stated directly. A measure the user asked for may
    be absent from the numbers only if the reply says so."""
    reply = _reply(db, text)["reply"]
    for measure in requested:
        assert measure.lower() in reply.lower(), (
            f"{measure} was requested and appears nowhere in the reply — "
            "neither answered nor reported unavailable"
        )


def test_when_nothing_can_be_answered_every_measure_is_still_named(db):
    """Refusing a two-measure question by describing one of them is the
    same silent-partial failure as answering one of them.

    Pinned on client registrations and conversions since Phase 17: this
    used connects and answered calls, and both of those gained real daily
    bindings when Phase 12 was restored, so the query now ANSWERS and the
    test had no unanswerable case left to check. The invariant is
    unchanged; only the pair of measures moved to two that genuinely have
    no daily source.
    """
    reply = _reply(db, "What are Zainab Tariq's client registrations and conversions today?")["reply"]
    assert reply.lower().count("i don't have") >= 2
    assert "Client Registrations" in reply and "Conversions" in reply
