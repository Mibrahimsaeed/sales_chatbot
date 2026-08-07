"""A metric's name in a sentence about a period it does not have.

A metric key is ONE measure at ONE period, and its label says so:
`total_connects` is "Total MTD Connects". That is the right caption on an
answer — the reply states the window it computed. It is the wrong name
inside the unavailable-period sentence, which was:

    "I don't have daily figures for Total MTD Connects yet
     — I hold MTD, YTD totals for it."

Two windows in one sentence, and the reader has to work out that only one
of them is their question. Worse, the MTD in it is not something they
asked for: it is which key the alias table happened to resolve. A user
asking for "daily connects" never said MTD anywhere.

periods.without_period() derives the neutral form from the metric's OWN
declared period, so the two names cannot drift and no metric carries a
second label maintained by hand. The exhaustive test at the bottom is
what makes that derivation trustworthy — it checks every metric in the
ontology, so a new one with period-worded label wording is caught here
rather than in a reply.

Deliberately narrow: metric_label() is UNCHANGED and still names the
period, because an answer's caption should. Only the sentence about an
absent period goes neutral.
"""

import pytest

from app.llm.metric_ontology import METRICS, measure_label, metric_label
from app.llm.periods import PERIOD_TOKENS, without_period
from app.llm.query_ir import MetricRef, QueryIR, Sort, TimeRange
from app.services.chat_service import _unanswerable_reply


def _ir(metric_key, period):
    return QueryIR(intent="leaderboard", metric=MetricRef(key=metric_key),
                   sort=Sort(metric=metric_key), time_range=TimeRange(period=period))


# ---------------------------------------------------------------------
# The neutral name, per label shape
# ---------------------------------------------------------------------


@pytest.mark.parametrize("key,expected", [
    # infix — the reported case
    ("total_connects", "Total Connects"),
    ("ytd_connects", "Total Connects"),
    # prefix
    ("client_registrations", "Client Registrations"),
    ("ytd_client_registrations", "Client Registrations"),
    ("mtd_cleared", "Revenue Cleared"),
    # parenthesised suffix
    ("cr_rate", "CR %"),
    ("ytd_cr_rate", "CR %"),
    ("answered_calls", "Answered Calls"),
    # the spelled-out 3M form
    ("three_month_cleared", "Revenue Cleared"),
    # a label that is ONLY its period plus one word
    ("mtd_target", "Target"),
])
def test_the_neutral_name_drops_the_period_whatever_shape_it_takes(key, expected):
    assert measure_label(key) == expected


def test_a_label_naming_no_period_is_unchanged():
    assert measure_label("achievement_pct") == "Target Achievement %"
    assert measure_label("portfolio_value") == "Portfolio Value"


@pytest.mark.parametrize("family_keys", [
    ("total_connects", "ytd_connects"),
    ("client_registrations", "ytd_client_registrations"),
    ("mtd_cleared", "ytd_cleared", "three_month_cleared"),
    ("cr_rate", "ytd_cr_rate"),
    ("overdue", "ytd_overdue"),
    ("pipeline_value", "ytd_pipeline_value"),
])
def test_every_member_of_a_period_family_has_the_same_neutral_name(family_keys):
    """A family IS one measure at several windows, so its members must
    agree on what that measure is called. Disagreement would mean the
    same unavailable question got two different answers depending on
    which key the alias table resolved."""
    names = {measure_label(k) for k in family_keys}
    assert len(names) == 1, f"{family_keys} disagree: {names}"


def test_removal_that_would_empty_the_label_keeps_it():
    """A metric named only for its period must stay visible, not blank."""
    assert without_period("MTD", "MTD") == "MTD"


def test_an_unknown_metric_key_is_returned_as_is():
    assert measure_label("not_a_metric") == "not_a_metric"
    assert measure_label(None) == "value"


# ---------------------------------------------------------------------
# The answer caption is deliberately NOT neutral
# ---------------------------------------------------------------------


@pytest.mark.parametrize("key,expected", [
    ("total_connects", "Total MTD Connects"),
    ("ytd_connects", "Total YTD Connects"),
    ("client_registrations", "MTD Client Registrations"),
])
def test_metric_label_still_names_the_period(key, expected):
    """The caption on a real answer states the window it computed —
    that is not the defect and must not change with it."""
    assert metric_label(key) == expected


# ---------------------------------------------------------------------
# The unavailable-period sentence
# ---------------------------------------------------------------------


# Phase 12 note: connects gained a real DAILY source, so a connects
# example must use 3M — the window that family still lacks — for these to
# be testing an UNAVAILABLE period at all. The label property is
# unchanged and so is the coverage of it; only the window moved.
@pytest.mark.parametrize("key,period,measure", [
    ("total_connects", "3M", "Total Connects"),
    ("ytd_connects", "3M", "Total Connects"),
    ("client_registrations", "DAILY", "Client Registrations"),
    ("cr_rate", "DAILY", "CR %"),
    ("conversion", "DAILY", "Conversions"),
])
def test_the_unavailable_reply_names_the_measure_without_a_second_period(
        key, period, measure):
    reply = _unanswerable_reply(_ir(key, period))
    assert measure in reply
    assert f"{measure} yet" in reply, "the measure must read as one phrase"


@pytest.mark.parametrize("key,period,window", [
    ("client_registrations", "DAILY", "daily"),
    ("conversion", "DAILY", "daily"),
    ("total_connects", "3M", "3-month"),
])
def test_the_requested_period_is_still_reported_correctly(key, period, window):
    """Making the LABEL neutral must not make the SENTENCE vague — the
    window the user asked for is the part they can act on."""
    reply = _unanswerable_reply(_ir(key, period))
    assert f"I don't have {window} figures" in reply


@pytest.mark.parametrize("key,period,available", [
    # narrowest to widest, so the list never reads as arbitrary
    ("total_connects", "3M", "DAILY, MTD, YTD"),
    ("client_registrations", "DAILY", "MTD, YTD"),
    ("mtd_cleared", "DAILY", "MTD, YTD, 3M"),
    ("meeting_rate", "DAILY", "MTD"),
])
def test_the_reply_still_lists_the_windows_that_do_exist(key, period, available):
    reply = _unanswerable_reply(_ir(key, period))
    assert f"I hold {available} totals" in reply


def test_the_reported_examples_read_exactly_as_specified():
    assert _unanswerable_reply(_ir("total_connects", "3M")) == (
        "I don't have 3-month figures for Total Connects yet — I hold "
        "DAILY, MTD, YTD totals for it. Ask for one of those and I can answer."
    )
    assert _unanswerable_reply(_ir("client_registrations", "DAILY")) == (
        "I don't have daily figures for Client Registrations yet — I hold "
        "MTD, YTD totals for it. Ask for one of those and I can answer."
    )


# ---------------------------------------------------------------------
# The drift guard
# ---------------------------------------------------------------------


def test_no_metric_in_the_ontology_has_a_period_left_in_its_neutral_name():
    """The whole ontology, not a sample. This is what lets the neutral
    name be DERIVED rather than declared per metric: a new metric — or a
    relabelled existing one — whose wording the derivation cannot handle
    fails here instead of shipping a two-period sentence."""
    offenders = {
        key: measure_label(key)
        for key in METRICS
        if any(token in measure_label(key)
               for tokens in PERIOD_TOKENS.values() for token in tokens)
    }
    assert offenders == {}


def test_every_metric_produces_a_non_empty_neutral_name():
    for key in METRICS:
        assert measure_label(key).strip(), key


def test_no_unavailable_reply_in_the_ontology_names_two_periods():
    """End to end over every metric: ask each one for a window it does
    not have and assert the sentence names exactly the requested one."""
    from app.llm.metric_ontology import supported_periods

    for key in METRICS:
        held = {p.value for p in supported_periods(key)}
        for period in ("DAILY", "MTD", "YTD", "3M"):
            if period in held:
                continue
            reply = _unanswerable_reply(_ir(key, period))
            measure = measure_label(key)
            assert f"for {measure} yet" in reply, (key, period, reply)
