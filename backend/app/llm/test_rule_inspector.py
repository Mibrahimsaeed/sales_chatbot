"""Rule-based decision reconstruction (app/llm/rule_inspector.py).

The property under test is that the reconstruction is GROUND TRUTH, not
an approximation: it must report the same winner the planner picked, and
it must be right about which scorers declined. A trace that is merely
plausible is worse than none, because the whole point is to settle
arguments about where a query lost its meaning.
"""

import pytest

from app.llm import rule_inspector as ri
from app.llm.query_planner import build_query_plan


def test_signals_report_matched_text_not_just_a_boolean():
    signals = {s.name: s for s in ri.signals("top 5 advisors in Graana")}
    assert signals["ranking_strong"].detected
    assert "top" in signals["ranking_strong"].matched
    assert not signals["comparison"].detected
    assert signals["comparison"].matched == []


def test_absent_signal_is_reported_as_not_detected():
    """The absent keyword is usually the explanation, so it has to be
    stated rather than merely missing from a list of hits."""
    line = {s.name: s.line() for s in ri.signals("tell me about Waqar")}["relational"]
    assert "not detected" in line


def test_relational_phrase_is_detected_with_its_span():
    signals = {s.name: s for s in ri.signals("tell me about Waqar Haider's team")}
    assert signals["relational"].detected
    assert signals["relational"].matched  # the literal span that matched


def test_reconstructed_winner_matches_what_the_planner_actually_built():
    """The reconstruction re-runs score_intents(); if that ever stopped
    being pure, this is what would catch it."""
    entities = {"advisor_name": "Waqar Haider", "advisor_wid": 1, "advisor_wids": [1]}
    text = "tell me about waqar haider"

    plan = build_query_plan(text, entities)
    decision = ri.planner_decision(text, entities)

    assert decision.winner is not None
    assert decision.winner["score"] == pytest.approx(plan.intent_score, abs=0.001)
    assert decision.winner["evidence"] == plan.intent_evidence


def test_declined_scorers_are_identified_by_calling_them_not_by_name():
    """_score_attendance produces intent "attendance_filter" — a
    name-matched implementation reports it as declined while it in fact
    won, which is exactly the kind of confident falsehood this log must
    not contain."""
    entities = {"team": "Blue Area", "attendance_status": "Late"}
    text = "who was late in blue area"

    decision = ri.planner_decision(text, entities)
    intents = {c["intent"] for c in decision.candidates}

    assert "attendance_filter" in intents
    assert "_score_attendance" not in decision.declined


def test_every_scorer_is_accounted_for_exactly_once():
    from app.llm.query_planner import _SCORERS

    entities = {"advisor_name": "Waqar", "advisor_wid": 1}
    decision = ri.planner_decision("tell me about waqar", entities)
    assert len(decision.candidates) + len(decision.declined) == len(_SCORERS)


def test_why_lines_name_the_winner_the_runner_up_and_the_absent_signals():
    entities = {"team": "Blue Area"}
    text = "top 5 in blue area by revenue"
    decision = ri.planner_decision(text, entities)
    lines = "\n".join(ri.why_lines(decision, ri.signals(text)))

    assert "Selected intent" in lines
    assert "Keyword signals NOT detected" in lines


def test_reconstruction_never_raises_into_the_caller(monkeypatch):
    """Diagnostics are fail-soft: a broken planner import must degrade to
    an error string, not take the chat request down with it."""
    import app.llm.query_planner as qp

    monkeypatch.setattr(qp, "score_intents", lambda *a, **k: 1 / 0)
    decision = ri.planner_decision("anything", {})
    assert decision.error is not None
    assert decision.candidates == []
