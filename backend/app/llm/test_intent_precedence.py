"""Phase 7 — intent precedence regression tests.

The Phase 6 audit found four defects with one cause: **a grounded subject
was not evidence in intent selection.** Scorers both proposed AND
suppressed, so a candidate that would have won was often never proposed
— and a candidate that is never proposed cannot lose, and cannot be
explained.

Each test names the precedence rule it pins. The unit tests assert the
table directly (no database); the end-to-end tests assert that the winner
survives all the way to the reply.
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
from app.llm import entity_extractor, intent_precedence as ip, routing
from app.llm.entity_extractor import extract_entities
from app.llm.query_planner import build_query_plan, score_intents, _evidence_for
from app.services.chat_service import handle_chat_message


class _Cand:
    """Stand-in for a proposed candidate: rank() only reads these."""

    def __init__(self, intent, score):
        self.intent = intent
        self.score = score


def _rank(intents_and_scores, **evidence):
    cands = [_Cand(i, s) for i, s in intents_and_scores]
    return ip.rank(cands, ip.Evidence(**evidence))


# ---------------------------------------------------------------------
# The precedence table, in isolation
# ---------------------------------------------------------------------


def test_two_subjects_with_a_measure_is_a_comparison_without_the_word():
    """P2. "Blue Area and Downtown revenue" named both sides and a
    measure; requiring the word "compare" dropped one of them."""
    r = _rank([("leaderboard", 0.98), ("comparison", 0.56)],
              named_groups=2, metric=True)
    assert r.winner.intent == "comparison"
    assert r.rule is not None


def test_one_advisor_and_a_measure_beats_a_ranking_word():
    """P1. A person is a LEAF — there is nothing inside them to rank, so
    "top revenue for Omar Farooq" can only mean his figure."""
    r = _rank([("leaderboard", 0.98), ("advisor_metric", 0.80)],
              named_advisors=1, metric=True, ranking_phrase=True)
    assert r.winner.intent == "advisor_metric"


def test_one_group_and_a_measure_is_a_group_metric():
    """P4."""
    r = _rank([("leaderboard", 0.48), ("group_metric", 0.68)],
              named_groups=1, metric=True, group_level="team")
    assert r.winner.intent == "group_metric"


def test_a_ranking_word_over_a_group_enumerates_its_members():
    """The one asymmetry, and not a special case: a group CONTAINS
    members, so a ranking word means rank what is inside it."""
    r = _rank([("leaderboard", 0.98), ("group_metric", 0.68)],
              named_groups=1, metric=True, ranking_phrase=True, group_level="team")
    assert r.winner.intent == "leaderboard"


def test_an_explicit_inner_level_word_enumerates_the_members():
    """P3. "advisors in Blue Area by revenue" names the group AND the
    level to enumerate inside it."""
    r = _rank([("roster", 0.95), ("leaderboard", 0.48), ("group_metric", 0.68)],
              named_groups=1, metric=True, roster_phrase=True,
              level_word="advisor", group_level="team")
    assert r.winner.intent == "leaderboard"


def test_a_roster_needs_no_measure_and_no_ranking():
    r = _rank([("roster", 0.95), ("entity_summary", 0.4)],
              named_groups=1, roster_phrase=True, level_word="advisor",
              group_level="team")
    assert r.winner.intent == "roster"


def test_a_person_without_a_measure_is_a_profile():
    r = _rank([("advisor_profile", 0.5)], named_advisors=1)
    assert r.winner.intent == "advisor_profile"


def test_ambiguity_is_not_multiplicity():
    """One reference matching several people is ONE unresolved subject,
    not several named ones. Treating it as absent triggered the
    no-subject leaderboard rule and answered about everybody."""
    r = _rank([("clarify_person", 0.99), ("leaderboard", 0.48)],
              named_advisors=1, metric=True, ambiguous_subject=True)
    assert r.winner.intent == "clarify_person"


def test_precedence_never_invents_an_intent():
    """A rule may only promote a candidate some scorer actually
    proposed — otherwise it could name a plan nothing can build."""
    r = _rank([("leaderboard", 0.48)], named_groups=1, metric=True,
              group_level="team")
    assert r.winner.intent == "leaderboard"


def test_a_specialised_intent_is_left_alone():
    """The table governs the subject/measure family only. "Who is X's
    unit head" is not a weaker version of "X's revenue"."""
    r = _rank([("reverse_hierarchy", 0.98), ("advisor_profile", 0.5)],
              named_advisors=1, reverse_phrase=True)
    assert r.winner.intent == "reverse_hierarchy"
    assert r.rule is None


def test_no_candidates_yields_no_winner():
    assert ip.rank([], ip.Evidence()).winner is None


def test_every_ranking_explains_itself():
    for kwargs in (dict(named_groups=2, metric=True),
                   dict(named_advisors=1, metric=True),
                   dict(named_groups=1, metric=True, group_level="team"),
                   dict(metric=True, ranking_phrase=True),
                   dict(named_advisors=1)):
        r = _rank([("leaderboard", 0.9), ("comparison", 0.8),
                   ("advisor_metric", 0.7), ("group_metric", 0.6),
                   ("advisor_profile", 0.5)], **kwargs)
        assert r.trace()
        assert r.winner is not None


def test_the_ranking_records_why_each_loser_lost():
    r = _rank([("leaderboard", 0.98), ("advisor_metric", 0.80)],
              named_advisors=1, metric=True, ranking_phrase=True)
    assert any(intent == "leaderboard" for intent, _why in r.rejected)
    assert all(why for _intent, why in r.rejected)


def test_the_table_is_deterministic():
    kwargs = dict(named_groups=1, metric=True, group_level="team")
    a = _rank([("leaderboard", 0.48), ("group_metric", 0.68)], **kwargs)
    b = _rank([("leaderboard", 0.48), ("group_metric", 0.68)], **kwargs)
    assert a.winner.intent == b.winner.intent
    assert a.trace() == b.trace()


# ---------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------

_PEOPLE = [
    (1, "Yasir Ali", "Blue Area", "Graana", "Beverly Center", "North/KPK",
     "Usman Ghani", "Fawad Hafeez", "Tariq Mehmood"),
    (2, "Waqar Haider", "Blue Area", "Graana", "Beverly Center", "North/KPK",
     "Usman Ghani", "Fawad Hafeez", "Tariq Mehmood"),
    (3, "Sana Tariq", "Blue Area", "Graana", "Beverly Center", "North/KPK",
     "Rabia Anjum", "Fawad Hafeez", "Tariq Mehmood"),
    (4, "Shehryar Abbasi", "Downtown", "Graana", "Gold Crest", "Central",
     "Rabia Anjum", "Adeel Aslam", "Tariq Mehmood"),
    (5, "Omar Farooq", "Downtown", "IMARAT", "Gold Crest", "Central",
     "Bilal Qadir", "Adeel Aslam", "Sadia Rehman"),
]


@pytest.fixture(scope="module")
def _ip_engine():
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
    for team, target, achieved in (("Blue Area", 3000, 2400), ("Downtown", 2000, 1100)):
        s.add(TeamTarget(team=team, target=target, achieved=achieved,
                         achievement_pct=round(achieved / target * 100, 1)))
    s.commit()
    with engine.begin() as conn:
        conn.exec_driver_sql(_ADVISOR_PROFILE_VIEW)
    yield engine
    s.close()


@pytest.fixture()
def org(_ip_engine, monkeypatch):
    from app.core.config import settings
    from app.llm import llm_client, semantic_parser

    monkeypatch.setattr(settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first", raising=False)
    monkeypatch.setattr(settings, "use_llm_planner", False, raising=False)
    monkeypatch.setattr(llm_client, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)

    s = sessionmaker(bind=_ip_engine)()
    entity_extractor._cache["loaded_at"] = 0
    yield s
    s.close()


def _action(org, query):
    return build_query_plan(query, extract_entities(query, org)).action


# (query, expected plan action)
MATRIX = [
    # advisor + metric, with and without a ranking word
    ("What is Yasir Ali's revenue?", "advisor_metric"),
    ("How many conversions does Sana Tariq have?", "advisor_metric"),
    ("Top revenue for Omar Farooq", "advisor_metric"),
    ("Highest conversions for Omar Farooq", "advisor_metric"),
    # group + metric
    ("Blue Area revenue", "group_metric"),
    ("Downtown pipeline", "group_metric"),
    ("Graana conversions", "group_metric"),
    ("North/KPK overdue", "group_metric"),
    ("Beverly Center connects", "group_metric"),
    # group + metric + ranking -> enumerate members
    ("Top revenue in Blue Area", "leaderboard"),
    ("Top 5 advisors in Blue Area by revenue", "leaderboard"),
    # roster
    ("advisors in Blue Area", "roster"),
    ("advisors in Blue Area by revenue", "leaderboard"),
    # leaderboard, no subject
    ("Top advisors by revenue", "leaderboard"),
    ("Lowest revenue", "leaderboard"),
    # comparison, with and without the word
    ("Compare Blue Area and Downtown on revenue", "comparison"),
    ("Blue Area and Downtown revenue", "comparison"),
    ("Blue Area vs Downtown revenue", "comparison"),
    ("Who has more revenue, Blue Area or Downtown?", "comparison"),
    ("Compare Yasir Ali and Sana Tariq on cleared", "comparison"),
    # profile / hierarchy are untouched
    ("Tell me about Yasir Ali", "lookup"),
    ("Who does Yasir Ali report to?", "reverse_hierarchy"),
]


@pytest.mark.parametrize("query,expected", MATRIX, ids=[q for q, _ in MATRIX])
def test_the_classification_matrix(org, query, expected):
    assert _action(org, query) == expected


def test_a_named_advisor_is_answered_about_that_advisor(org):
    """The P1 defect in its worst form: the reply named a DIFFERENT
    person. Omar Farooq is last by revenue, so a leaderboard reading
    cannot accidentally look correct."""
    response = handle_chat_message(org, "Top revenue for Omar Farooq", session_id=None)
    assert "Omar Farooq" in response["reply"]
    assert "Yasir Ali" not in response["reply"]


def test_two_named_groups_are_both_answered_about(org):
    response = handle_chat_message(org, "Blue Area and Downtown revenue", session_id=None)
    assert response["type"] == "comparison"
    assert "Blue Area" in response["reply"] and "Downtown" in response["reply"]


def test_a_ranked_roster_is_ranked(org):
    response = handle_chat_message(org, "advisors in Blue Area by revenue",
                                   session_id=None)
    assert response["type"] == "leaderboard"


def test_a_group_metric_answers_with_the_groups_own_figure(org):
    from app.llm import aggregation

    truth = aggregation.metric_value(org, "team", "Blue Area", "mtd_cleared")
    response = handle_chat_message(org, "Blue Area revenue", session_id=None)
    assert response["type"] == "metric_value"
    assert f"{truth:,.0f}" in response["reply"]


def test_group_metric_does_not_depend_on_downstream_compensation(org):
    """P4 — the intent is explicit now, not inferred from a leaderboard
    that happens to return one row."""
    assert _action(org, "Blue Area revenue") == "group_metric"


# ---------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------


def test_the_trace_names_the_winner_the_rule_and_the_losers(org):
    handle_chat_message(org, "Top revenue for Omar Farooq", session_id=None)
    step = next(s for s in routing.current_trace().steps if s.stage == "Intent")

    assert step.chose == "advisor_metric"
    assert "precedence:" in step.why
    assert "not leaderboard" in step.why


def test_every_matrix_query_records_an_intent_decision(org):
    for query, _expected in MATRIX:
        handle_chat_message(org, query, session_id=None)
        steps = [s for s in routing.current_trace().steps if s.stage == "Intent"]
        assert steps, f"{query!r} recorded no intent decision"
        assert steps[0].why


# ---------------------------------------------------------------------
# Single ownership
# ---------------------------------------------------------------------


def test_scorers_do_not_suppress_on_a_rival_intents_evidence(org):
    """advisor_metric must be PROPOSED even when a ranking word is
    present. It used to return None, so the ranking was never a contest
    and the outcome could not be explained."""
    query = "Top revenue for Omar Farooq"
    _ctx, candidates = score_intents(query, extract_entities(query, org))
    assert "advisor_metric" in {c.intent for c in candidates}
    assert "leaderboard" in {c.intent for c in candidates}


def test_the_pipeline_does_not_hardcode_an_action_when_inheriting_a_metric():
    """The context-inheritance path used to write plan.action directly,
    making the pipeline a second owner of intent — and hardcoding
    "leaderboard", which is wrong for a subject-only follow-up."""
    import inspect

    from app.llm import nlu_pipeline

    # Comment lines are stripped first: this module's own comments
    # QUOTE the removed line to explain why it went, and matching prose
    # would make the assertion pass or fail on documentation.
    source = inspect.getsource(nlu_pipeline.resolve)
    code = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("#"))
    assert 'plan.action = "leaderboard"' not in code
