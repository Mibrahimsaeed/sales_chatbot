"""Phase 10 — the conversation corpus.

Phase 2 (test_conversation_context.py) established that a MERGE exists and
is correct. This module tests the thing one level up: that every turn
which produces an answer actually reaches it.

The Phase 10 audit found that it did not. `merge()` runs on the IR path,
and the rule-based PLAN path — advisor profiles, entity summaries,
rosters, hierarchy chains — neither read the conversation nor wrote to
it. Both halves of that were silent:

    "Blue Area revenue" -> "what about Downtown?"
      planned as a bare entity mention, answered with a generic Downtown
      card. The measure named one message earlier was simply gone.

    ... -> "and Graana?"
      turn 2 had left no record of itself, so turn 3 inherited from turn
      ONE and the conversation forked without saying so.

Each test below is a CHAIN, asserted on the executed IR rather than on
the reply text, because every one of these failures returned a real
number from a real scope — just not the scope the conversation had
established. A test asserting "an answer came back" stays green through
all of it.
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
from app.llm import conversation_memory, entity_extractor, nlu_pipeline, routing

PEOPLE = [
    (1, "Yasir Ali", "Blue Area", "Graana", "Beverly Center", "North/KPK"),
    (2, "Waqar Haider", "Blue Area", "Graana", "Beverly Center", "North/KPK"),
    (3, "Shehryar Abbasi", "Downtown", "Graana", "Gold Crest", "Central"),
    (4, "Hina Malik", "Downtown", "IMARAT", "Gold Crest", "Central"),
    (5, "Nadia Sheikh", "Gulberg", "IMARAT", "Emporium", "South"),
]


@pytest.fixture(scope="module")
def _engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    for wid, name, team, company, office, region in PEOPLE:
        s.add(Advisor(wid=wid, name=name, team=team, company=company,
                      rm="Tariq Mehmood", portfolio_lead="Fawad Hafeez",
                      management_lead="Usman Ghani", office=office,
                      region=region, unit="A", in_master_sheet=True))
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
        s.add(Attendance(wid=wid, biometric_mtd_ontime=15, biometric_mtd_late=5,
                         biometric_mtd_not_marked=0, login_mtd_ontime=18,
                         login_mtd_late=2, login_mtd_not_marked=0))
    for team, target, achieved in (("Blue Area", 3000, 2400), ("Downtown", 2000, 1100),
                                   ("Gulberg", 1500, 700)):
        s.add(TeamTarget(team=team, target=target, achieved=achieved,
                         achievement_pct=round(achieved / target * 100, 1)))
    s.commit()
    yield engine
    s.close()


@pytest.fixture()
def org(_engine, monkeypatch):
    """The deterministic pipeline: no LLM planner, no LLM parser.

    Every assertion below must hold with the model switched off — a
    conversation that only holds together when an LLM is reachable is not
    a conversation the system understands.
    """
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


# ---------------------------------------------------------------------
# Reading a chain
# ---------------------------------------------------------------------


def _chain(db, session, turns):
    """Run a conversation, returning the state AFTER EVERY TURN.

    Every turn is captured, not just the last, because the failures this
    module exists to catch are divergences: the answer stays plausible
    while the scope quietly stops being the one under discussion, and
    only a turn-by-turn record shows which turn it happened on.
    """
    seen = []
    for turn in turns:
        resolution = nlu_pipeline.resolve(turn, db, session_id=session)
        seen.append((turn, resolution, conversation_memory.get(session)))
    return seen


def _scope(resolution):
    if resolution.kind != "ir":
        return set()
    return {(f.field, f.value) for f in resolution.ir.filters}


def _metric(resolution):
    if resolution.kind == "ir" and resolution.ir.metric:
        return resolution.ir.metric.key
    plan = getattr(resolution, "plan", None)
    return plan.metric if plan else None


def _subjects(resolution):
    if resolution.kind != "ir":
        return set()
    return {(s.type, s.value) for s in resolution.ir.subjects}


def _last(chain):
    return chain[-1][1]


# ---------------------------------------------------------------------
# Metric continuation — the subject survives, the measure changes
# ---------------------------------------------------------------------


def test_a_metric_chain_keeps_the_subject_on_every_turn(org):
    chain = _chain(org, "m1", ["Blue Area revenue", "What about pipeline?",
                               "And overdue?", "What about conversions?"])
    metrics = [_metric(r) for _, r, _ in chain]
    assert metrics == ["mtd_cleared", "pipeline_value", "overdue", "conversion"]
    for turn, resolution, _ in chain:
        assert ("team", "Blue Area") in _scope(resolution), f"{turn!r} lost the team"


# ---------------------------------------------------------------------
# Period continuation — the subject and measure survive
# ---------------------------------------------------------------------


def test_a_period_follow_up_changes_only_the_window(org):
    chain = _chain(org, "p1", ["Blue Area revenue this month",
                               "What about year to date?"])
    final = _last(chain)
    assert final.ir.time_range.period == "YTD"
    assert ("team", "Blue Area") in _scope(final)
    # the measure follows the window: "revenue YTD" is ytd_cleared, and
    # relabelling MTD numbers as YTD is the failure this pins
    assert _metric(final) == "ytd_cleared"


# ---------------------------------------------------------------------
# Subject continuation — the measure survives, the subject changes
# ---------------------------------------------------------------------


def test_a_bare_new_subject_inherits_the_measure(org):
    """THE Phase 10 defect. "what about Downtown?" plans as a bare entity
    mention, which is a complete question on its own and was answered as
    one — with a generic entity card, one message after the user said
    which measure they cared about."""
    chain = _chain(org, "s1", ["Blue Area revenue", "What about Downtown?"])
    final = _last(chain)
    assert final.kind == "ir"
    assert _metric(final) == "mtd_cleared"
    assert ("team", "Downtown") in _scope(final)


def test_a_new_subject_at_the_same_level_replaces_rather_than_intersects(org):
    """Blue Area AND Downtown matches nobody, and returns no rows without
    saying why."""
    final = _last(_chain(org, "s2", ["Blue Area revenue", "What about Downtown?"]))
    assert [v for f, v in _scope(final) if f == "team"] == ["Downtown"]


def test_a_subject_chain_does_not_fork_after_the_second_turn(org):
    """Turn 2 answered on the plan path and stored nothing, so turn 3
    inherited from turn ONE — the conversation carried on from a state
    the user had already left, and nothing in the reply said so."""
    chain = _chain(org, "s3", ["Blue Area revenue", "What about Downtown?",
                               "And overdue?"])
    final = _last(chain)
    assert _metric(final) == "overdue"
    assert ("team", "Downtown") in _scope(final)
    assert ("team", "Blue Area") not in _scope(final)


# ---------------------------------------------------------------------
# Filter continuation
# ---------------------------------------------------------------------


def test_no_filter_disappears_across_a_four_turn_chain(org):
    chain = _chain(org, "f1", ["Top advisors by revenue", "Only Graana",
                               "Top 5", "This month"])
    final = _last(chain)
    assert ("company", "Graana") in _scope(final)
    assert final.ir.limit == 5
    assert final.ir.time_range.period == "MTD"
    assert _metric(final) == "mtd_cleared"


def test_narrowing_at_a_new_level_joins_rather_than_replaces(org):
    """"only IMARAT" after a Downtown question narrows Downtown; it does
    not move the question to IMARAT."""
    chain = _chain(org, "f2", ["Downtown revenue", "Only IMARAT", "Top 5"])
    final = _last(chain)
    assert ("team", "Downtown") in _scope(final)
    assert ("company", "IMARAT") in _scope(final)
    assert final.ir.limit == 5


# ---------------------------------------------------------------------
# Comparison — both sides survive every follow-up
# ---------------------------------------------------------------------


def test_a_comparison_keeps_both_sides_through_four_follow_ups(org):
    chain = _chain(org, "c1", ["Compare Blue Area and Downtown on revenue",
                               "What about overdue?", "Year to date", "Only Graana"])
    for turn, resolution, _ in chain:
        assert resolution.ir.intent == "comparison", f"{turn!r} stopped being a comparison"
        assert _subjects(resolution) == {("team", "Blue Area"), ("team", "Downtown")}, \
            f"{turn!r} lost a side"
    final = _last(chain)
    assert final.ir.time_range.period == "YTD"
    assert ("company", "Graana") in _scope(final)


def test_a_narrowed_comparison_stays_grouped_at_its_subjects_level(org):
    """"only Graana" names a COMPANY, but the sides being compared are
    teams. Letting the filter set the level grouped two teams by company:
    one row, named after the filter, presented as the comparison."""
    final = _last(_chain(org, "c2", ["Compare Blue Area and Downtown on revenue",
                                     "Only Graana"]))
    assert final.ir.subject_level == "team"


def test_a_comparison_never_intersects_its_own_sides(org):
    """A scope filter inherited at the level the sides occupy matches one
    of them and silently answers half the question."""
    final = _last(_chain(org, "c3", ["Blue Area overdue", "compare with Downtown"]))
    assert final.ir.intent == "comparison"
    assert _subjects(final) == {("team", "Blue Area"), ("team", "Downtown")}
    assert not [f for f, _ in _scope(final) if f == "team"]


def test_a_comparison_missing_a_side_takes_it_from_the_conversation(org):
    """"compare with Downtown" grounds one target. The other is the
    subject the conversation has already established — refusing with
    "I need two things to compare" asks for something already said."""
    final = _last(_chain(org, "c4", ["Blue Area revenue", "compare with Downtown"]))
    assert final.kind == "ir"
    assert final.ir.intent == "comparison"
    assert _metric(final) == "mtd_cleared"


# ---------------------------------------------------------------------
# Topic change — context must NOT survive
# ---------------------------------------------------------------------


def test_a_standalone_question_inherits_nothing(org):
    final = _last(_chain(org, "x1", ["Blue Area revenue", "Top advisors by conversions"]))
    assert _scope(final) == set()
    assert _metric(final) == "conversion"


def test_a_person_lookup_does_not_leak_the_previous_scope_into_the_next_turn(org):
    """The leak was one turn late: the profile answered correctly and
    left the team question in place as the follow-up base, so the turn
    AFTER it silently went back to Blue Area."""
    chain = _chain(org, "x2", ["Blue Area revenue", "What is Yasir Ali's performance?",
                               "Top 5"])
    assert chain[1][2] is None, "the profile turn left a stale follow-up base"
    assert _scope(_last(chain)) == set()


def test_answering_off_the_plan_path_clears_the_follow_up_base(org):
    _chain(org, "x3", ["Blue Area revenue", "Tell me about Downtown's advisors"])
    assert conversation_memory.get("x3") is None


def test_a_clarification_does_not_clear_the_follow_up_base(org):
    """"compare with Nowhere" cannot be answered; the conversation has
    not moved, and the context the user is being asked to complete must
    still be there when they do."""
    _chain(org, "x4", ["Blue Area revenue", "compare Nowhere and Elsewhere"])
    prior = conversation_memory.get("x4")
    assert prior is not None
    assert prior.metric.key == "mtd_cleared"


# ---------------------------------------------------------------------
# Long chains
# ---------------------------------------------------------------------


TEN_TURNS = ["Blue Area revenue", "pipeline", "year to date", "top 5",
             "only Graana", "overdue", "this month", "compare with Downtown",
             "conversions", "top 3"]


def test_a_ten_turn_chain_never_loses_an_established_field(org):
    chain = _chain(org, "L1", TEN_TURNS)
    for turn, resolution, _ in chain:
        assert resolution.kind == "ir", f"{turn!r} fell off the IR path"
        assert resolution.ir.metric is not None, f"{turn!r} lost the measure"

    final = _last(chain)
    assert _metric(final) == "conversion"
    assert final.ir.limit == 3
    assert final.ir.time_range.period == "MTD"
    assert ("company", "Graana") in _scope(final)
    assert _subjects(final) == {("team", "Blue Area"), ("team", "Downtown")}


def test_a_twenty_turn_chain_ends_on_exactly_the_query_the_last_turns_describe(org):
    """The tail is a deliberate topic change followed by three
    modifiers. Every field of the final query is named in the last four
    turns, so anything else surviving from the first sixteen is a leak."""
    chain = _chain(org, "L2", TEN_TURNS + [
        "what about Gulberg?", "revenue", "top 10", "only IMARAT", "connects",
        "year to date", "top advisors by conversions", "this month",
        "only Graana", "top 5"])
    final = _last(chain)

    assert _metric(final) == "conversion"
    assert final.ir.time_range.period == "MTD"
    assert final.ir.limit == 5
    assert _scope(final) == {("company", "Graana")}
    assert _subjects(final) == set(), "a comparison from sixteen turns ago survived"


def test_a_twenty_turn_conversation_does_not_grow_without_bound(org):
    """Structured state is ONE IR, not a transcript. A context window
    that grows with the conversation is the failure mode this design
    exists to avoid, and it is invisible until production."""
    from app.core.config import settings

    _chain(org, "L3", TEN_TURNS * 2)
    state = conversation_memory._store["L3"]
    assert len(state.turns) <= settings.conversation_window_turns * 2
    assert len(conversation_memory.get("L3").model_dump_json()) < 2000


# ---------------------------------------------------------------------
# Observability — every turn accounts for what it did with the context
# ---------------------------------------------------------------------


@pytest.mark.parametrize("follow_up", [
    "What about pipeline?",     # merged on the IR path
    "top 5",                    # patched deterministically
    "What about Downtown?",     # carried into the planner
    "Tell me about Yasir Ali",  # answered off the plan path
])
def test_every_follow_up_records_a_context_decision(org, follow_up):
    """The turns whose inheritance is most implicit had no Context step
    at all — the patcher and the plan path both merged (or dropped) state
    without saying so, which is exactly where a context bug hides."""
    _chain(org, f"o-{follow_up}", ["Blue Area revenue", follow_up])
    steps = [s for s in routing.current_trace().steps if s.stage == "Context"]
    assert steps, f"{follow_up!r} made a context decision and did not record it"
    assert any(s.why for s in steps), "a context decision with no reason"


def test_the_context_trace_names_the_previous_and_merged_state(org):
    _chain(org, "o2", ["Blue Area revenue", "What about pipeline?"])
    step = next(s for s in routing.current_trace().steps if s.stage == "Context")
    assert "previous:" in step.why and "merged:" in step.why
    assert "mtd_cleared" in step.why       # what it was
    assert "pipeline_value" in step.why    # what it became


# ---------------------------------------------------------------------
# Pronoun relations stay owned by the relation resolver
# ---------------------------------------------------------------------


def test_a_pronoun_follow_up_resolves_the_relation(org, monkeypatch):
    """"his team" is resolved by cross_turn_resolver, from the identity
    conversation_memory settled on the previous turn. Asserted here only
    to pin that the relation half reaches the right GROUP; deliberately
    no second pronoun resolver in the context layer, because that
    ownership already has an owner.

    Gated on RELATION_INFERENCE_ENABLED, which still ships default-off —
    so out of the box this follow-up resolves nothing.
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "relation_inference_enabled", True, raising=False)
    final = _last(_chain(org, "r1", ["Yasir Ali's revenue", "How is his team doing?"]))
    plan = getattr(final, "plan", None)
    assert (plan.level if plan else final.ir.subject_level) == "team"
    assert (plan.entity_value if plan else None) == "Blue Area"


def test_a_measure_asked_of_one_PERSON_does_not_carry_to_the_next_turn(org, monkeypatch):
    """A KNOWN LIMITATION, pinned so it stays visible.

    "Yasir Ali's revenue" is answered as `advisor_metric` on the
    rule-based plan path, which QueryIR cannot express — the IR has no
    way to say "this one person" as a scope, which is why that shape is
    on the plan path at all. So the turn leaves no structured state and
    the measure does not reach "how is his team doing", which answers
    with the team's summary card instead of the team's revenue.

    Storing a fabricated IR would be worse than storing none: the closest
    thing plan_to_ir can build for this plan is an advisor leaderboard
    with NO person in it, and the next "top 5" would then rank everybody
    while looking like a continuation. Closing this properly means
    representing a single-person metric in the IR — an extension to the
    IR, not a change to the merge — so it is recorded here rather than
    worked around in the context layer.
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "relation_inference_enabled", True, raising=False)
    chain = _chain(org, "r2", ["Yasir Ali's revenue", "How is his team doing?"])
    assert chain[0][2] is None, "an advisor_metric turn now stores an IR — update this"
    assert _metric(_last(chain)) is None
