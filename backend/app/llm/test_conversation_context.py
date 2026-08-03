"""Phase 2 — cross-turn context regression tests.

Every failure these lock down was silent. The chatbot answered with a
real number from a real scope; it was simply a different scope than the
conversation had established. A test asserting only "an answer came back"
stays green through all of it, so each of these asserts the MERGED IR —
which fields were inherited, which were overridden, and which were
dropped.
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
from app.llm import (
    conversation_context as ctx, conversation_memory, entity_extractor,
    nlu_pipeline, routing,
)
from app.llm.query_ir import Filter, MetricRef, QueryIR, Sort, TimeRange


# ---------------------------------------------------------------------
# TurnSpec — what a turn specified
# ---------------------------------------------------------------------


def test_a_turn_naming_a_measure_and_a_level_stands_alone():
    spec = ctx.TurnSpec(metric=True, level_word=True)
    assert spec.stands_alone


def test_a_turn_naming_a_measure_and_a_subject_stands_alone():
    assert ctx.TurnSpec(metric=True, subject=True).stands_alone


def test_a_turn_naming_a_measure_and_a_ranking_stands_alone():
    assert ctx.TurnSpec(metric=True, ranking=True).stands_alone


def test_a_measure_alone_does_not_stand_alone():
    """"pipeline" needs the previous turn to say pipeline OF WHAT."""
    assert not ctx.TurnSpec(metric=True).stands_alone


def test_a_subject_alone_does_not_stand_alone():
    """"only Graana" needs the previous turn to say Graana's WHAT."""
    assert not ctx.TurnSpec(subject=True).stands_alone


def test_a_bare_modifier_does_not_stand_alone():
    assert not ctx.TurnSpec(limit=True).stands_alone


# ---------------------------------------------------------------------
# Ellipsis — explicitly NOT plan.action
# ---------------------------------------------------------------------


def test_no_prior_turn_means_nothing_is_elliptical():
    d = ctx.ellipsis(ctx.TurnSpec(), has_prior=False)
    assert not d.is_elliptical


def test_a_self_standing_turn_is_not_elliptical():
    d = ctx.ellipsis(ctx.TurnSpec(metric=True, level_word=True), has_prior=True)
    assert not d.is_elliptical
    assert "stands as its own question" in d.why


def test_an_incomplete_turn_is_elliptical_and_says_what_is_missing():
    d = ctx.ellipsis(ctx.TurnSpec(subject=True), has_prior=True)
    assert d.is_elliptical
    assert "no measure" in d.why


def test_discourse_openers_are_stripped_without_emptying_the_text():
    assert ctx.strip_openers("now only IMARAT") == "only IMARAT"
    assert ctx.strip_openers("and revenue") == "revenue"
    assert ctx.strip_openers("what about pipeline") == "pipeline"
    assert ctx.strip_openers("only Graana") == "only Graana"
    # a message that is nothing BUT an opener keeps its last word
    assert ctx.strip_openers("now") == "now"


# ---------------------------------------------------------------------
# merge — field ownership
# ---------------------------------------------------------------------


def _ir(metric="mtd_cleared", period="MTD", level="advisor", filters=(), limit=10):
    return QueryIR(
        intent="leaderboard", subject_level=level,
        metric=MetricRef(key=metric), sort=Sort(metric=metric),
        time_range=TimeRange(mode="snapshot", period=period),
        filters=[Filter(field=f, operator="=", value=v) for f, v in filters],
        limit=limit,
    )


def _elliptical():
    return ctx.Ellipsis(True, "test")


def test_a_turn_that_stands_alone_inherits_nothing():
    prior = _ir(filters=[("team", "Blue Area")])
    current = _ir(metric="overdue")
    spec = ctx.TurnSpec(metric=True, level_word=True)
    result = ctx.merge(prior, current, spec, ctx.ellipsis(spec, True))

    assert result.ir.filters == []
    assert result.discarded


def test_a_missing_measure_is_inherited():
    prior = _ir(metric="pipeline_value")
    current = _ir(metric="mtd_cleared")
    result = ctx.merge(prior, current, ctx.TurnSpec(subject=True), _elliptical())

    assert result.ir.metric.key == "pipeline_value"
    assert any(f == "metric" for f, _ in result.inherited)


def test_a_named_measure_overrides_the_previous_one():
    prior = _ir(metric="mtd_cleared")
    current = _ir(metric="pipeline_value")
    result = ctx.merge(prior, current, ctx.TurnSpec(metric=True), _elliptical())

    assert result.ir.metric.key == "pipeline_value"
    assert any(f == "metric" for f, _ in result.overridden)


def test_a_missing_period_is_inherited():
    prior = _ir(period="YTD")
    current = _ir(period="MTD")
    result = ctx.merge(prior, current, ctx.TurnSpec(metric=True), _elliptical())

    assert result.ir.time_range.period == "YTD"


def test_a_named_period_overrides():
    prior = _ir(period="MTD")
    current = _ir(period="YTD")
    result = ctx.merge(prior, current, ctx.TurnSpec(period=True), _elliptical())

    assert result.ir.time_range.period == "YTD"


def test_scope_at_a_new_level_joins_rather_than_replaces():
    """"Downtown pipeline" then "now only IMARAT" narrows to advisors who
    are in BOTH — replacing the team scope would widen the answer."""
    prior = _ir(filters=[("team", "Downtown")])
    current = _ir(filters=[("company", "IMARAT")])
    result = ctx.merge(prior, current, ctx.TurnSpec(subject=True), _elliptical())

    fields = {(f.field, f.value) for f in result.ir.filters}
    assert ("team", "Downtown") in fields
    assert ("company", "IMARAT") in fields


def test_scope_at_the_same_level_replaces():
    """"Blue Area revenue" then "only Downtown" is a correction. Keeping
    both would intersect two teams and silently return nothing."""
    prior = _ir(filters=[("team", "Blue Area")])
    current = _ir(filters=[("team", "Downtown")])
    result = ctx.merge(prior, current, ctx.TurnSpec(subject=True), _elliptical())

    teams = [f.value for f in result.ir.filters if f.field == "team"]
    assert teams == ["Downtown"]


def test_a_missing_limit_is_inherited():
    prior = _ir(limit=3)
    current = _ir(limit=10)
    result = ctx.merge(prior, current, ctx.TurnSpec(metric=True), _elliptical())
    assert result.ir.limit == 3


def test_a_stated_limit_overrides():
    prior = _ir(limit=10)
    current = _ir(limit=3)
    result = ctx.merge(prior, current, ctx.TurnSpec(limit=True), _elliptical())
    assert result.ir.limit == 3


def test_merge_never_mutates_the_previous_ir():
    prior = _ir(filters=[("team", "Blue Area")])
    before = prior.model_dump_json()
    ctx.merge(prior, _ir(), ctx.TurnSpec(subject=True), _elliptical())
    assert prior.model_dump_json() == before


def test_every_merge_decision_carries_a_reason():
    prior = _ir(filters=[("team", "Blue Area")], limit=5, period="YTD")
    result = ctx.merge(prior, _ir(), ctx.TurnSpec(), _elliptical())
    for _field, why in result.inherited + result.overridden + result.discarded:
        assert why


def test_merge_with_no_prior_discards_everything():
    result = ctx.merge(None, _ir(), ctx.TurnSpec(), _elliptical())
    assert result.discarded == [("all", "no previous turn")]


# ---------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------

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


def _run(org, session, turns):
    """Run a conversation, returning the final resolution."""
    result = None
    for turn in turns:
        result = nlu_pipeline.resolve(turn, org, session_id=session)
    return result


def _scope(resolution):
    if resolution.kind != "ir":
        return None
    return {(f.field, f.value) for f in resolution.ir.filters}


def _metric(resolution):
    if resolution.kind == "ir" and resolution.ir.metric:
        return resolution.ir.metric.key
    plan = getattr(resolution, "plan", None)
    return plan.metric if plan else None


def test_filter_survives_a_later_limit_change(org):
    """Top advisors by revenue -> only Graana -> top 3."""
    r = _run(org, "c1", ["Top advisors by revenue", "Only Graana", "Top 3"])
    assert ("company", "Graana") in _scope(r)
    assert r.ir.limit == 3


def test_scope_survives_a_metric_switch(org):
    """Blue Area revenue -> what about pipeline?"""
    r = _run(org, "c2", ["Blue Area revenue", "What about pipeline?"])
    assert _metric(r) == "pipeline_value"
    assert ("team", "Blue Area") in _scope(r)


def test_a_bare_metric_follow_up_keeps_the_subject(org):
    """Blue Area revenue -> pipeline."""
    r = _run(org, "c3", ["Blue Area revenue", "pipeline"])
    assert ("team", "Blue Area") in _scope(r)


def test_narrowing_at_a_new_level_keeps_both_scopes(org):
    """Show Downtown pipeline -> now only IMARAT -> top 5."""
    r = _run(org, "c4", ["Show Downtown pipeline", "Now only IMARAT", "Top 5"])
    scope = _scope(r)
    assert ("team", "Downtown") in scope
    assert ("company", "IMARAT") in scope
    assert r.ir.limit == 5


def test_a_four_turn_chain_keeps_every_established_field(org):
    """Top advisors -> only IMARAT -> this month -> top 3."""
    r = _run(org, "c5", ["Top advisors by revenue", "Only IMARAT",
                         "this month", "Top 3"])
    assert ("company", "IMARAT") in _scope(r)
    assert r.ir.limit == 3
    assert r.ir.time_range.period == "MTD"


def test_a_same_level_subject_corrects_rather_than_intersects(org):
    r = _run(org, "c6", ["Blue Area revenue", "only Downtown"])
    teams = [v for f, v in _scope(r) if f == "team"]
    assert teams == ["Downtown"]


def test_a_period_follow_up_keeps_the_scope(org):
    r = _run(org, "c7", ["Blue Area revenue", "year to date"])
    assert ("team", "Blue Area") in _scope(r)
    assert r.ir.time_range.period == "YTD"


def test_a_complete_new_question_discards_the_previous_scope(org):
    """Context expiration: a turn that stands alone starts fresh."""
    r = _run(org, "c8", ["Blue Area revenue", "Top advisors by overdue"])
    assert _scope(r) == set()
    assert _metric(r) == "overdue"


def test_a_subject_only_follow_up_inherits_the_measure(org):
    """"now only IMARAT" after a pipeline question must not ask WHICH
    measure — the previous turn said it one message ago."""
    r = _run(org, "c9", ["Show Downtown pipeline", "Now only IMARAT"])
    assert r.kind == "ir"
    assert _metric(r) == "pipeline_value"


def test_the_first_turn_of_a_conversation_inherits_nothing(org):
    r = _run(org, "c10", ["Top advisors by revenue"])
    assert _scope(r) == set()


def test_two_sessions_do_not_share_context(org):
    _run(org, "c11a", ["Blue Area revenue"])
    r = _run(org, "c11b", ["Top advisors by revenue"])
    assert _scope(r) == set()


# ---------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------


def test_the_trace_records_what_was_inherited_and_why(org):
    _run(org, "t1", ["Blue Area revenue", "What about pipeline?"])
    step = next(s for s in routing.current_trace().steps if s.stage == "Context")

    assert "inherited" in step.chose
    assert "Blue Area" in step.why          # the field's value
    assert "still applies" in step.why      # the reason it survived


def test_the_trace_records_an_override(org):
    _run(org, "t2", ["Blue Area revenue", "What about pipeline?"])
    step = next(s for s in routing.current_trace().steps if s.stage == "Context")
    assert "overridden" in step.chose
    assert "metric" in step.chose


def test_the_trace_records_a_discard_on_a_topic_change(org):
    _run(org, "t3", ["Blue Area revenue", "Top advisors by overdue"])
    step = next(s for s in routing.current_trace().steps if s.stage == "Context")
    assert "discarded" in step.chose


# ---------------------------------------------------------------------
# Pronoun relations
# ---------------------------------------------------------------------


def test_pronoun_relations_resolve_when_the_feature_is_enabled(org, monkeypatch):
    """"his team" is owned by cross_turn_resolver and gated on
    RELATION_INFERENCE_ENABLED, which ships default-off. Pinned here so
    the flag's effect is visible, and so nothing in this module grows a
    SECOND pronoun resolver — that ownership already has an owner.
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "relation_inference_enabled", True, raising=False)
    r = _run(org, "p1", ["Tell me about Yasir Ali", "How is his team doing?"])

    plan = getattr(r, "plan", None)
    assert plan is not None
    assert plan.level == "team"
    assert plan.entity_value == "Blue Area"


def test_an_explicitly_named_subject_beats_a_remembered_one(org):
    """Design principle 5: the current turn always wins."""
    r = _run(org, "p2", ["What is Yasir Ali's cleared?", "Downtown revenue"])
    assert ("team", "Downtown") in _scope(r)
