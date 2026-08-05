"""Phase 1 — routing pipeline regression tests.

Every test here pins a routing DECISION, not a rendered answer: which
metric a query reaches, whether a shortcut was allowed to claim it,
whether a refusal explains itself. The three defects these lock down
(P1/P2/P3 in app/llm/routing.py) were all invisible at the metric layer —
`attendance_rate` and `login_rate` computed correctly the entire time
they were unreachable — so a test that only asserted on numbers would
have stayed green through all of it.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import (
    Advisor, Attendance, Calls, Performance, PerformancePeriod, SalesFunnel,
)
from app.database.session import Base
from app.llm import entity_extractor, nlu_pipeline, routing


@pytest.fixture()
def org():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    for wid, name in ((1, "Ahmed Khan"), (2, "Sara Ali")):
        db.add(Advisor(wid=wid, name=name, team="Blue Area", company="Graana",
                       in_master_sheet=True, unit="A"))
        db.add(Attendance(wid=wid, biometric_mtd_ontime=15, biometric_mtd_late=5,
                          biometric_mtd_not_marked=0, login_mtd_ontime=18,
                          login_mtd_late=2, login_mtd_not_marked=0))
        db.add(SalesFunnel(wid=wid, mtd_cr=20, mtd_new_meeting=6,
                           mtd_followup_meeting=4, mtd_conversion=5,
                           mtd_meetings_planned=20, mtd_meetings_conducted=15))
        db.add(Calls(wid=wid, answered_calls_mtd=100))
        db.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                           target=100, cleared=50, pct=50))
    db.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db
    db.close()


def _metric_of(resolution):
    """The metric a query actually reached, whichever path served it."""
    if resolution.kind == "ir" and resolution.ir.metric:
        return resolution.ir.metric.key
    plan = getattr(resolution, "plan", None)
    return plan.metric if plan else None


# ---------------------------------------------------------------------
# P1 — shortcuts must never hijack a metric question
# ---------------------------------------------------------------------

# The two metrics the shortcut made unreachable. Both were fully bound
# and computing correctly throughout; only routing was broken.
PERSON_SCOPED_RATE_QUERIES = [
    ("What is Ahmed Khan's attendance percentage?", "attendance_rate"),
    ("What is Ahmed Khan's attendance rate?", "attendance_rate"),
    ("attendance rate for Ahmed Khan", "attendance_rate"),
    ("What is Ahmed Khan's login rate?", "login_rate"),
    ("What is Ahmed Khan's login percentage?", "login_rate"),
]


@pytest.mark.parametrize("query,expected", PERSON_SCOPED_RATE_QUERIES)
def test_person_scoped_rate_queries_reach_their_metric(org, query, expected):
    """P1: classify_intent() used to claim these before entity extraction
    ran, answering every one with a canned 'no attendance issues' sweep."""
    resolution = nlu_pipeline.resolve(query, org, session_id=None)

    assert resolution.kind != "shortcut", (
        f"{query!r} was hijacked by a shortcut handler"
    )
    assert _metric_of(resolution) == expected


@pytest.mark.parametrize("query", [
    "What is Ahmed Khan's attendance percentage?",
    "What is Ahmed Khan's login rate?",
])
def test_the_trace_records_why_the_shortcut_was_skipped(org, query):
    """A skipped shortcut must say what outranked it — the reason is what
    makes a future mis-route diagnosable."""
    nlu_pipeline.resolve(query, org, session_id=None)
    trace = routing.current_trace()

    assert trace.chose("Shortcut") == "skipped"
    why = next(s.why for s in trace.steps if s.stage == "Shortcut")
    assert "resolved" in why or "rate/percentage" in why


@pytest.mark.parametrize("query", [
    "show attendance issues",
    "attendance",
])
def test_generic_attendance_questions_still_use_the_shortcut(org, query):
    """The fix must not overshoot: a generic sweep has no person and no
    rate phrase, so the canned handler is still the right answer. Gating
    on 'did any metric resolve' would have broken these, because the bare
    word 'attendance' is itself an attendance_rate synonym."""
    resolution = nlu_pipeline.resolve(query, org, session_id=None)

    assert resolution.kind == "shortcut"
    assert resolution.shortcut_intent == "attendance_check"


@pytest.mark.parametrize("query", ["hello there", "thanks!", "what can you do?"])
def test_non_metric_shortcuts_are_untouched(org, query):
    resolution = nlu_pipeline.resolve(query, org, session_id=None)
    assert resolution.kind == "shortcut"


def test_group_scoped_rate_queries_still_reach_the_metric(org):
    """This one worked before the fix (the word 'team' tripped the old
    regex guard) and must keep working after it."""
    resolution = nlu_pipeline.resolve("Show me attendance rate by team", org,
                                      session_id=None)
    assert _metric_of(resolution) == "attendance_rate"


# ---------------------------------------------------------------------
# P2 — an unavailable measure always explains itself
# ---------------------------------------------------------------------

# Every phrase in the UNAVAILABLE registry, asked BOTH ways: with a name
# that resolves and one that does not. The defect was that only the
# unresolved form produced the explanation — naming the person in full
# downgraded the answer to a profile card.
# Three entries left this list when working_days.py made CR %,
# Connect % and Meeting % computable — a declared refusal is superseded
# the moment its missing ingredient arrives. Portfolio % remains, and is
# a DIFFERENT kind of refusal: it has no target to measure against at
# all, so no data can retire it.
#
# P2 is unchanged by that. What it guarantees is that an unavailable
# measure explains itself however the query is phrased, and the one
# remaining entry exercises exactly that.
UNAVAILABLE_QUERIES = [
    ("portfolio %", "no portfolio target"),
]

# The retired refusals, kept as a pinned list: each must now reach its
# RATE — not the count it used to be redirected to, which was the
# substitution the refusal existed to prevent.
RETIRED_REFUSALS = [
    ("connect %", "answered_calls_rate"),
    ("CR %", "cr_rate"),
    ("meetings %", "meeting_rate"),
]


@pytest.mark.parametrize("measure,metric", RETIRED_REFUSALS)
def test_a_retired_refusal_now_answers_with_its_rate(org, measure, metric):
    from app.llm.metric_ontology import resolve_metric

    assert resolve_metric(measure) == metric
    resolution = nlu_pipeline.resolve(f"What is Blue Area's {measure}?", org,
                                      session_id=None)
    assert resolution.kind != "clarify"


@pytest.mark.parametrize("measure,expected_reason", UNAVAILABLE_QUERIES)
@pytest.mark.parametrize("subject", ["Ahmed Khan", "Ahmed"])
def test_unavailable_metrics_always_explain(org, measure, expected_reason, subject):
    """P2: the explanation is a property of the METRIC, so it cannot
    depend on whether identity resolution happened to succeed."""
    resolution = nlu_pipeline.resolve(f"What is {subject}'s {measure}?", org,
                                      session_id=None)

    assert resolution.kind == "clarify"
    reply = resolution.clarify_message
    assert measure.lower() in reply.lower()
    assert expected_reason.lower() in reply.lower()


@pytest.mark.parametrize("measure", [m for m, _ in UNAVAILABLE_QUERIES])
def test_unavailable_metrics_offer_an_alternative(org, measure):
    """A refusal that names no alternative is a dead end. Each
    UNAVAILABLE entry declares an `instead`, and it must reach the user."""
    resolution = nlu_pipeline.resolve(f"What is Ahmed Khan's {measure}?", org,
                                      session_id=None)
    assert "instead" in resolution.clarify_message.lower()


@pytest.mark.parametrize("measure", [m for m, _ in UNAVAILABLE_QUERIES])
def test_unavailable_metrics_keep_the_clarify_metric_plan_action(org, measure):
    """Backward compatibility: consumers read plan.action to tell WHICH
    clarification this is. Moving the check earlier changed WHEN it fires,
    and must not change that contract."""
    resolution = nlu_pipeline.resolve(f"top advisors by {measure}", org,
                                      session_id=None)
    assert resolution.plan is not None
    assert resolution.plan.action == "clarify_metric"


def test_an_unavailable_measure_never_degrades_to_a_profile_lookup(org):
    """The exact P2 defect: the better-specified query got the worse
    answer, because a resolved person routed to action='lookup' with
    metric=None and the reason was dropped on the floor."""
    # Uses portfolio %, the one refusal working_days.py did not retire.
    # Connect % is a computable rate now, so it can no longer exercise
    # "an UNAVAILABLE measure must not become a profile card" — the
    # invariant this test is named for.
    resolution = nlu_pipeline.resolve("What is Ahmed Khan's portfolio %?", org,
                                      session_id=None)

    assert resolution.kind == "clarify"
    plan = resolution.plan
    assert plan is None or plan.action != "lookup"


# ---------------------------------------------------------------------
# P3 — a named-but-unresolved person is not "no person"
# ---------------------------------------------------------------------


def test_an_unresolvable_name_asks_who_was_meant(org):
    """P3: 'Ahmed' matches nobody (the advisor is 'Ahmed Khan'), and the
    metric's primary_level='team' then answered about Blue Area without
    saying the subject had changed."""
    resolution = nlu_pipeline.resolve("What is Ahmed's achievement %?", org,
                                      session_id=None)

    assert resolution.kind == "clarify"
    assert "Ahmed" in resolution.clarify_message
    assert "couldn't find" in resolution.clarify_message.lower()


def test_an_unresolvable_name_never_answers_about_the_team(org):
    resolution = nlu_pipeline.resolve("What is Ahmed's achievement %?", org,
                                      session_id=None)
    assert "Blue Area" not in (resolution.clarify_message or "")
    assert resolution.kind != "ir"


def test_a_resolvable_name_is_unaffected(org):
    resolution = nlu_pipeline.resolve("What is Ahmed Khan's achievement %?", org,
                                      session_id=None)
    assert resolution.kind != "clarify"
    assert _metric_of(resolution) == "achievement_pct"


@pytest.mark.parametrize("query,expected", [
    # No subject named at all — primary_level is the right default here,
    # which is the case P3 must NOT disturb.
    ("What is the achievement %?", "achievement_pct"),
    ("Top advisors by revenue", "mtd_cleared"),
    # A grounded group possessive is not an unresolved person.
    ("What is Blue Area's achievement %?", "achievement_pct"),
])
def test_queries_without_an_unresolved_person_route_normally(org, query, expected):
    resolution = nlu_pipeline.resolve(query, org, session_id=None)
    assert resolution.kind != "clarify"
    assert _metric_of(resolution) == expected


def test_a_possessive_relation_query_is_not_an_unresolved_person(org):
    """Managers are deliberately absent from the advisor gazetteer — they
    are grounded later, against the manager columns. An early P3 refusal
    broke every "X's team" traversal until reference_parser was consulted."""
    assert routing.unresolved_subject("show Adeel Dogar's team", {}) is None
    assert routing.unresolved_subject("Adeel Dogar's advisors", {}) is None


def test_a_name_with_no_measure_is_not_a_p3_case(org):
    """P3 fixes a METRIC question answered at the wrong level. A query
    naming no measure cannot hit that defect, so it must not be refused."""
    assert routing.unresolved_subject("show Zainab Malik's team", {}) is None


# ---------------------------------------------------------------------
# Routing order and determinism
# ---------------------------------------------------------------------


def test_entity_extraction_runs_before_the_shortcut_check(org):
    """The structural guarantee behind P1. If the shortcut check ever
    moves back above extraction, the entity dict it receives goes empty
    and this fails."""
    seen = {}
    original = nlu_pipeline.classify_shortcut

    def spy(text, entities):
        seen["entities"] = entities
        return original(text, entities)

    nlu_pipeline.classify_shortcut = spy
    try:
        nlu_pipeline.resolve("What is Ahmed Khan's attendance percentage?", org,
                             session_id=None)
    finally:
        nlu_pipeline.classify_shortcut = original

    assert seen["entities"], "the shortcut classifier was handed an empty dict"
    assert seen["entities"].get("advisor_wids") == [1]


@pytest.mark.parametrize("query", [
    "What is Ahmed Khan's attendance percentage?",
    "What is Ahmed Khan's connect %?",
    "What is Ahmed's achievement %?",
    "show attendance issues",
    "Top advisors by revenue",
])
def test_routing_is_deterministic(org, query):
    """The same query must route the same way every time — the routing
    predicates are pure, so this is a real invariant rather than a hope."""
    first = nlu_pipeline.resolve(query, org, session_id=None)
    first_steps = [(s.stage, s.chose) for s in routing.current_trace().steps]

    second = nlu_pipeline.resolve(query, org, session_id=None)
    second_steps = [(s.stage, s.chose) for s in routing.current_trace().steps]

    assert first.kind == second.kind
    assert _metric_of(first) == _metric_of(second)
    assert first_steps == second_steps


def test_the_trace_records_the_whole_route(org):
    nlu_pipeline.resolve("What is Ahmed Khan's attendance percentage?", org,
                         session_id=None)
    trace = routing.current_trace()
    stages = [s.stage for s in trace.steps]

    assert "Shortcut" in stages
    assert "Advisor" in stages
    assert "Planner" in stages
    rendered = trace.render()
    assert "Ahmed Khan" in rendered
    assert "↓" in rendered


def test_every_routing_step_states_a_reason(org):
    """A decision log that records only the outcome cannot tell you where
    a requirement was dropped."""
    nlu_pipeline.resolve("What is Ahmed Khan's attendance percentage?", org,
                         session_id=None)
    for step in routing.current_trace().steps:
        assert step.why, f"{step.stage} recorded no reason"


# ---------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------


def test_a_metric_key_outside_the_ontology_is_refused_not_emptied(org):
    """ir_validator's key check is scoped to three intents; an IR with any
    other intent could carry an invented key to the compiler, which found
    no binding and returned an empty result that reads as 'no data'."""
    from app.llm.query_ir import MetricRef, QueryIR, Sort

    # "breakdown" is outside ir_validator's leaderboard/comparison/
    # filtered_list key check, which is exactly the gap this covers.
    ir = QueryIR(intent="breakdown", subject_level="advisor",
                 metric=MetricRef(key="not_a_real_metric"),
                 sort=Sort(metric="not_a_real_metric"), limit=10)

    problem = routing.validate_route(ir)
    assert problem is not None
    assert "not_a_real_metric" in problem


def test_a_valid_ir_passes_validation(org):
    from app.llm.query_ir import MetricRef, QueryIR, Sort

    ir = QueryIR(intent="leaderboard", subject_level="advisor",
                 metric=MetricRef(key="mtd_cleared"),
                 sort=Sort(metric="mtd_cleared"), limit=10)
    assert routing.validate_route(ir) is None


# ---------------------------------------------------------------------
# Follow-up routing keeps working
# ---------------------------------------------------------------------


def test_a_follow_up_still_patches_the_prior_ir(org):
    """The routing gates run before the patcher; none of them may swallow
    a bare follow-up modifier."""
    nlu_pipeline.resolve("top advisors by revenue", org, session_id="s1")
    resolution = nlu_pipeline.resolve("only Graana", org, session_id="s1")

    assert resolution.kind == "ir"
    assert any(f.field == "company" and f.value == "Graana"
               for f in resolution.ir.filters)
