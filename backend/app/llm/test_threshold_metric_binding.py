"""A comparator binds to the measure it was written beside.

`_extract_thresholds` returned {operator, value} pairs with no measure,
so `_threshold_filters` bound every one of them to the single key the
plan had resolved. "advisors with target achievement below 50% and
answered calls % below 20%" therefore compiled to

    achievement_pct < 50 AND achievement_pct < 20

— the second condition applied to the first condition's column. For that
pair the AND reduces to `< 20` and the query returns nobody; the display
layer, asked to show the metrics the conditions named, correctly showed
the one metric the IR actually held.

Only reachable when the LLM is unavailable: the semantic parser emits
distinct filters, and this is the rule-based degrade path underneath it.
That path is what serves every query when the provider is down, so it has
to answer the same question.

The pairing rule is NEAREST MENTION BY GAP, which is why the
threshold-before-metric and threshold-after-metric cases below are both
here — "less than 50% target achievement" puts the comparator first and
"answered calls % less than 20%" puts it second, in one sentence.
"""

import pytest

from app.database.models import Advisor, Calls, Performance, PerformancePeriod
from app.llm import entity_extractor
from app.llm.entity_extractor import extract_entities
from app.llm.preprocessing import normalize
from app.llm.query_compiler import compile_and_run
from app.llm.query_ir import plan_to_ir
from app.llm.query_planner import build_query_plan


@pytest.fixture()
def org(db_session):
    """Four advisors spanning both conditions independently, so binding
    the second threshold to the wrong measure selects a different set
    rather than the same one by luck.

    answered_calls_rate is answered calls against a target of 10 per
    advisor per working day, so these counts land A and B under 20% and
    "Low Ach High Calls" far above it.
    """
    people = [
        (1, "Person A", 42.3, 9),             # both conditions
        (2, "Person B", 35.7, 11),            # both conditions
        (3, "Low Ach High Calls", 40.0, 300),  # achievement only
        (4, "High Ach Low Calls", 92.0, 5),    # calls only
    ]
    for wid, name, pct, answered in people:
        db_session.add(Advisor(wid=wid, name=name, team="Alpha", company="Graana",
                               in_master_sheet=True))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=pct, pct=pct))
        db_session.add(Calls(wid=wid, connects_mtd=1000 + wid, answered_calls_mtd=answered))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    return db_session


def _ir_for(text, db):
    cleaned = normalize(text)
    entities = extract_entities(cleaned, db)
    return plan_to_ir(build_query_plan(cleaned, entities), entities)


def _conditions(ir):
    return [(f.field, f.operator, f.value) for f in ir.filters]


def _thresholds(text):
    return entity_extractor._extract_thresholds(normalize(text))


# ------------------------------------------------------- multi-metric
def test_two_metrics_two_conditions_bind_to_their_own_measures(org):
    """The reported failure, end to end."""
    ir = _ir_for("advisors with less than 50% target achievement "
                 "and answered calls % less than 20%", org)

    assert _conditions(ir) == [
        ("achievement_pct", "<", 50.0),
        ("answered_calls_rate", "<", 20.0),
    ]


def test_the_threshold_written_before_its_metric_binds_to_it(org):
    """"less than 50% target achievement" — comparator first."""
    ir = _ir_for("advisors with less than 50% target achievement", org)
    assert _conditions(ir) == [("achievement_pct", "<", 50.0)]


def test_the_threshold_written_after_its_metric_binds_to_it(org):
    """"answered calls % less than 20%" — comparator second. Paired with
    the test above, these are the two orders that appear together in one
    sentence in the failing query."""
    ir = _ir_for("advisors with answered calls % less than 20%", org)
    assert _conditions(ir) == [("answered_calls_rate", "<", 20.0)]


def test_both_orders_in_one_sentence_resolve_independently(org):
    """Comparator-before-metric and comparator-after-metric, together.
    A rule that read only forwards, or only backwards, gets one of these
    two wrong."""
    assert _thresholds("advisors with less than 50% target achievement "
                       "and answered calls % less than 20%") == [
        {"operator": "<", "value": 50.0, "metric": "achievement_pct"},
        {"operator": "<", "value": 20.0, "metric": "answered_calls_rate"},
    ]


def test_two_count_conditions_bind_separately(org):
    ir = _ir_for("advisors with connects > 1000 and answered calls > 500", org)
    assert _conditions(ir) == [
        ("total_connects", ">", 1000.0),
        ("answered_calls", ">", 500.0),
    ]


def test_mixed_percentage_and_count_bind_separately(org):
    ir = _ir_for("advisors with target achievement < 50% and connects > 1000", org)
    assert _conditions(ir) == [
        ("achievement_pct", "<", 50.0),
        ("total_connects", ">", 1000.0),
    ]


# ------------------------------------------------------ single metric
def test_a_single_metric_query_is_untouched(org):
    """The overwhelming majority. No measure to disambiguate, so the
    extractor emits exactly the dict it always did — no `metric` key —
    and the threshold binds to plan.metric as before."""
    assert _thresholds("advisors with achievement above 80 percent") == [
        {"operator": ">", "value": 80.0},
    ]
    ir = _ir_for("advisors with achievement above 80 percent", org)
    assert _conditions(ir) == [("achievement_pct", ">", 80.0)]


def test_a_band_keeps_both_bounds_on_one_metric(org):
    """"between 80 and 100" is two bounds on ONE measure. Splitting them
    across measures would be the same defect in the other direction."""
    assert _thresholds("advisors with achievement between 80 and 100") == [
        {"operator": ">=", "value": 80.0},
        {"operator": "<=", "value": 100.0},
    ]
    ir = _ir_for("advisors with achievement between 80 and 100", org)
    assert _conditions(ir) == [
        ("achievement_pct", ">=", 80.0),
        ("achievement_pct", "<=", 100.0),
    ]


def test_a_band_stays_on_one_metric_even_beside_another_measure(org):
    """Two measures named, but the range is one span: both of its bounds
    must still land on the measure it belongs to."""
    thresholds = _thresholds("advisors with achievement between 80 and 100 "
                             "and answered calls % less than 20%")
    assert [(t["operator"], t["value"], t.get("metric")) for t in thresholds] == [
        (">=", 80.0, "achievement_pct"),
        ("<=", 100.0, "achievement_pct"),
        ("<", 20.0, "answered_calls_rate"),
    ]


# --------------------------------------------------- filtering itself
def test_the_two_conditions_select_the_right_people(org):
    """The point of the fix. Before it, this returned nobody: the AND of
    `achievement_pct < 50` and `achievement_pct < 20` is `< 20`, and no
    advisor here is under 20% achievement."""
    ir = _ir_for("advisors with less than 50% target achievement "
                 "and answered calls % less than 20%", org)
    rows = compile_and_run(org, ir) or []

    assert sorted(r["name"] for r in rows) == ["Person A", "Person B"]


def test_each_condition_alone_selects_its_own_superset(org):
    """Each half on its own, so the AND above is checkable against its
    parts rather than against an expectation written by hand."""
    ach = compile_and_run(org, _ir_for(
        "advisors with less than 50% target achievement", org)) or []
    calls = compile_and_run(org, _ir_for(
        "advisors with answered calls % less than 20%", org)) or []

    assert sorted(r["name"] for r in ach) == [
        "Low Ach High Calls", "Person A", "Person B"]
    assert sorted(r["name"] for r in calls) == [
        "High Ach Low Calls", "Person A", "Person B"]


# ------------------------------------------------------ the whole path
def test_both_measures_reach_the_rendered_table(org):
    """Extractor -> plan -> IR -> compiler -> columns -> reply, on the
    rule-based path, for the exact reported query."""
    from app.llm.response_formatter import format_ir_reply
    from app.llm.response_planner import plan_response
    from app.services import chat_service

    ir = _ir_for("advisors with less than 50% target achievement "
                 "and answered calls % less than 20%", org)
    rows = compile_and_run(org, ir) or []
    keys = chat_service._attach_bundle_columns(org, ir, rows)

    assert keys == ["achievement_pct", "answered_calls_rate"]

    reply = format_ir_reply(ir, rows, total_count=len(rows),
                            plan=plan_response(ir, rows))
    assert "Target Achievement %" in reply
    assert "Answered Calls % of Target" in reply
    assert "42.3%" in reply and "35.7%" in reply


def test_every_displayed_value_matches_the_engine(org):
    """Each cell against advisor_service for that WID — the owner the
    filter's own expression is built from."""
    from app.services import advisor_service, chat_service

    ir = _ir_for("advisors with less than 50% target achievement "
                 "and answered calls % less than 20%", org)
    rows = compile_and_run(org, ir) or []
    keys = chat_service._attach_bundle_columns(org, ir, rows)

    assert rows
    for row in rows:
        for key in keys:
            assert row["columns"][key]["value"] == \
                advisor_service.get_advisor_metric(org, row["wid"], key), key
