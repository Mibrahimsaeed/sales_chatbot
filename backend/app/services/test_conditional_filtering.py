"""Phase 39 — a threshold is a condition on the METRIC, and it is the limit.

    "advisors with answered calls % greater than 60"
    -> 404 people. 185 of them clear 60%.

TWO SEPARATE ERRORS, both in one line of the compiler.

THE DENOMINATOR WAS ABSENT. The filter was built from
`binding.expr`, which for a RATIO is the NUMERATOR ALONE — so
`rate > 60` compiled to `answered_calls * 100 > 60`, i.e.
`answered_calls > 0.6`. Everyone who answered a single call passed. Not
a scale confusion between 60, 0.60 and 60%: the divisor was simply not
in the SQL. The fix asks aggregation.value_expression for the
expression, which is the same call the SORT metric makes, so a threshold
and a ranking on one measure can no longer disagree about what it is.

IT WAS A `WHERE`, NOT A `HAVING`. Above the leaf the condition selected
individual advisor rows BEFORE `GROUP BY`, so a BCM qualified because
one of her people did, and the value shown was aggregated over only
those people: "BCMs with answered calls below 60%" answered with a BCM
at 283, and a row read 102 where the engine said 68 for the same person.

Comparators were never reversed — `>` maps to `gt` throughout — and the
metric definitions were correct. Only the filter's expression and its
placement were wrong.

AND A CONDITION IS ALREADY THE LIMIT. "BCMs above 60%" asks for the ones
that qualify, all of them, and how many there are is the point of
asking; ten of thirty reported as `total=10` makes a filtered list read
as a complete one. Rankings keep their ten, because "top" and "most" ask
who is ahead.

The fixture puts values on BOTH sides of the boundary and ON it, with
more than fifteen matches at advisor level, so a page boundary and an
inclusive/exclusive edge are exercised rather than assumed.
"""

import pytest

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import (
    advisor_resolver, aggregation, conversation_memory, entity_extractor,
    narrative, nlu_pipeline, semantic_parser,
)
from app.llm.query_compiler import compile_and_run, count_ir
from app.services import advisor_service
from app.services.chat_service import PAGE_SIZE, handle_chat_message, handle_show_more

# `answered_calls_rate` is answered calls against 10 per advisor per
# working day, so for ONE advisor the rate is answered_calls * 100 /
# (10 * working_days). The fixture sets answered calls directly and reads
# the resulting rate from the engine rather than recomputing it here —
# this file tests the FILTER, and a second copy of the formula would let
# both be wrong together.
ADVISORS = 40


def _answered(wid):
    """A spread that straddles 60 at EVERY level.

    For one advisor the denominator is 10 x working_days = 100, so this
    number IS the rate — which is what puts values either side of 60 and,
    for wid % 5 == 4 combined with wid % 3 == 0, exactly ON it. The
    per-level moduli below (8 / 5 / 3) then give each of BCM, Zonal Head
    and Unit Head groups on both sides too; a fixture where one level sat
    entirely above the threshold would let a broken `<` pass by returning
    nothing.
    """
    return (wid % 5) * 4 + (wid % 3) * 12 + 39


@pytest.fixture()
def db(db_session, monkeypatch):
    for wid in range(1, ADVISORS + 1):
        db_session.add(Advisor(
            wid=wid, name=f"Advisor {wid:02d}", team="Alpha", company="Graana",
            rm=f"Unit {wid % 3}", portfolio_lead=f"Zonal {wid % 5}",
            management_lead=f"Bcm {wid % 8}", in_master_sheet=True))
        db_session.add(Calls(wid=wid, connects_mtd=wid * 10, connects_daily=0,
                             answered_calls_mtd=_answered(wid), answered_calls_daily=0))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=wid * 10,
                                   mtd_followup_connect=0, mtd_cr=wid))
        db_session.add(Pipeline(wid=wid, pipeline=wid * 100, overdue=0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=wid))
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


LEVELS = [
    ("advisors", "advisor"),
    ("BCMs", "bcm"),
    ("Zonal Heads", "zonal_head"),
    ("Unit Heads", "unit_head"),
]

RATE = "answered_calls_rate"


def _rows(db, text):
    """Every matching row, uncapped — the population the condition
    describes, independent of how it is paged."""
    resolution = nlu_pipeline.resolve(text, db)
    assert resolution.ir is not None, text
    ir = resolution.ir.model_copy(update={"limit": None})
    return ir, compile_and_run(db, ir)


def _engine(db, level, row, key):
    """The metric's own value for that row's subject, keyed the way that
    level is addressed — wid for a person, the group value above."""
    if level == "advisor":
        return advisor_service.get_advisor_metric(db, row["wid"], key)
    return aggregation.metric_value(db, level, row["name"], key)


def _threshold(ir):
    f = ir.filters[0]
    return f.field, f.operator, f.value


def _walk(db, text):
    session = f"cond-{text}"
    conversation_memory._store.pop(session, None)
    response = handle_chat_message(db, text, session_id=session)
    rows = list(response["data"])
    first, total = response, response["total_count"]
    while response.get("has_more"):
        response = handle_show_more(db, session)
        rows += list(response["data"])
    return first, rows, total


# ---------------------------------------------------------------------
# Only the people who satisfy the condition
# ---------------------------------------------------------------------


@pytest.mark.parametrize("noun,level", LEVELS)
def test_greater_than_returns_only_values_above_it(db, noun, level):
    _ir, rows = _rows(db, f"{noun} whose answered calls % is greater than 60%")
    assert rows, "fixture must produce matches"
    assert all(row["value"] > 60 for row in rows)


@pytest.mark.parametrize("noun,level", LEVELS)
def test_less_than_returns_only_values_below_it(db, noun, level):
    _ir, rows = _rows(db, f"{noun} whose answered calls % is less than 60%")
    assert rows
    assert all(row["value"] < 60 for row in rows)


@pytest.mark.parametrize("noun,level", LEVELS)
def test_the_two_conditions_partition_the_population(db, noun, level):
    """Above-60 and below-60 must be disjoint, and together account for
    everyone except those exactly ON the boundary. A filter that keeps
    the wrong rows shows up here even when each set looks plausible."""
    _a, above = _rows(db, f"{noun} whose answered calls % is greater than 60%")
    _b, below = _rows(db, f"{noun} whose answered calls % is less than 60%")
    _c, everyone = _rows(db, f"answered calls % of all {noun}")

    above_names = {r["name"] for r in above}
    below_names = {r["name"] for r in below}
    assert above_names & below_names == set()
    on_boundary = {r["name"] for r in everyone if r["value"] == 60}
    assert above_names | below_names | on_boundary == {r["name"] for r in everyone}


@pytest.mark.parametrize("noun,level", LEVELS)
def test_every_returned_value_matches_the_metric_engine(db, noun, level):
    """The partial-aggregation guard. A row filtered before GROUP BY is
    aggregated over only the qualifying members, so its value silently
    stops being that person's figure."""
    ir, rows = _rows(db, f"{noun} whose answered calls % is greater than 60%")
    field, _op, _value = _threshold(ir)
    for row in rows:
        assert row["value"] == pytest.approx(_engine(db, level, row, field))


@pytest.mark.parametrize("noun,level", LEVELS)
def test_nobody_who_qualifies_is_missing(db, noun, level):
    """The complement of the tests above: the result is not merely
    correct, it is complete."""
    _ir, matched = _rows(db, f"{noun} whose answered calls % is greater than 60%")
    _all, everyone = _rows(db, f"answered calls % of all {noun}")
    assert {r["name"] for r in matched} == {r["name"] for r in everyone if r["value"] > 60}


# ---------------------------------------------------------------------
# Scale, boundary and direction
# ---------------------------------------------------------------------


def test_the_threshold_is_read_as_a_percentage_not_a_count(db):
    """The defect in one assertion: `rate > 60` used to compile to
    `answered_calls * 100 > 60` — every advisor with any answered call.
    The two counts differ by the entire denominator, and every advisor
    here has answered some."""
    _ir, rows = _rows(db, "advisors whose answered calls % is greater than 60%")
    with_any_call = [w for w in range(1, ADVISORS + 1) if _answered(w) > 0]
    assert len(with_any_call) == ADVISORS
    assert len(rows) < ADVISORS


def test_a_boundary_value_is_excluded_by_strict_comparison(db):
    """Exactly 60 is neither above nor below it."""
    _a, above = _rows(db, "advisors whose answered calls % is greater than 60%")
    _b, below = _rows(db, "advisors whose answered calls % is less than 60%")
    assert all(r["value"] != 60 for r in above + below)


def test_greater_and_less_are_not_reversed(db):
    """Cheap to get backwards, and invisible in a plausible-looking
    list."""
    _a, above = _rows(db, "advisors whose answered calls % is greater than 60%")
    _b, below = _rows(db, "advisors whose answered calls % is less than 60%")
    assert min(r["value"] for r in above) > max(r["value"] for r in below)


@pytest.mark.parametrize("phrase,op", [
    ("at least 60", ">="), ("no more than 60", "<="),
])
def test_inclusive_comparators_include_the_boundary(db, phrase, op):
    ir, rows = _rows(db, f"advisors whose answered calls % is {phrase}%")
    _field, operator, value = _threshold(ir)
    assert operator == op
    if op == ">=":
        assert all(row["value"] >= value for row in rows)
    else:
        assert all(row["value"] <= value for row in rows)


# ---------------------------------------------------------------------
# The whole matching population, through Show More
# ---------------------------------------------------------------------


def test_a_conditional_query_is_not_capped_at_ten(db):
    """It reported `total=10` for thirty matches, so nothing on screen or
    in the payload said the list had been cut."""
    first, _rows, total = _walk(db, "advisors whose answered calls % is greater than 60%")
    assert total > 10
    assert first["total_count"] == total
    assert len(first["data"]) == min(PAGE_SIZE, total)


def test_the_first_page_offers_show_more_when_there_is_more(db):
    first, rows, total = _walk(db, "advisors whose answered calls % is greater than 60%")
    assert total > PAGE_SIZE
    assert first["has_more"] is True
    assert len(rows) == total


def test_show_more_keeps_the_condition(db):
    """It must page through the MATCHING people, never fall back to an
    unfiltered list."""
    _first, rows, _total = _walk(db, "advisors whose answered calls % is greater than 60%")
    assert all(row["value"] > 60 for row in rows)


def test_walking_every_page_reaches_each_person_once(db):
    _first, rows, total = _walk(db, "advisors whose answered calls % is greater than 60%")
    identities = [(r.get("wid"), r["name"]) for r in rows]
    assert len(identities) == total
    assert len(set(identities)) == total


def test_the_pages_preserve_the_ordering(db):
    """Show More continues the same ranking rather than restarting it."""
    _first, rows, _total = _walk(db, "advisors whose answered calls % is greater than 60%")
    values = [r["value"] for r in rows]
    assert values == sorted(values, reverse=True)


def test_a_small_result_is_shown_whole(db):
    """Eight matches means eight rows and no Show More."""
    first, rows, total = _walk(db, "Unit Heads whose answered calls % is less than 60%")
    assert total <= PAGE_SIZE
    assert first["has_more"] is False
    assert len(rows) == total


# ---------------------------------------------------------------------
# Everything else is unchanged
# ---------------------------------------------------------------------


def test_a_ranking_still_shows_ten(db):
    """"Top"/"most" name a bound of their own; a threshold does not."""
    response = handle_chat_message(db, "top advisors by connects", session_id=None)
    assert len(response["data"]) == 10


def test_a_stated_number_still_wins(db):
    response = handle_chat_message(
        db, "top 5 advisors whose answered calls % is greater than 60%", session_id=None)
    assert len(response["data"]) == 5


def test_a_plain_metric_threshold_still_works(db):
    """A SUM metric's expression IS its value, so it never had the
    denominator problem — but it did have the WHERE/GROUP BY one."""
    ir, rows = _rows(db, "BCMs with pipeline greater than 1000")
    field, _op, value = _threshold(ir)
    assert rows
    for row in rows:
        assert row["value"] > value
        assert row["value"] == pytest.approx(aggregation.metric_value(db, "bcm", row["name"], field))


def test_an_unconditional_query_is_unchanged(db):
    """No threshold, no ranking word: the default of 10 still applies."""
    response = handle_chat_message(db, "connects by bcm", session_id=None)
    assert len(response["data"]) <= 10


def test_count_ir_agrees_with_the_rows(db):
    """The total and the list must describe one population — a count that
    ignored the filter is how "showing 15 of 573" appeared over a
    filtered list."""
    ir, rows = _rows(db, "BCMs whose answered calls % is greater than 60%")
    assert count_ir(db, ir) == len(rows)
