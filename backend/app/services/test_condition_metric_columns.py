"""A conditional list shows the value of every metric it filtered on.

"advisors with target achievement below 50% and answered calls % below
50%" returned the right people and printed one of the two numbers: the
columns came from the ontology's bundle declaration
(metric_ontology.BUNDLES), which knows the measures that COMPLETE a
primary but nothing about what the user asked. `achievement_pct` is in no
bundle, so the answer to a two-condition question carried one column and
could not be checked against the question.

These lock the fix and, as importantly, the things it must not disturb:
the filtering is untouched (tests 7/10), an unconditional ranking still
renders exactly as before (test 11), and a single-person answer is not a
table (test 12).
"""

from app.database.models import Advisor, Calls, Performance, PerformancePeriod
from app.llm import aggregation
from app.llm.query_compiler import compile_and_run
from app.llm.query_ir import Filter, MetricRef, QueryIR, Sort, Subject
from app.llm.response_formatter import format_ir_reply
from app.llm.response_planner import plan_response
from app.services import chat_service


# SOURCE OF TRUTH FOR CONNECTS is calls.connects_mtd, not SalesFunnel —
# Phase 17 named the Answered Calls tab authoritative for the connects
# family. Seeding the CCMC columns instead is how a fixture ends up
# asserting against a metric that reads 0.
# `answered` IS A COUNT; `answered_rate` IS A PERCENTAGE.
#
# Both end up in the same column, because answered_calls_rate is derived
# from the count — but the DENOMINATOR is the number of working days
# elapsed this month, which moves every day. A test that filters on
# `answered_calls_rate < 50` and seeds a raw count is therefore pinned to
# whatever the calendar said on the day it was written: the counts here
# (38, 44, 10, 20) sat comfortably under 50% through most of a month and
# read as 380%, 440%, 100% and 200% on its first working day, at which
# point four tests selected nobody.
#
# Naming the two units separately is what stops the next fixture making
# the same mistake silently.
def _seed(db, wid, name, *, cleared=None, target=100, connects=None,
          answered=None, answered_rate=None, bcm=None, adv_name=None):
    db.add(Advisor(wid=wid, name=adv_name or name, team="Alpha", company="IMARAT",
                   management_lead=bcm, in_master_sheet=True))
    if cleared is not None:
        db.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                           cleared=cleared, target=target))
    if answered_rate is not None:
        from app.llm import working_days

        # Invert the metric's own formula, so the RATE is the same on
        # every date. Float column, so the conversion is exact.
        answered = answered_rate * working_days.month_to_date() / 10.0
    if connects is not None or answered is not None:
        db.add(Calls(wid=wid, connects_mtd=connects or 0,
                     answered_calls_mtd=answered or 0))


def _ir(**kw):
    kw.setdefault("intent", "filtered_list")
    kw.setdefault("subject_level", "advisor")
    kw.setdefault("limit", 10)
    return QueryIR(**kw)


def _columns(db, ir):
    """The attached cells for one query, as (keys, rows)."""
    rows = compile_and_run(db, ir)
    keys = chat_service._attach_bundle_columns(db, ir, rows)
    return keys, rows


def _render(db, ir, rows, total=None):
    plan = plan_response(ir, rows)
    return format_ir_reply(ir, rows, total_count=total if total is not None else len(rows),
                           plan=plan)


# ---------------------------------------------------------------- 1
def test_one_percentage_condition_shows_that_percentage_column(db_session):
    _seed(db_session, 1, "Low", cleared=42.3)
    _seed(db_session, 2, "High", cleared=90.0)
    db_session.commit()

    ir = _ir(metric=MetricRef(key="achievement_pct"),
             filters=[Filter(field="achievement_pct", operator="<", value=50)],
             sort=Sort(metric="achievement_pct", direction="desc"))
    keys, rows = _columns(db_session, ir)

    assert keys == ["achievement_pct"]
    assert [r["name"] for r in rows] == ["Low"]
    # The unit survives: this printed a bare "42" through f"{value:,.0f}".
    assert rows[0]["columns"]["achievement_pct"]["display"] == "42.3%"
    assert rows[0]["columns"]["achievement_pct"]["label"] == "Target Achievement %"


# ---------------------------------------------------------------- 2
def test_two_percentage_conditions_show_both_columns(db_session):
    _seed(db_session, 1, "Person A", cleared=42.3, answered_rate=14.6)
    _seed(db_session, 2, "Person B", cleared=35.7, answered_rate=16.9)
    _seed(db_session, 3, "Passes Neither", cleared=95.0, answered_rate=346.0)
    db_session.commit()

    ir = _ir(metric=MetricRef(key="achievement_pct"),
             filters=[Filter(field="achievement_pct", operator="<", value=50),
                      Filter(field="answered_calls_rate", operator="<", value=50)],
             sort=Sort(metric="achievement_pct", direction="desc"))
    keys, rows = _columns(db_session, ir)

    assert keys == ["achievement_pct", "answered_calls_rate"]
    assert {r["name"] for r in rows} == {"Person A", "Person B"}
    for row in rows:
        assert set(row["columns"]) == {"achievement_pct", "answered_calls_rate"}
        assert row["columns"]["achievement_pct"]["display"].endswith("%")
        assert row["columns"]["answered_calls_rate"]["display"].endswith("%")


# ---------------------------------------------------------------- 3
def test_two_count_conditions_show_both_columns(db_session):
    _seed(db_session, 1, "Busy", connects=1200, answered=600)
    _seed(db_session, 2, "Quiet", connects=100, answered=50)
    db_session.commit()

    ir = _ir(metric=MetricRef(key="total_connects"),
             filters=[Filter(field="total_connects", operator=">", value=1000),
                      Filter(field="answered_calls", operator=">", value=500)],
             sort=Sort(metric="total_connects", direction="desc"))
    keys, rows = _columns(db_session, ir)

    assert [r["name"] for r in rows] == ["Busy"]
    # Both conditions are present. The connects BUNDLE also contributes
    # answered_calls_rate — existing behaviour this fix unions with
    # rather than replaces.
    assert keys[:2] == ["total_connects", "answered_calls"]
    assert set(keys) >= {"total_connects", "answered_calls"}
    # A count stays a count: no percent sign, no decimal.
    assert rows[0]["columns"]["total_connects"]["display"] == "1,200"
    assert rows[0]["columns"]["answered_calls"]["display"] == "600"


# ---------------------------------------------------------------- 4
def test_mixed_percentage_and_count_conditions(db_session):
    _seed(db_session, 1, "Match", cleared=40.0, connects=1200)
    _seed(db_session, 2, "Miss", cleared=40.0, connects=10)
    db_session.commit()

    ir = _ir(metric=MetricRef(key="achievement_pct"),
             filters=[Filter(field="achievement_pct", operator="<", value=50),
                      Filter(field="total_connects", operator=">", value=1000)],
             sort=Sort(metric="achievement_pct", direction="desc"))
    keys, rows = _columns(db_session, ir)

    assert [r["name"] for r in rows] == ["Match"]
    assert keys == ["achievement_pct", "total_connects"]
    assert rows[0]["columns"]["achievement_pct"]["display"] == "40%"
    assert rows[0]["columns"]["total_connects"]["display"] == "1,200"


# ---------------------------------------------------------------- 5
def test_and_conditions_preserve_every_referenced_metric(db_session):
    _seed(db_session, 1, "All Three", cleared=40.0, connects=1200, answered=600)
    db_session.commit()

    ir = _ir(metric=MetricRef(key="achievement_pct"),
             filters=[Filter(field="achievement_pct", operator="<", value=50),
                      Filter(field="total_connects", operator=">", value=1000),
                      Filter(field="answered_calls", operator=">", value=500)],
             sort=Sort(metric="achievement_pct", direction="desc"))
    keys, rows = _columns(db_session, ir)

    for key in ("achievement_pct", "total_connects", "answered_calls"):
        assert key in keys, f"{key} was filtered on but has no column"
        assert rows[0]["columns"][key]["value"] is not None


def test_a_band_on_one_metric_yields_one_column(db_session):
    """Two filters, one measure — "advisors who almost achieved their
    target" is >=80 AND <100 on achievement_pct (ir_examples.py). The
    ordered union must dedupe it to a single column."""
    _seed(db_session, 1, "Close", cleared=85.0)
    db_session.commit()

    ir = _ir(metric=MetricRef(key="achievement_pct"),
             filters=[Filter(field="achievement_pct", operator=">=", value=80),
                      Filter(field="achievement_pct", operator="<", value=100)],
             sort=Sort(metric="achievement_pct", direction="desc"))
    keys, rows = _columns(db_session, ir)

    assert keys == ["achievement_pct"]
    assert [r["name"] for r in rows] == ["Close"]


def test_entity_filters_do_not_become_columns(db_session):
    """`team`/`company` are not measures. They are already named in the
    reply's header, and a column of the same string on every row is not
    an answer to anything."""
    _seed(db_session, 1, "In Alpha", cleared=40.0)
    db_session.commit()

    ir = _ir(metric=MetricRef(key="achievement_pct"),
             filters=[Filter(field="achievement_pct", operator="<", value=50),
                      Filter(field="team", operator="=", value="Alpha")],
             sort=Sort(metric="achievement_pct", direction="desc"))
    keys, _ = _columns(db_session, ir)

    assert keys == ["achievement_pct"]


# ---------------------------------------------------------------- 6
def test_displayed_value_matches_the_metric_engine(db_session):
    """The column and the condition must be the same number. They come
    from one owner — aggregation.value_expression — and this asserts the
    equality rather than trusting it."""
    _seed(db_session, 1, "A1", cleared=40.0, connects=1200, answered=600, bcm="BCM One")
    _seed(db_session, 2, "A2", cleared=30.0, connects=300, answered=200, bcm="BCM One")
    db_session.commit()

    ir = _ir(subject_level="bcm", metric=MetricRef(key="total_connects"),
             filters=[Filter(field="total_connects", operator=">", value=1000),
                      Filter(field="answered_calls", operator=">", value=500)],
             sort=Sort(metric="total_connects", direction="desc"))
    keys, rows = _columns(db_session, ir)

    assert [r["name"] for r in rows] == ["BCM One"]
    for key in keys:
        # The SAME (level, name) scope the row was built from.
        expected = aggregation.metric_value(db_session, "bcm", "BCM One", key)
        assert rows[0]["columns"][key]["value"] == expected, key


def test_advisor_column_matches_the_wid_keyed_engine(db_session):
    _seed(db_session, 7, "Solo", cleared=42.3, answered_rate=14.6)
    db_session.commit()

    ir = _ir(metric=MetricRef(key="achievement_pct"),
             filters=[Filter(field="achievement_pct", operator="<", value=50),
                      Filter(field="answered_calls_rate", operator="<", value=50)],
             sort=Sort(metric="achievement_pct", direction="desc"))
    _, rows = _columns(db_session, ir)

    from app.services import advisor_service
    assert rows[0]["columns"]["answered_calls_rate"]["value"] == \
        advisor_service.get_advisor_metric(db_session, 7, "answered_calls_rate")


# ---------------------------------------------------------------- 7
def test_comparators_still_filter_exactly_as_before(db_session):
    for wid, name, pct in [(1, "Forty", 40.0), (2, "Fifty", 50.0), (3, "Sixty", 60.0)]:
        _seed(db_session, wid, name, cleared=pct)
    db_session.commit()

    def _matched(operator, value):
        ir = _ir(metric=MetricRef(key="achievement_pct"),
                 filters=[Filter(field="achievement_pct", operator=operator, value=value)],
                 sort=Sort(metric="achievement_pct", direction="asc"))
        return [r["name"] for r in compile_and_run(db_session, ir)]

    assert _matched("<", 50) == ["Forty"]
    assert _matched("<=", 50) == ["Forty", "Fifty"]
    assert _matched(">", 50) == ["Sixty"]
    assert _matched(">=", 50) == ["Fifty", "Sixty"]


# ---------------------------------------------------------------- 8, 9, 10
_MATCHING = chat_service.PAGE_SIZE + 5


def _paginating_ir(db):
    """More matching advisors than fit on one page, so the result is a
    real two-page set rather than a simulated one."""
    for wid in range(1, _MATCHING + 1):
        _seed(db, wid, f"Advisor {wid:02d}", cleared=float(wid), answered=1)
    db.commit()
    return _ir(metric=MetricRef(key="achievement_pct"),
               filters=[Filter(field="achievement_pct", operator="<", value=50),
                        Filter(field="answered_calls_rate", operator="<", value=50)],
               sort=Sort(metric="achievement_pct", direction="desc"), limit=None)


def test_more_than_a_page_of_matches_still_paginates(db_session):
    ir = _paginating_ir(db_session)
    total = chat_service.count_ir(db_session, ir)
    assert total == _MATCHING

    page1 = compile_and_run(db_session, chat_service._page_ir(ir, 0, total), offset=0)
    assert len(page1) == chat_service.PAGE_SIZE
    assert total > len(page1)


def test_show_more_preserves_exactly_the_same_columns(db_session):
    ir = _paginating_ir(db_session)
    total = chat_service.count_ir(db_session, ir)

    # Page 1, then page 2 through the SAME calls _show_more makes.
    page1 = compile_and_run(db_session, chat_service._page_ir(ir, 0, total), offset=0)
    keys1 = chat_service._attach_bundle_columns(db_session, ir, page1)
    offset = len(page1)
    page2 = compile_and_run(db_session, chat_service._page_ir(ir, offset, total), offset=offset)
    keys2 = chat_service._attach_bundle_columns(db_session, ir, page2)

    assert keys1 == keys2 == ["achievement_pct", "answered_calls_rate"]
    assert page2, "page 2 should carry the remaining rows"
    for row in page2:
        assert set(row["columns"]) == set(keys1)


def test_pages_cover_every_row_once(db_session):
    ir = _paginating_ir(db_session)
    total = chat_service.count_ir(db_session, ir)

    page1 = compile_and_run(db_session, chat_service._page_ir(ir, 0, total), offset=0)
    offset = len(page1)
    page2 = compile_and_run(db_session, chat_service._page_ir(ir, offset, total), offset=offset)

    names = [r["name"] for r in page1 + page2]
    assert len(names) == _MATCHING
    assert len(set(names)) == _MATCHING, "a row was repeated across pages"


# ---------------------------------------------------------------- 11
def test_unconditional_leaderboard_is_unchanged(db_session):
    """No metric filter means no condition columns — an unbundled measure
    still attaches nothing at all, exactly as before."""
    _seed(db_session, 1, "One", cleared=90.0)
    _seed(db_session, 2, "Two", cleared=80.0)
    db_session.commit()

    ir = QueryIR(intent="leaderboard", subject_level="advisor",
                 metric=MetricRef(key="achievement_pct"),
                 sort=Sort(metric="achievement_pct", direction="desc"), limit=10)
    keys, rows = _columns(db_session, ir)

    assert keys == []
    assert all(r.get("columns") is None for r in rows)
    assert "🏆 Top 2 by Target Achievement %" in _render(db_session, ir, rows)


def test_unconditional_bundled_leaderboard_keeps_its_bundle(db_session):
    """The connects bundle still produces its three columns in its
    established order — the union must not reorder or extend it."""
    _seed(db_session, 1, "One", connects=1200, answered=600)
    db_session.commit()

    ir = QueryIR(intent="leaderboard", subject_level="advisor",
                 metric=MetricRef(key="total_connects"),
                 sort=Sort(metric="total_connects", direction="desc"), limit=10)
    keys, _ = _columns(db_session, ir)

    assert keys == ["total_connects", "answered_calls", "answered_calls_rate"]


# ---------------------------------------------------------------- 12
def test_single_person_query_is_unchanged(db_session):
    """One named subject and no condition is a sentence, not a table.

    Built the way the pipeline builds it: a leaderboard that resolves to
    exactly one row, which response_planner turns into single_value
    (Phase 3). `lookup` is a registry-unsupported intent and would be
    rejected before rendering, so it would not exercise this at all.
    """
    _seed(db_session, 1, "Solo Person", cleared=42.3)
    _seed(db_session, 2, "Someone Else", cleared=90.0)
    db_session.commit()

    ir = QueryIR(intent="leaderboard", subject_level="advisor",
                 subjects=[Subject(type="advisor", value="Solo Person", resolved_wid=1)],
                 metric=MetricRef(key="achievement_pct"),
                 sort=Sort(metric="achievement_pct", direction="desc"), limit=10)
    keys, rows = _columns(db_session, ir)

    assert keys == []
    assert [r["name"] for r in rows] == ["Solo Person"]
    assert rows[0].get("columns") is None
    plan = plan_response(ir, rows)
    assert plan.shape == "single_value"


# ---------------------------------------------------------------- 13
def test_duplicate_advisor_names_are_valued_by_wid(db_session):
    """Names are not identifiers (238 duplicate-name groups in
    production). Two people called "Same Name" with different figures
    must each get their own."""
    _seed(db_session, 1, "x", adv_name="Same Name", cleared=40.0, answered_rate=3.8)
    _seed(db_session, 2, "x", adv_name="Same Name", cleared=30.0, answered_rate=7.7)
    db_session.commit()

    ir = _ir(metric=MetricRef(key="achievement_pct"),
             filters=[Filter(field="achievement_pct", operator="<", value=50),
                      Filter(field="answered_calls_rate", operator="<", value=50)],
             sort=Sort(metric="achievement_pct", direction="desc"))
    keys, rows = _columns(db_session, ir)

    assert keys == ["achievement_pct", "answered_calls_rate"]
    assert len(rows) == 2
    from app.services import advisor_service
    by_wid = {r["wid"]: r for r in rows}
    assert set(by_wid) == {1, 2}
    for wid, row in by_wid.items():
        assert row["columns"]["answered_calls_rate"]["value"] == \
            advisor_service.get_advisor_metric(db_session, wid, "answered_calls_rate")
    # And the two rows genuinely differ, so a name-keyed read could not
    # have passed this by accident.
    assert by_wid[1]["columns"]["answered_calls_rate"]["value"] != \
        by_wid[2]["columns"]["answered_calls_rate"]["value"]


# ------------------------------------------------- Change 2: formatting
def test_filtered_list_renders_the_columns_as_a_table(db_session):
    _seed(db_session, 1, "Person A", cleared=42.3, answered_rate=14.6)
    _seed(db_session, 2, "Person B", cleared=35.7, answered_rate=16.9)
    db_session.commit()

    ir = _ir(metric=MetricRef(key="achievement_pct"),
             filters=[Filter(field="achievement_pct", operator="<", value=50),
                      Filter(field="answered_calls_rate", operator="<", value=50)],
             sort=Sort(metric="achievement_pct", direction="desc"))
    _, rows = _columns(db_session, ir)
    reply = _render(db_session, ir, rows)

    assert "Target Achievement %" in reply
    assert "Answered Calls % of Target" in reply
    assert "42.3%" in reply and "35.7%" in reply


def test_filtered_list_without_columns_still_formats_its_unit(db_session):
    """The no-columns path kept f"{value:,.0f}", which printed a
    percentage as a bare integer. It goes through the ontology now."""
    _seed(db_session, 1, "In Alpha", cleared=42.3)
    db_session.commit()

    ir = _ir(metric=MetricRef(key="achievement_pct"),
             filters=[Filter(field="team", operator="=", value="Alpha")],
             sort=Sort(metric="achievement_pct", direction="desc"))
    keys, rows = _columns(db_session, ir)

    assert keys == []
    assert "42.3%" in _render(db_session, ir, rows)


def test_filtered_list_with_no_rows_is_unchanged(db_session):
    ir = _ir(metric=MetricRef(key="achievement_pct"),
             filters=[Filter(field="achievement_pct", operator="<", value=50)],
             sort=Sort(metric="achievement_pct", direction="desc"))
    assert _render(db_session, ir, []) == "No results matched those conditions."
