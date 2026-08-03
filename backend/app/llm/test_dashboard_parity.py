"""Phase 5.7 — the four dashboard business-logic parity defects.

The dashboard specification is the source of truth. Each defect below
produced a CONFIDENT WRONG ANSWER rather than a refusal, which is the
worst failure mode this system has.

  FIX 1  "bottom 5 advisors" returned the TOP advisors, unlimited.
         "bottom" was in no vocabulary at all — not the ranking words, not
         the direction words, not the limit pattern — so it contributed
         neither a direction nor an N.

  FIX 2  Status was banded on the RAW percentage. The dashboard rounds
         first, so anything in [84.5, 85) was yellow here and green there.

  FIX 3  Connect->CR divided by CONNECTS. The spec divides by ANSWERED
         CALLS — a different, smaller denominator, so every value was
         understated.

  FIX 4  Conversion / Pipeline / Portfolio had no thresholds at all, so
         those boards had no status. The spec makes them pass/fail on
         whether the total is above zero.
"""

import pytest

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, Pipeline, Portfolio, SalesFunnel,
)
from app.llm import aggregation, entity_extractor, intent_catalog as cat
from app.llm.entity_extractor import extract_entities
from app.llm.metric_ontology import PASS_FAIL_POSITIVE, round_half_up, status_for
from app.llm.preprocessing import normalize
from app.llm.query_compiler import compile_and_run
from app.llm.query_ir import MetricRef, QueryIR, Sort, plan_to_ir
from app.llm.query_planner import _sort_signal, build_query_plan


@pytest.fixture()
def org(db_session):
    """Six advisors with a clean spread, so a ranking's direction and
    length are both unambiguous.

      revenue  900 800 700 600 500 400   (higher is better)
      overdue    0   1   2   3   4   5   (LOWER is better)
    """
    for i in range(6):
        wid = i + 1
        db_session.add(Advisor(wid=wid, name=f"Adv {wid}", team="Blue Area",
                               company="Graana", in_master_sheet=True))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=1000, cleared=900 - i * 100, pct=90 - i * 10))
        db_session.add(Pipeline(wid=wid, pipeline=1000 * (6 - i), overdue=i))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=100, mtd_followup_connect=0,
                                   mtd_cr=30, mtd_new_meeting=10, mtd_followup_meeting=0,
                                   mtd_conversion=2))
        db_session.add(Calls(wid=wid, answered_calls_mtd=200, connects_mtd=100))
        db_session.add(Portfolio(wid=wid, value=1000 * (6 - i)))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    return db_session


def _run(text, db):
    cleaned = normalize(text)
    entities = extract_entities(cleaned, db)
    ir = plan_to_ir(build_query_plan(cleaned, entities), entities)
    rows = compile_and_run(db, ir) or []
    return ir, [(r["name"], r["value"]) for r in rows]


# =====================================================================
# FIX 1 — Bottom N
# =====================================================================

def test_bottom_5_returns_the_bottom_five(org):
    """The reported defect, end to end. This returned all six advisors
    led by the HIGHEST earner."""
    ir, rows = _run("bottom 5 advisors by revenue", org)

    assert ir.limit == 5
    assert len(rows) == 5
    assert ir.sort.direction == "asc"
    assert rows[0] == ("Adv 6", 400)          # the actual bottom
    assert rows[0][1] < rows[-1][1]


def test_top_5_still_returns_the_top_five(org):
    ir, rows = _run("top 5 advisors by revenue", org)

    assert ir.limit == 5
    assert len(rows) == 5
    assert ir.sort.direction == "desc"
    assert rows[0] == ("Adv 1", 900)


def test_top_and_bottom_are_opposite_ends(org):
    _ir, top = _run("top 3 advisors by revenue", org)
    _ir, bottom = _run("bottom 3 advisors by revenue", org)

    assert {n for n, _v in top} & {n for n, _v in bottom} == set()
    assert max(v for _n, v in bottom) < min(v for _n, v in top)


@pytest.mark.parametrize("text,expected_limit", [
    ("top 5 advisors by revenue", 5),
    ("bottom 5 advisors by revenue", 5),
    ("worst 3 advisors by revenue", 3),
    ("lowest 4 advisors by revenue", 4),
    ("highest 2 advisors by revenue", 2),
    ("best 6 advisors by revenue", 6),
])
def test_an_explicit_limit_is_always_respected(org, text, expected_limit):
    """The limit pattern was `top\\s+(\\d+)` — every other ranking word
    lost its N and silently fell back to 10."""
    ir, rows = _run(text, org)
    assert ir.limit == expected_limit, text
    assert len(rows) == expected_limit, text


def test_no_explicit_limit_still_defaults(org):
    ir, _rows = _run("bottom advisors by revenue", org)
    assert ir.limit == 10


def test_at_least_is_not_read_as_a_limit(org):
    """"least" is a ranking word AND sits inside "at least 80", so a naive
    limit pattern would read a limit of 80 out of a threshold."""
    entities = extract_entities(normalize("advisors with at least 80 percent achievement"), org)
    assert "limit" not in entities or entities.get("limit") is None


# ---- polarity: bottom is not merely "reverse the sort" ----

def test_bottom_of_a_lower_is_better_metric_is_the_highest_count(org):
    """SPEC: overdue ranks ascending because lower is better. So the
    BOTTOM performers are the ones with the MOST overdue. Reversing the
    sort instead would show the advisors with the FEWEST overdue items —
    the best performers, labelled worst."""
    ir, rows = _run("bottom 3 advisors by overdue", org)

    assert ir.sort.direction == "desc"
    assert rows[0][1] == 5              # the most overdue
    assert rows[0][1] > rows[-1][1]


def test_top_of_a_lower_is_better_metric_is_the_lowest_count(org):
    ir, rows = _run("top 3 advisors by overdue", org)

    assert ir.sort.direction == "asc"
    assert rows[0][1] == 0


@pytest.mark.parametrize("metric,lower_better,expected", [
    ("mtd_cleared", False, "asc"),      # bottom of higher-is-better = lowest
    ("overdue", True, "desc"),          # bottom of lower-is-better  = highest
])
def test_bottom_resolves_against_polarity(metric, lower_better, expected):
    from app.llm.metric_ontology import lower_is_better

    assert lower_is_better(metric) is lower_better
    signal = _sort_signal(f"bottom 5 advisors by {metric}", metric)
    assert ("asc" if signal else "desc") == expected


def test_worst_and_bottom_agree(org):
    """They name the same end, so they must resolve identically for both
    polarities."""
    for metric in ("mtd_cleared", "overdue"):
        assert _sort_signal(f"worst advisors by {metric}", metric) == \
               _sort_signal(f"bottom advisors by {metric}", metric), metric


def test_lowest_and_highest_stay_absolute(org):
    """ABSOLUTE words name a numeric end regardless of polarity —
    "lowest overdue" is the smallest number, not the worst performer."""
    assert _sort_signal("lowest overdue", "overdue") is True
    assert _sort_signal("highest overdue", "overdue") is False
    assert _sort_signal("lowest revenue", "mtd_cleared") is True
    assert _sort_signal("highest revenue", "mtd_cleared") is False


def test_bottom_is_ranking_evidence():
    """Without this the query is not a leaderboard at all — it resolved
    to `unresolved` and produced no ranking."""
    from app.llm import token_match

    assert "bottom" in cat.RANKING_STRONG
    assert token_match.contains_any("bottom 5 advisors", cat.RANKING_STRONG)


def test_the_spec_example_query_works(org):
    """Verbatim from the specification's example list."""
    ir, rows = _run("Who are the bottom 5 advisors by revenue this month?", org)
    assert ir.limit == 5
    assert ir.sort.direction == "asc"
    assert len(rows) == 5


# =====================================================================
# FIX 2 — rounding before banding
# =====================================================================

@pytest.mark.parametrize("raw,expected_status", [
    (84.49, "yellow"),
    (84.50, "green"),     # half UP, like JS Math.round
    (84.51, "green"),
    (59.49, "red"),
    (59.50, "yellow"),
    (59.51, "yellow"),
])
def test_status_is_banded_on_the_rounded_value(raw, expected_status):
    assert status_for("achievement_pct", raw) == expected_status


def test_rounding_is_half_up_not_bankers():
    """Python's round() is banker's rounding: round(84.5) == 84, which
    would put the exact boundary case in the WRONG band. The dashboard is
    JavaScript, where Math.round(84.5) === 85."""
    assert round_half_up(84.5) == 85
    assert round_half_up(59.5) == 60
    assert round_half_up(0.5) == 1
    assert round(84.5) == 84          # the built-in this replaces


@pytest.mark.parametrize("value", [85.0, 60.0, 100.0, 0.0])
def test_exact_band_edges_are_unchanged(value):
    """Rounding must not move a value that is already an integer."""
    assert round_half_up(value) == value


def test_rounding_lives_in_exactly_one_place():
    """Centralised in Thresholds.status(), the single point where a value
    becomes a status. A second implementation is how the two surfaces
    diverged in the first place."""
    import inspect

    from app.llm.metric_ontology import Thresholds

    assert "round_half_up" in inspect.getsource(Thresholds.status)


def test_a_none_value_has_no_status():
    assert status_for("achievement_pct", None) is None


# =====================================================================
# FIX 3 — Connect->CR denominator
# =====================================================================

def test_connect_to_cr_divides_by_answered_calls(org):
    """SPEC: `CR / AnsweredCalls x 100`. Every advisor has cr=30,
    answered=200, connects=100 — so the right answer is 15%, and the old
    connects denominator gave 30%."""
    ir = QueryIR(intent="leaderboard", subject_level="advisor",
                 metric=MetricRef(key="connect_to_cr_rate"),
                 sort=Sort(metric="connect_to_cr_rate"))
    rows = compile_and_run(org, ir)

    assert rows[0]["value"] == pytest.approx(15.0)
    assert rows[0]["value"] != pytest.approx(30.0)   # the old denominator


def test_the_rollup_uses_the_same_denominator(org):
    """6 advisors: cr 180 total, answered 1200 total -> 15%. Leaderboard,
    comparison and summary must all agree."""
    assert aggregation.metric_value(org, "team", "Blue Area",
                                    "connect_to_cr_rate") == pytest.approx(15.0)


def test_every_path_agrees_on_connect_to_cr(org):
    from app.services import comparison_service

    ir = QueryIR(intent="leaderboard", subject_level="team",
                 metric=MetricRef(key="connect_to_cr_rate"),
                 sort=Sort(metric="connect_to_cr_rate"))
    leaderboard = compile_and_run(org, ir)[0]["value"]
    engine = aggregation.metric_value(org, "team", "Blue Area", "connect_to_cr_rate")
    comparison = comparison_service._metric_value(org, "team", "Blue Area", "connect_to_cr_rate")

    assert leaderboard == pytest.approx(engine) == pytest.approx(comparison)


def test_the_extra_table_is_joined_not_cross_joined(org):
    """The failure mode this guards. Referencing Calls without joining it
    does not raise — SQLAlchemy appends it to the FROM clause, and the
    denominator gets multiplied by the row count. With 6 advisors a
    cartesian product would give 15/6 = 2.5, not 15."""
    value = aggregation.metric_value(org, "team", "Blue Area", "connect_to_cr_rate")
    assert value == pytest.approx(15.0)
    assert value != pytest.approx(2.5)


def test_the_binding_declares_its_extra_table():
    from app.database.models import Calls as CallsModel
    from app.llm.metric_ontology import METRICS

    binding = METRICS["connect_to_cr_rate"].bindings["advisor"]
    assert CallsModel in binding.join_models


def test_the_other_funnel_ratios_are_untouched(org):
    """Only Connect->CR was wrong. CR->Meeting and Meeting->Conversion
    stay on SalesFunnel and must not have moved."""
    assert aggregation.metric_value(org, "team", "Blue Area",
                                    "cr_to_meeting_rate") == pytest.approx(60 / 180 * 100)
    assert aggregation.metric_value(org, "team", "Blue Area",
                                    "meeting_to_conversion_rate") == pytest.approx(12 / 60 * 100)


# =====================================================================
# FIX 4 — pass/fail thresholds
# =====================================================================

@pytest.mark.parametrize("metric", ["conversion", "pipeline_value", "portfolio_value"])
def test_pass_fail_metrics_have_a_status(metric):
    """These returned None — no colour at all, not a wrong one."""
    assert status_for(metric, 0) == "red"
    assert status_for(metric, 1) == "green"
    assert status_for(metric, 10_000) == "green"


@pytest.mark.parametrize("metric", ["conversion", "pipeline_value", "portfolio_value"])
def test_pass_fail_metrics_are_not_banded(metric):
    """SPEC grounding prompt #3: these are pass/fail on whether the total
    is >0, NOT percentage bands. A value of 50 is green here and would be
    red under the 85/60 bands."""
    from app.llm.metric_ontology import METRICS

    thresholds = METRICS[metric].thresholds
    assert thresholds is not None
    assert not thresholds.is_banded
    assert status_for(metric, 50) == "green"


def test_overdue_keeps_its_inverted_rule():
    """SPEC: Overdue is green ONLY at zero — the mirror image of the
    other three, and it must not be flattened into them."""
    assert status_for("overdue", 0) == "green"
    assert status_for("overdue", 1) == "red"
    assert PASS_FAIL_POSITIVE.zero_is_green is False


def test_banded_metrics_are_unaffected():
    assert status_for("achievement_pct", 90) == "green"
    assert status_for("achievement_pct", 70) == "yellow"
    assert status_for("achievement_pct", 40) == "red"


def test_the_pass_fail_rule_is_declared_once():
    """Three metrics, one constant — not three copies of the shape."""
    from app.llm.metric_ontology import METRICS

    for metric in ("conversion", "pipeline_value", "portfolio_value"):
        assert METRICS[metric].thresholds is PASS_FAIL_POSITIVE


def test_no_metric_produces_a_cartesian_product(org, recwarn):
    """A cross join does NOT raise — SQLAlchemy emits a warning and
    returns a number. For a pure ratio that number is even correct, since
    both sums scale by the same factor. So the only reliable detector is
    the warning itself, and it is checked for EVERY metric at both a leaf
    and a group level."""
    import warnings

    from app.llm.metric_ontology import METRICS

    offenders = []
    for key in METRICS:
        for level in ("advisor", "team"):
            ir = QueryIR(intent="leaderboard", subject_level=level,
                         metric=MetricRef(key=key), sort=Sort(metric=key))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                compile_and_run(org, ir)
            if any("cartesian product" in str(w.message).lower() for w in caught):
                offenders.append(f"{key}@{level}")

    assert not offenders, f"cartesian product in: {offenders}"
