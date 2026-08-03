"""Thresholds travel intact: extractor -> plan -> IR -> compiler -> rows.

Step 2 gave QueryPlan a `thresholds` field and taught plan_to_ir to turn
it into filters, which fixed the DROPPED-constraint half of F8. This file
covers the rest of the comparator vocabulary and, more importantly, the
half that was worse than dropping:

  POLARITY. "no more than 50" parsed as "> 50" and "no less than 50" as
  "< 50" — the exact COMPLEMENT of what was asked. A dropped filter
  returns a superset, which at least contains the right answer; an
  inverted one returns precisely the rows the user ruled out.

  The cause was a declaration in the wrong field: the negations lived in
  Comparator.exemplars, which only resolve through an embedding call, so
  with no LLM reachable they never matched and the shorter "more than"
  inside them did. comparators.py's own docstring already claimed
  "'no more than' before 'more than'" — the ordering machinery was right,
  the phrases were just declared somewhere it couldn't see them.

  RANGES. "between 60 and 80" produced no filter at all. It now compiles
  to two AND-combined filters (>= 60, <= 80) — QueryIR.filters is already
  AND-combined, so a range needs no new operator, no SQL change and no
  change to the LLM schema.

Every test here asserts the ROWS, not just the IR. An IR holding the
right operator is not the same as an answer that respects it.
"""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import comparators, entity_extractor
from app.llm.entity_extractor import extract_entities
from app.llm.preprocessing import normalize
from app.llm.query_compiler import compile_and_run
from app.llm.query_ir import plan_to_ir
from app.llm.query_planner import build_query_plan


@pytest.fixture()
def org(db_session):
    """Achievement spread so every boundary is distinguishable:
    90, 80, 70, 60, 50. A comparator that is off by one end, or
    inverted, selects a visibly different set."""
    for wid, pct in ((1, 90.0), (2, 80.0), (3, 70.0), (4, 60.0), (5, 50.0)):
        db_session.add(Advisor(wid=wid, name=f"Adv {int(pct)}", team=f"Team {int(pct)}",
                               company="Graana", in_master_sheet=True))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=pct, pct=pct))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=wid, mtd_followup_connect=0))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    return db_session


def _run(text, db):
    cleaned = normalize(text)
    entities = extract_entities(cleaned, db)
    ir = plan_to_ir(build_query_plan(cleaned, entities), entities)
    rows = compile_and_run(db, ir) or []
    filters = [(f.operator, f.value) for f in ir.filters if f.field == "achievement_pct"]
    return filters, sorted(r["value"] for r in rows)


# ---------------------------------------------------------------------
# The two required cases
# ---------------------------------------------------------------------

def test_advisors_above_80_percent_achievement(org):
    """Required case 1. achievement_pct > 80."""
    filters, values = _run("Advisors above 80% achievement", org)

    assert filters == [(">", 80.0)]
    assert values == [90.0]          # strict: 80 itself is excluded


def test_teams_between_60_and_80_achievement(org):
    """Required case 2. A range — two AND-combined bounds."""
    filters, values = _run("Teams between 60 and 80 achievement", org)

    assert sorted(filters) == [("<=", 80.0), (">=", 60.0)]
    assert values == [60.0, 70.0, 80.0]   # inclusive at both ends


# ---------------------------------------------------------------------
# Every comparator, end to end
# ---------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_filters,expected_values", [
    ("advisors above 80 percent achievement", [(">", 80.0)], [90.0]),
    ("advisors over 80 percent achievement", [(">", 80.0)], [90.0]),
    ("advisors with more than 80 percent achievement", [(">", 80.0)], [90.0]),
    ("advisors with at least 80 percent achievement", [(">=", 80.0)], [80.0, 90.0]),
    ("advisors below 60 percent achievement", [("<", 60.0)], [50.0]),
    ("advisors under 60 percent achievement", [("<", 60.0)], [50.0]),
    ("advisors with less than 60 percent achievement", [("<", 60.0)], [50.0]),
    ("advisors with at most 60 percent achievement", [("<=", 60.0)], [50.0, 60.0]),
])
def test_each_comparator_selects_the_right_rows(org, text, expected_filters, expected_values):
    filters, values = _run(text, org)
    assert filters == expected_filters, text
    assert values == expected_values, text


def test_strict_and_inclusive_differ_at_the_boundary(org):
    """The whole point of having both > and >=. If these agreed, the
    comparator would not be carrying any information."""
    _f, strict = _run("advisors above 80 percent achievement", org)
    _f, inclusive = _run("advisors with at least 80 percent achievement", org)

    assert 80.0 not in strict
    assert 80.0 in inclusive


# ---------------------------------------------------------------------
# Polarity — the inversions
# ---------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_operator,expected_values", [
    ("advisors with no more than 60 percent achievement", "<=", [50.0, 60.0]),
    ("advisors with not more than 60 percent achievement", "<=", [50.0, 60.0]),
    ("advisors with no greater than 60 percent achievement", "<=", [50.0, 60.0]),
    ("advisors not above 60 percent achievement", "<=", [50.0, 60.0]),
    ("advisors with no less than 80 percent achievement", ">=", [80.0, 90.0]),
    ("advisors with not less than 80 percent achievement", ">=", [80.0, 90.0]),
    ("advisors not below 80 percent achievement", ">=", [80.0, 90.0]),
    ("advisors not under 80 percent achievement", ">=", [80.0, 90.0]),
])
def test_a_negated_comparator_is_not_inverted(org, text, expected_operator, expected_values):
    filters, values = _run(text, org)
    assert filters == [(expected_operator, filters[0][1])], text
    assert values == expected_values, text


def test_a_negation_emits_exactly_one_filter(org):
    """The subtle half of the fix. Patterns are longest-first but the
    extractor collected EVERY match, so "no more than 60" matched the
    negation AND the "more than 60" inside it — emitting <= 60 and > 60
    together, a contradiction that returns nothing and reads as "no
    results" rather than as a misreading."""
    filters, values = _run("advisors with no more than 60 percent achievement", org)

    assert len(filters) == 1
    assert values          # not an empty contradiction


def test_a_negation_is_the_complement_of_its_positive(org):
    """"no more than 60" and "more than 60" must partition the advisors
    between them. Inverted polarity made them identical."""
    _f, negated = _run("advisors with no more than 60 percent achievement", org)
    _f, positive = _run("advisors with more than 60 percent achievement", org)

    assert set(negated) & set(positive) == set()
    assert set(negated) | set(positive) == {50.0, 60.0, 70.0, 80.0, 90.0}


# ---------------------------------------------------------------------
# Suffix forms
# ---------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_operator,expected_values", [
    ("advisors with 80 percent or higher achievement", ">=", [80.0, 90.0]),
    ("advisors with 80 percent or more achievement", ">=", [80.0, 90.0]),
    ("advisors with 60 percent or lower achievement", "<=", [50.0, 60.0]),
    ("advisors with 60 percent or less achievement", "<=", [50.0, 60.0]),
])
def test_a_comparator_after_the_number_still_parses(org, text, expected_operator, expected_values):
    filters, values = _run(text, org)
    assert filters == [(expected_operator, filters[0][1])], text
    assert values == expected_values, text


# ---------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_values", [
    ("teams with achievement between 60 and 80", [60.0, 70.0, 80.0]),
    ("advisors between 70 and 90 achievement", [70.0, 80.0, 90.0]),
    ("advisors with achievement between 50 and 60", [50.0, 60.0]),
])
def test_a_range_bounds_both_ends(org, text, expected_values):
    filters, values = _run(text, org)
    assert len(filters) == 2, text
    assert values == expected_values, text


def test_a_range_is_inclusive_at_both_ends(org):
    _f, values = _run("advisors with achievement between 60 and 80", org)
    assert 60.0 in values and 80.0 in values


def test_a_backwards_range_is_read_as_the_same_range(org):
    """"between 80 and 60" is the same request said backwards. Taken
    literally it emits >= 80 AND <= 60, which matches nothing and looks
    like an empty result rather than a misreading."""
    _f, forwards = _run("advisors with achievement between 60 and 80", org)
    _f, backwards = _run("advisors with achievement between 80 and 60", org)
    assert forwards == backwards


def test_the_and_in_a_range_is_not_two_constraints(org):
    """Matched before the single-comparator patterns and its span
    consumed, so neither bound is re-read on its own."""
    filters, _v = _run("advisors with achievement between 60 and 80", org)
    assert sorted(filters) == [("<=", 80.0), (">=", 60.0)]


def test_a_numeric_range_is_not_mistaken_for_a_date_range(org):
    """temporal_parser refuses custom date ranges ("between Jan 1 and Mar
    31"). A numeric range must not trip that and become a refusal."""
    from app.llm.temporal_parser import parse_period

    cleaned = normalize("advisors with achievement between 60 and 80")
    assert parse_period(cleaned) is None
    assert "period_unsupported" not in extract_entities(cleaned, org)


# ---------------------------------------------------------------------
# The chain, and coexistence
# ---------------------------------------------------------------------

def test_the_threshold_is_on_the_plan_before_it_is_on_the_ir(org):
    """The link Step 2 added: QueryPlan is where it used to die."""
    cleaned = normalize("advisors with achievement between 60 and 80")
    plan = build_query_plan(cleaned, extract_entities(cleaned, org))
    assert plan.thresholds == [
        {"operator": ">=", "value": 60.0},
        {"operator": "<=", "value": 80.0},
    ]


def test_a_range_coexists_with_an_entity_filter(org):
    cleaned = normalize("advisors in Team 70 with achievement between 60 and 80")
    entities = extract_entities(cleaned, org)
    ir = plan_to_ir(build_query_plan(cleaned, entities), entities)

    assert {f.field for f in ir.filters} == {"team", "achievement_pct"}
    assert [r["value"] for r in compile_and_run(org, ir)] == [70.0]


def test_a_query_with_no_threshold_is_unaffected(org):
    filters, values = _run("top advisors by achievement", org)
    assert filters == []
    assert values == [50.0, 60.0, 70.0, 80.0, 90.0]


# ---------------------------------------------------------------------
# The registry is the single source
# ---------------------------------------------------------------------

def test_every_declared_phrase_parses_deterministically():
    """The defect was a phrase declared where only an embedding call
    could see it. Every phrase and suffix_phrase must parse with no LLM
    at all — that is what `phrases` means."""
    from app.llm.entity_extractor import _extract_thresholds

    for comparator in comparators.COMPARATORS:
        for phrase in comparator.phrases:
            got = _extract_thresholds(f"achievement {phrase} 42")
            assert got == [{"operator": comparator.operator, "value": 42.0}], phrase
        for phrase in comparator.suffix_phrases:
            got = _extract_thresholds(f"achievement 42 {phrase}")
            assert got == [{"operator": comparator.operator, "value": 42.0}], phrase


def test_no_phrase_is_declared_in_two_operators():
    """A phrase claimed by two operators would resolve by pattern length,
    silently."""
    seen: dict[str, str] = {}
    for comparator in comparators.COMPARATORS:
        for phrase in comparator.phrases + comparator.suffix_phrases:
            assert phrase not in seen, f"{phrase!r}: {seen.get(phrase)} and {comparator.operator}"
            seen[phrase] = comparator.operator


def test_negations_are_deterministic_not_exemplars():
    """Regression guard for the exact mistake: a negation in `exemplars`
    is invisible without an LLM, and the positive phrase inside it wins."""
    for comparator in comparators.COMPARATORS:
        for exemplar in comparator.exemplars:
            assert not exemplar.startswith(("no ", "not ")), exemplar
