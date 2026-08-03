"""
The golden-query runner.

One test per case, so a failure names the question that broke rather than
reporting "42 of 131 cases failed". The diff shows only the fields the
case pins, which is what makes a failure readable: a metric regression
reports the metric, not a whole struct.

SCOPE, as specified: semantic understanding only. Nothing here executes a
metric or inspects a reply. A case asserts what the pipeline UNDERSTOOD —
intent, metric, period, level, filters, comparator, ranking, limit — and
stops there.
"""

from __future__ import annotations

import pytest

from tests.golden_queries.cases import ALL_CASES, CATEGORIES, Case
from tests.golden_queries.understanding import Understanding, observe


def _case_id(item) -> str:
    category, case = item
    return f"{category}::{case.query}"


@pytest.mark.parametrize("category,case", ALL_CASES, ids=[_case_id(i) for i in ALL_CASES])
def test_golden_query(org, category: str, case: Case):
    """The pipeline's understanding must match the case exactly on every
    field the case pins."""
    got = observe(case.query, org)
    expected = dict(case.expect)
    actual = got.compared_on(case.keys)

    if actual != expected:
        pytest.fail(_report(case, expected, actual, got))


def _report(case: Case, expected: dict, actual: dict, got: Understanding) -> str:
    lines = [
        f"query: {case.query!r}",
        "",
        f"{'field':<14}{'expected':<34}{'actual'}",
    ]
    for key in expected:
        mark = " " if expected[key] == actual[key] else ">"
        lines.append(f"{mark}{key:<13}{str(expected[key]):<34}{actual[key]}")
    lines += ["", f"full understanding: {got}"]
    if case.known_gap:
        lines += ["", "This case is marked known_gap:", f"  {case.known_gap}",
                  "",
                  "A known_gap case pins CURRENT behaviour. If the behaviour improved,",
                  "update the case and delete the known_gap note."]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Properties of the corpus itself
# ---------------------------------------------------------------------

def test_the_corpus_covers_every_required_category():
    required = {
        "leaderboards", "comparisons", "kpi_questions", "hierarchy",
        "periods", "thresholds", "attendance",
        # Phase 5.6
        "reverse_hierarchy", "kpi_terminology", "clarifications",
        "ambiguous_entities", "negative_cases",
    }
    assert set(CATEGORIES) == required


def test_the_corpus_is_at_least_a_hundred_queries():
    assert len(ALL_CASES) >= 100, f"only {len(ALL_CASES)} cases"


def test_every_category_is_substantial():
    """A category with three cases in it is a placeholder, not coverage."""
    for category, cases in CATEGORIES.items():
        assert len(cases) >= 10, f"{category} has only {len(cases)} cases"


def test_no_query_appears_twice():
    """A duplicate is either a copy-paste or two cases disagreeing about
    the same question — both worth knowing about."""
    seen: dict[str, str] = {}
    for category, case in ALL_CASES:
        key = case.query.lower()
        assert key not in seen, f"{case.query!r} in both {seen.get(key)} and {category}"
        seen[key] = category


def test_every_case_pins_intent_and_something_else():
    """`expect` is a partial match, so a case pinning only `intent` is
    barely a test — it would pass while the metric, period and level all
    changed underneath it."""
    for category, case in ALL_CASES:
        assert "intent" in case.expect, f"{category}::{case.query} does not pin intent"
        assert len(case.expect) >= 2, f"{category}::{case.query} pins only intent"


def test_every_expected_field_exists_on_the_record():
    """A typo'd key would be silently ignored by the partial match, so the
    case would pass without asserting anything."""
    valid = set(Understanding.__dataclass_fields__)
    for category, case in ALL_CASES:
        unknown = set(case.expect) - valid
        assert not unknown, f"{category}::{case.query} pins unknown field(s) {unknown}"


def test_known_gaps_are_explained_and_few():
    """A known_gap must say what the right answer would be — a bare flag
    is indistinguishable from an untested case. They are also counted: if
    this list grows, understanding is getting worse, not better."""
    gaps = [(c, k) for k, c in ALL_CASES if c.known_gap]
    for case, _category in [(c, k) for k, c in ALL_CASES if c.known_gap]:
        assert len(case.known_gap) > 40, f"{case.query!r}: known_gap is too vague"

    assert len(gaps) <= 8, (
        f"{len(gaps)} cases are marked known_gap. Each one is a query the system "
        "misunderstands; they should be shrinking."
    )


def test_no_case_routes_an_unrelated_query_to_attendance(org):
    """Finding F1 as a standing property rather than a single case:
    'late' is a substring of calculated/related/escalate/translate, and
    attendance_filter scores 0.98 and returns before the parser runs, so a
    collision here hijacks the whole query."""
    for text in ("how is the answered calls percentage calculated",
                 "show me related teams", "escalate to the manager",
                 "please translate this", "who is the representative"):
        assert observe(text, org).intent != "attendance_filter", text


# ---------------------------------------------------------------------
# Phase 5.6 — structural coverage
#
# A corpus can grow large and still miss a whole dimension. These assert
# that every VALUE the parser can produce is exercised by some case, so a
# new metric, level, period or intent arrives with coverage rather than
# without it.
# ---------------------------------------------------------------------

def _observed(org):
    """Every case's understanding, computed once per test that needs it."""
    return [observe(case.query, org) for _category, case in ALL_CASES]


def test_every_documented_intent_is_exercised(org):
    """The catalog documents an intent; the corpus must ask for it. An
    intent nobody tests is an intent nobody notices breaking."""
    from app.llm import intent_catalog as cat

    produced = {u.intent for u in _observed(org)}
    # Normalised names differ from plan actions for two intents — see
    # understanding._PLAN_INTENTS.
    aliases = {"advisor_profile": "lookup", "hierarchy": "breakdown",
               # the planner's action for entity_summary is "summary"
               "entity_summary": "summary"}
    missing = [
        intent for intent in cat.INTENT_DOCS
        if intent not in produced and aliases.get(intent, intent) not in produced
    ]
    assert not missing, f"no case produces: {missing}"


def test_every_hierarchy_level_is_exercised(org):
    from app.llm import hierarchy

    levels = {u.level for u in _observed(org)}
    missing = [lvl for lvl in hierarchy.HIERARCHY_LEVELS if lvl not in levels]
    assert not missing, f"no case resolves to level: {missing}"


def test_every_period_is_exercised(org):
    from app.llm import periods

    seen = {u.period for u in _observed(org)}
    missing = [p for p in periods.PERIODS if p not in seen]
    assert not missing, f"no case resolves to period: {missing}"


def test_every_comparator_is_exercised(org):
    from app.llm import comparators

    seen = {op for u in _observed(org) for op in u.comparators}
    missing = [op for op in comparators.operators() if op not in seen]
    assert not missing, f"no case produces comparator: {missing}"


def test_both_sort_directions_are_exercised(org):
    directions = {u.ranking for u in _observed(org)}
    assert {"asc", "desc"} <= directions


def test_a_broad_span_of_metrics_is_exercised(org):
    """Not every metric — some have no natural phrasing yet — but a
    corpus that only ever asks for revenue is not covering the ontology.
    """
    from app.llm.metric_ontology import METRICS

    seen = {u.metric for u in _observed(org) if u.metric}
    ratio = len(seen) / len(METRICS)
    assert ratio >= 0.6, (
        f"only {len(seen)}/{len(METRICS)} metrics are exercised ({ratio:.0%})"
    )


# ---------------------------------------------------------------------
# Every resolved bug keeps a permanent regression test
# ---------------------------------------------------------------------

# The audit findings that have been FIXED. Each must be pinned by at
# least one case carrying `finding=`, so a fix cannot lose its guard when
# this file is reorganised. Findings still open are deliberately absent.
RESOLVED_FINDINGS = {
    "F1": "'late' matched inside calculated/related/escalate",
    "F4": "the user's period was extracted then discarded",
    "F6": "an unresolvable metric silently became revenue",
    "F8": "thresholds were extracted then dropped",
    "F9": "ranking words matched inside other words",
    "F10": "a manager's manager could not be asked for",
    "F13": "a percentage phrase resolved to its count",
}


@pytest.mark.parametrize("finding", sorted(RESOLVED_FINDINGS))
def test_every_resolved_finding_has_a_regression_case(finding):
    tagged = [c for _k, c in ALL_CASES if c.finding == finding]
    assert tagged, (
        f"{finding} ({RESOLVED_FINDINGS[finding]}) has no case tagged finding="
        f"{finding!r} — a fixed bug with no guard will come back"
    )


def test_finding_tags_are_known():
    """A typo'd tag would silently satisfy nothing."""
    tags = {c.finding for _k, c in ALL_CASES if c.finding}
    unknown = tags - set(RESOLVED_FINDINGS)
    assert not unknown, f"unknown finding tag(s): {unknown}"


# ---------------------------------------------------------------------
# Coverage statistics
# ---------------------------------------------------------------------

def test_coverage_report(org, capsys):
    """Prints the corpus's coverage. Always passes — it is a report, not
    a gate; the gates are the tests above. Run with `-s` to read it."""
    from app.llm import comparators, hierarchy, periods
    from app.llm.metric_ontology import METRICS

    observations = _observed(org)
    intents = {u.intent for u in observations}
    metrics = {u.metric for u in observations if u.metric}
    levels = {u.level for u in observations if u.level}
    seen_periods = {u.period for u in observations if u.period}
    ops = {op for u in observations for op in u.comparators}
    gaps = [c for _k, c in ALL_CASES if c.known_gap]
    tagged = {c.finding for _k, c in ALL_CASES if c.finding}

    lines = [
        "",
        "=" * 62,
        "GOLDEN QUERY COVERAGE",
        "=" * 62,
        f"cases                {len(ALL_CASES)} in {len(CATEGORIES)} categories",
        "",
        "by category:",
    ]
    for category, cases in sorted(CATEGORIES.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {category:<22}{len(cases):>4}")
    lines += [
        "",
        f"intents produced     {len(intents)}",
        f"metrics exercised    {len(metrics)}/{len(METRICS)}"
        f"  ({len(metrics) / len(METRICS):.0%})",
        f"levels exercised     {len(levels)}/{len(hierarchy.HIERARCHY_LEVELS)}",
        f"periods exercised    {len(seen_periods)}/{len(periods.PERIODS)}",
        f"comparators          {len(ops)}/{len(comparators.operators())}",
        f"findings pinned      {len(tagged)}  ({', '.join(sorted(tagged))})",
        f"known gaps           {len(gaps)}",
        "",
        "metrics NOT exercised:",
    ]
    unexercised = sorted(set(METRICS) - metrics)
    lines.append("  " + (", ".join(unexercised) if unexercised else "(none)"))
    lines.append("=" * 62)

    with capsys.disabled():
        print("\n".join(lines))
