"""Phase 13B — a question naming several measures gets several answers.

THE DEFECT. `metric_aliases.resolve()` scans an index sorted
longest-phrase-first and returns on the first hit, so "connects and
answered calls" resolved to `answered_calls` and the other measure was
gone before the planner, the IR or the compiler could have carried it.
The surviving one was not even the first named — the winner is whichever
ALIAS STRING is longer, so "answered calls and connects" resolved to
`answered_calls` too.

The reply that came back was a correct number, correctly labelled, for
one of the two measures asked about, with nothing anywhere saying the
other had been dropped. That is the failure mode these tests exist to
make impossible: a partial answer indistinguishable from a complete one.

WHAT CHANGED, AND WHAT DID NOT. `resolve()` is untouched — every existing
caller keeps the longest-alias winner. `resolve_all()` continues the same
scan over the same index with the same matcher, masking each hit so the
matches cannot overlap. Nothing about how a SINGLE measure resolves is
different, which is why the single-metric tests below sit beside the new
ones rather than being replaced by them.
"""

import pytest

from app.llm import metric_aliases, token_match
from app.llm.metric_intent import detect


# ---------------------------------------------------------------------
# resolve_all — detection
# ---------------------------------------------------------------------


def test_a_single_measure_resolves_to_a_list_of_one():
    assert [m.metric for m in metric_aliases.resolve_all("connects")] == ["total_connects"]


def test_two_measures_both_survive():
    assert [m.metric for m in metric_aliases.resolve_all("connects and answered calls")] == [
        "total_connects", "answered_calls"]


def test_three_measures_all_survive():
    assert [m.metric for m in metric_aliases.resolve_all(
        "connects, answered calls and client registrations")] == [
        "total_connects", "answered_calls", "client_registrations"]


def test_the_order_is_the_order_they_were_NAMED_not_alias_length():
    """The old winner was whichever alias string was longer, which is why
    "answered calls and connects" resolved to answered_calls. Position in
    the text is the only ordering a reader would predict."""
    assert [m.metric for m in metric_aliases.resolve_all("answered calls and connects")] == [
        "answered_calls", "total_connects"]
    assert [m.metric for m in metric_aliases.resolve_all("connects and answered calls")] == [
        "total_connects", "answered_calls"]


def test_one_measure_named_twice_counts_once():
    assert [m.metric for m in metric_aliases.resolve_all("connects and connections")] == [
        "total_connects"]


def test_matches_do_not_overlap():
    """Without masking, "answered calls" matches and then "calls" matches
    again inside the span it already claimed — two measures reported
    where the user named one."""
    metrics = [m.metric for m in metric_aliases.resolve_all("answered calls")]
    assert metrics == ["answered_calls"]


def test_a_period_qualified_phrase_resolves_to_its_own_key():
    assert [m.metric for m in metric_aliases.resolve_all("ytd connects")] == ["ytd_connects"]


def test_text_naming_no_measure_resolves_to_nothing():
    assert metric_aliases.resolve_all("who is on the Blue Area team") == []


@pytest.mark.parametrize("text,expected", [
    ("connects", "total_connects"),
    ("connects and answered calls", "answered_calls"),
    ("answered calls and connects", "answered_calls"),
    ("revenue", "mtd_cleared"),
    ("ytd connects", "ytd_connects"),
])
def test_resolve_is_completely_unchanged(text, expected):
    """The single-metric entry point keeps its exact behaviour, including
    the longest-alias winner on multi-metric text — every existing caller
    depends on it and none of them was asked to change."""
    assert metric_aliases.resolve(text).metric == expected


# ---------------------------------------------------------------------
# token_match.find — the positional half
# ---------------------------------------------------------------------


def test_find_reports_where_the_phrase_is():
    assert token_match.find("give me the connects please", "connects").start() == 12


def test_find_and_contains_agree():
    for text, phrase in [("the connects", "connects"), ("across the board", "cr"),
                         ("answered calls", "calls"), ("nothing here", "connects")]:
        assert (token_match.find(text, phrase) is not None) == token_match.contains(text, phrase)


# ---------------------------------------------------------------------
# metric_intent — the primary stays primary
# ---------------------------------------------------------------------


def test_detect_reports_every_named_measure():
    assert detect("connects and answered calls", {}).keys == [
        "total_connects", "answered_calls"]


def test_detect_keeps_its_primary_key_unchanged():
    """`key` is what every existing reader uses; only `keys` is new."""
    assert detect("connects and answered calls", {}).key == "answered_calls"


def test_a_single_measure_query_reports_one_key_and_is_not_multi():
    intent = detect("connects", {})
    assert intent.keys == ["total_connects"]
    assert not intent.is_multi


def test_a_caller_supplied_metric_wins_and_collapses_the_list():
    """entities["metric"] comes from a filled clarification slot or an
    ir_patcher carry — better information than the alias scan has. When
    the two disagree the scan is wrong about this query, so the list
    collapses to the primary and the turn stays single-measure."""
    intent = detect("connects and answered calls", {"metric": "mtd_cleared"})
    assert intent.key == "mtd_cleared"
    assert intent.keys == ["mtd_cleared"]
    assert not intent.is_multi


def test_keys_is_never_empty_for_a_resolved_intent():
    assert detect("revenue", {}).keys == ["mtd_cleared"]
