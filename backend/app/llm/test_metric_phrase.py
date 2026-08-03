"""Metric phrasing and metric-request evidence (M7).

`metric_phrase` derives a sentence-shaped name from the curated label
rather than storing a second per-metric string, so a metric added later
reads correctly without anyone remembering to add a phrase for it. These
tests hold that derivation to English.
"""

import pytest

from app.llm import intent_catalog as cat
from app.llm.metric_ontology import (
    METRICS, metric_phrase, resolve_metric, resolve_metric_evidence,
)
from app.llm.response_formatter import format_advisor_metric_reply


@pytest.mark.parametrize("key,expected", [
    ("total_connects", "MTD connects"),        # "Total MTD Connects"
    ("total_meetings", "MTD meetings"),        # "Total MTD Meetings"
    ("pipeline_value", "MTD open pipeline"),
    ("mtd_target", "MTD target"),
    ("mtd_cleared", "MTD revenue cleared"),
    ("bookings", "MTD bookings stored"),
    ("ytd_cleared", "YTD revenue cleared"),
])
def test_phrases_read_as_english(key, expected):
    assert metric_phrase(key) == expected


def test_acronyms_survive_lowering():
    assert "MTD" in metric_phrase("total_connects")
    assert "YTD" in metric_phrase("ytd_cleared")


def test_every_metric_has_a_usable_phrase():
    """Structural: a metric added later must not produce an empty or
    upper-cased phrase."""
    for key in METRICS:
        phrase = metric_phrase(key)
        assert phrase and phrase == phrase.rstrip()
        assert not phrase.lower().startswith("total ")


def test_unknown_metric_degrades_quietly():
    assert metric_phrase(None) == "value"
    assert metric_phrase("no_such_metric") == "no_such_metric"


# ---------------------------------------------------------------------
# Evidence — which synonym matched
# ---------------------------------------------------------------------

def test_evidence_reports_the_matched_synonym():
    assert resolve_metric_evidence("connects of X") == ("total_connects", "connects")
    assert resolve_metric_evidence("tell me about X") is None


def test_resolve_metric_is_unchanged_by_the_evidence_split():
    """Backward compatibility: the old entry point still returns a key."""
    assert resolve_metric("connects of X") == "total_connects"
    assert resolve_metric("tell me about X") is None


def test_general_interest_words_are_recognised_as_such():
    key, synonym = resolve_metric_evidence("performance of X")
    assert key == "achievement_pct"
    assert synonym in cat.GENERAL_INTEREST_SYNONYMS


def test_a_specific_phrase_containing_a_general_word_is_specific():
    """_SYNONYM_INDEX is longest-first, so the specific phrase wins."""
    _key, synonym = resolve_metric_evidence("performance against target of X")
    assert synonym not in cat.GENERAL_INTEREST_SYNONYMS


def test_cr_booked_resolves_to_bookings():
    assert resolve_metric("cr booked of X") == "bookings"
    assert resolve_metric("cr bookings of X") == "bookings"


def test_bare_cr_is_not_a_metric():
    """A two-letter synonym would fire inside ordinary words — synonym
    matching is plain substring."""
    assert resolve_metric("increase for X") != "bookings"
    assert resolve_metric("concrete plans for X") != "bookings"


# ---------------------------------------------------------------------
# Reply shape
# ---------------------------------------------------------------------

def test_counts_render_as_integers():
    assert format_advisor_metric_reply("A", "total_connects", 2.0) == "A has 2 MTD connects."


def test_large_numbers_keep_separators():
    assert format_advisor_metric_reply("A", "pipeline_value", 7500.0) == "A has 7,500 MTD open pipeline."


def test_fractions_keep_one_decimal():
    assert format_advisor_metric_reply("A", "conversion", 12.34).endswith("12.3 MTD conversions.")


def test_percentage_metrics_read_as_percentages():
    assert format_advisor_metric_reply("A", "achievement_pct", 80.0) == (
        "A is at 80% target achievement."
    )


def test_absent_value_is_stated_not_zeroed():
    assert format_advisor_metric_reply("A", "total_connects", None) == (
        "I don't have MTD connects on file for A."
    )
