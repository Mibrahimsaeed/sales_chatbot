"""Phase 5.1 — one authoritative interpretation of periods.

The vocabulary was written out by hand in FIVE places: llm_client's JSON
schema, planner_schema's Literal and its JSON schema, prompt_builder's IR
schema, planner_prompt's rules. They agreed only by luck, and the moment
DAILY was added they stopped.

That was not cosmetic. llm_client's copy is sent with `strict: True`, so
grammar-constrained decoding made DAILY *unemittable*: an LLM-parsed
"revenue today" was forced back to MTD. Finding F5 was fixed on the
rule-based path and still live on the default one — the same shape as the
hierarchy drift Phase 5 fixed for levels, repeated for periods.

The vocabulary now lives in app/llm/periods.py, a leaf module with no
imports (llm_client cannot import temporal_parser: that module reaches
entity_linker -> embeddings -> llm_client). Every consumer derives from
it, and the tests below assert they cannot diverge again.
"""

import pytest

from app.llm import periods, planner_schema, prompt_builder, temporal_parser
from app.llm.llm_client import QUERY_IR_JSON_SCHEMA
from app.llm.planner_prompt import _RULES
from app.llm.query_ir import TimeRange


# ---------------------------------------------------------------------
# One source of truth
# ---------------------------------------------------------------------

def test_the_ir_schema_enum_is_the_vocabulary():
    """The grammar-constrained one. If this narrows, the LLM silently
    loses the ability to express a period the parser recognises."""
    enum = QUERY_IR_JSON_SCHEMA["properties"]["time_range"]["properties"]["period"]["enum"]
    assert list(enum) == list(periods.PERIODS)


def test_the_planner_schema_is_the_vocabulary():
    assert list(planner_schema.Period.__args__) == list(periods.PERIODS)
    enum = planner_schema.QUERY_PLAN_JSON_SCHEMA["properties"]["period"]["anyOf"][0]["enum"]
    assert list(enum) == list(periods.PERIODS)


def test_the_ir_time_range_is_the_vocabulary():
    assert list(TimeRange.model_fields["period"].annotation.__args__) == list(periods.PERIODS)


def test_the_ir_prompt_offers_the_vocabulary():
    schema_text = prompt_builder._ir_schema()
    for period in periods.PERIODS:
        assert f'"{period}"' in schema_text, period


def test_the_planner_prompt_offers_the_vocabulary():
    for period in periods.PERIODS:
        assert period in _RULES, period


def test_temporal_parser_re_exports_the_same_vocabulary():
    """Callers importing PERIODS from temporal_parser must get the same
    tuple, not a copy that can drift."""
    assert temporal_parser.PERIODS is periods.PERIODS


def test_every_period_has_a_human_label():
    """A reply that has to explain a refusal needs wording for the window
    it could not use."""
    for period in periods.PERIODS:
        assert periods.label_for(period) != period, period


def test_the_vocabulary_module_has_no_dependencies():
    """The reason it exists. If it grows an app import, llm_client's
    schema can cycle again and the hardcoded copies come back."""
    import pathlib

    source = pathlib.Path(periods.__file__).read_text()
    assert "from app." not in source
    assert "import app." not in source


def test_daily_is_emittable_by_the_llm():
    """The F5 remnant this phase fixes, stated as its own test. With
    strict decoding an absent enum value cannot be produced at all."""
    enum = QUERY_IR_JSON_SCHEMA["properties"]["time_range"]["properties"]["period"]["enum"]
    assert "DAILY" in enum


# ---------------------------------------------------------------------
# The required vocabulary
# ---------------------------------------------------------------------

@pytest.mark.parametrize("phrase,expected", [
    ("today", "DAILY"),
    ("currently", "MTD"),
    ("current", "MTD"),
    ("right now", "DAILY"),
    ("this morning", "DAILY"),
    ("this afternoon", "DAILY"),
    ("this evening", "DAILY"),
    ("this month", "MTD"),
    ("this year", "YTD"),
])
def test_the_required_vocabulary_resolves(phrase, expected):
    match = temporal_parser.parse_period(phrase)
    assert match is not None, phrase
    assert match.kind == "equivalent", phrase
    assert match.period == expected, phrase


def test_this_week_is_recognised_and_refused():
    """The tenth required phrase. It IS in the vocabulary — the parser
    recognises it and explains itself — but there is no week-level data,
    so per requirement 4 it fails honestly rather than falling back."""
    match = temporal_parser.parse_period("this week")
    assert match is not None
    assert match.kind == "unsupported"
    assert "week" in match.reason
    assert temporal_parser.match_period("this week") is None


@pytest.mark.parametrize("phrase,expected", [
    ("current month", "MTD"),
    ("current year", "YTD"),
    ("current quarter", "3M"),
    ("tonight", "DAILY"),
    ("as of now", "DAILY"),
    ("at the moment", "MTD"),
])
def test_the_qualified_forms_beat_the_bare_one(phrase, expected):
    """A bare "current" means MTD, so "current year" must be matched
    first or it would be swallowed as MTD — the classic longest-first
    requirement, here between members of one family."""
    assert temporal_parser.match_period(phrase) == expected, phrase


# ---------------------------------------------------------------------
# Unsupported windows still refuse
# ---------------------------------------------------------------------

@pytest.mark.parametrize("phrase", [
    "yesterday", "last month", "previous month", "last week",
    "past 7 days", "last 14 days", "last quarter", "previous quarter",
])
def test_an_unsupported_window_fails_honestly(phrase):
    """Requirement 4. Adding vocabulary must not turn a real calendar
    window into a silent guess."""
    match = temporal_parser.parse_period(phrase)
    assert match.kind == "unsupported", phrase
    assert match.reason, phrase
    assert temporal_parser.match_period(phrase) is None, phrase


def test_a_new_phrase_did_not_shadow_an_unsupported_one():
    """"current" is a substring of nothing dangerous, but "this
    morning"/"tonight" sit next to "yesterday"/"last week" in meaning.
    The unsupported list is checked FIRST, and that must stay true."""
    assert temporal_parser.parse_period("yesterday morning").kind == "unsupported"
    assert temporal_parser.parse_period("last week").kind == "unsupported"


@pytest.mark.parametrize("phrase,expected", [
    ("month to date", "MTD"),
    ("mtd", "MTD"),
    ("year to date", "YTD"),
    ("ytd", "YTD"),
    ("this quarter", "3M"),
    ("last 3 months", "3M"),
    ("three months", "3M"),
    ("3m", "3M"),
])
def test_the_pre_existing_vocabulary_is_unchanged(phrase, expected):
    assert temporal_parser.match_period(phrase) == expected, phrase


def test_a_query_with_no_time_words_names_no_period():
    """The common case. A period must not be invented for a question
    that did not ask for one — the planner falls back to the metric's own
    window only when nothing was stated."""
    assert temporal_parser.parse_period("top 5 advisors by revenue") is None
    assert temporal_parser.parse_period("who is Yasir Ali's unit head") is None
