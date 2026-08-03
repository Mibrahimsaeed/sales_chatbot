"""Relational reference parsing (app/llm/reference_parser.py) — M1.

Two properties carry the design:

1. It must fire on a relationship and NOT on a topic. "Waqar Haider's
   team" is a reference; "team performance" is a subject heading. A
   parser that confused them would inject a group filter into queries
   that never asked for one — a silent wrong-scope answer, the failure
   class this whole programme exists to remove.
2. Its vocabulary must be DERIVED from the M0 declarations, so declaring
   a relationship makes its phrasings parseable with no edit here.
"""

import pytest

from app.llm import reference_parser
from app.llm.reference_parser import parse, references_to


def _levels(text):
    return [r.target_level for r in parse(text)]


@pytest.mark.parametrize("text,level", [
    ("Tell me about Waqar Haider's team", "team"),
    ("How is Waqar Haider's team doing", "team"),
    ("Top 5 advisors in Waqar Haider's company", "company"),
    ("Waqar Haider's teams", "team"),
    ("show me Adeel Dogar's business center", "office"),
    ("Waqar Haider's unit head", "unit_head"),
])
def test_possessive_references_are_detected(text, level):
    assert level in _levels(text)


@pytest.mark.parametrize("text", [
    "team performance",
    "company revenue this month",
    "top 5 advisors by revenue",
    "how is Blue Area doing",
    "teamwork",              # \b guard: "team" must not match inside a word
    "",
])
def test_non_references_are_not_detected(text):
    assert parse(text) == []


def test_pronoun_forms_are_not_handled_in_m1():
    """"his team" is milestone M4 — it needs conversation state, which
    this parser deliberately has no access to."""
    assert parse("how is his team doing") == []


def test_source_span_is_captured_for_later_milestones():
    """M1 resolves the source from the query-wide identity resolution,
    but compound references (M6) need the span per reference."""
    [reference] = parse("Waqar Haider's team")
    assert "waqar haider" in reference.source_span.lower()
    assert reference.matched_text.strip().endswith("team")


def test_multiple_references_are_all_returned_in_order():
    references = parse("compare Waqar Haider's team with Sana Tariq's company")
    assert [r.target_level for r in references] == ["team", "company"]
    assert "waqar haider" in references[0].source_span.lower()
    assert "sana tariq" in references[1].source_span.lower()


def test_longest_keyword_wins():
    """"business center" must not be claimed by "center"."""
    [reference] = parse("Adeel Dogar's business center")
    assert reference.target_level == "office"


def test_vocabulary_is_derived_from_the_relation_declarations():
    """Every declared advisor relation that has level keywords is
    parseable — no hand-maintained list in this module."""
    from app.llm import hierarchy, relations

    for target in relations.registry.targets_for("advisor"):
        keywords = hierarchy.LEVEL_KEYWORDS.get(target)
        if not keywords:
            continue          # e.g. regional_manager has no level keywords
        assert target in _levels(f"Waqar Haider's {keywords[0]}")


def test_references_to_filters_by_enabled_levels():
    text = "compare Waqar Haider's team with his company"
    assert [r.target_level for r in references_to(text, {"team"})] == ["team"]
    assert references_to(text, set()) == []
    assert references_to(text, None) == []


def test_parse_is_pure_and_total():
    """No database, no config, no state — same input, same output."""
    text = "Waqar Haider's team"
    assert parse(text) == parse(text)
    for junk in ("'s", "s'", "'s 's", "   ", "'s team's team"):
        parse(junk)  # must not raise
