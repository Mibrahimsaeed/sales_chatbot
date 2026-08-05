"""Characterisation of extract_name_spans (M0 — audit debt D1).

NOT a specification of desired behaviour — a record of ACTUAL behaviour,
captured before the Relationship Inference Engine touches anything near
it.

Why it exists: the architectural audit found that the relational
structure of "Waqar Haider's team" is destroyed before entity resolution
runs. `_SPAN_STOPWORDS` deletes "team", "s", "his", "under", "manager"
and the rest of the org vocabulary, so the resolver sees only
["waqar haider"]. That deletion is deliberate and load-bearing — over-
stripping breaks name matching outright — which is exactly why M1 must
parse relational references from the RAW text BEFORE this function, and
must not "fix" the stripping itself.

These tests fail if that stripping changes. That is the point: M1 needs
to be told, mechanically, if it has disturbed the foundation it was
supposed to build beside.
"""

import pytest

from app.llm.advisor_resolver import extract_name_spans


@pytest.mark.parametrize("text,expected", [
    # The canonical failing query from the audit: the possessive and the
    # level keyword are both gone by the time resolution sees this.
    ("Tell me about Waqar Haider's team", ["waqar haider"]),
    ("show adeel dogar's team", ["adeel dogar"]),
    ("who is Waqar Haider's BM", ["waqar haider"]),
    ("who does Kaleem Ullah report to", ["kaleem ullah"]),
    ("who works under Kaleem Ullah", ["kaleem ullah"]),
    ("how is his team doing", []),
    ("tell me about Waqar Haider", ["waqar haider"]),
    ("advisors in Blue Area", ["blue area"]),
])
def test_name_spans_are_what_they_have_always_been(text, expected):
    assert extract_name_spans(text) == expected


def test_org_vocabulary_is_stripped_from_spans():
    """The words that CARRY the relationship are the words removed."""
    for word in ("team", "company", "unit", "head", "manager", "under", "advisor"):
        assert extract_name_spans(f"Waqar Haider {word}") == ["waqar haider"]


def test_possessive_marker_is_removed_not_kept_as_a_token():
    assert extract_name_spans("Waqar Haider's") == ["waqar haider"]


def test_longest_span_first_is_preserved():
    """Ordering matters for resolution: a longer span is more specific
    and must be tried before its prefix."""
    spans = extract_name_spans("adeel mubarik dogar and ali")
    assert spans[0] == "adeel mubarik dogar"


def test_non_name_words_still_survive_as_candidate_spans():
    """A span is NOT a claim that a person was named. "top" is not a
    stopword and comes through as a candidate; the 0.90 PERSON_FLOOR is
    what rejects it downstream, not this function. Any M1 work reading
    spans must not mistake a span for a person.

    Phase 5B narrowed this: "revenue" used to survive too, and the
    original docstring noted that as harmless. It was not entirely — a
    measure word adjacent to a name glued to it ("sana tariq cleared"),
    producing a span that resolves to nobody and losing a side of a
    comparison. Metric words are now dropped, DERIVED from
    metric_aliases so the list cannot drift from the registry.
    """
    assert extract_name_spans("top 5 advisors by revenue") == ["top"]


def test_empty_text_yields_no_spans():
    assert extract_name_spans("") == []
    assert extract_name_spans("who is the manager") == []
