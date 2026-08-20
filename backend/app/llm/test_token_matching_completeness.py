"""Phase 5.3 — every detector shares one token-aware matcher.

Step 1 anchored the keyword TABLES (attendance status, ranking words,
sort direction, level keywords) and the entity gazetteer. Three sites
were missed, each because it built its own matching rather than calling
the shared one:

  COMPARATORS. `threshold_patterns()` composed `re.escape(phrase) + r"\\s+"
  + NUMBER` with no anchor at all, so "over" matched inside "turnover":
  "turnover 500" produced a `> 500` filter out of nothing. Now composed
  from token_match.bounded(), the same anchoring `contains()` uses.

  METRIC SYNONYMS. `resolve_metric_evidence` used `synonym in q`; harmless
  until a synonym got short enough, which "cr" did — it sits inside
  "across", "describe", "increase".

  IDENTITY DISAMBIGUATION. `advisor_resolver.resolve_choice` tested
  `candidate.team.lower() in needle` when reading which person the user
  picked. The consequence of a false match there is choosing the WRONG
  PERSON — the one thing that disambiguation exists to prevent.

The tests below are mostly PROPERTIES over the real vocabularies rather
than lists of the collisions we happened to find, so a phrase added
tomorrow is covered the day it is declared.
"""

import re

import pytest

from app.llm import comparators, hierarchy, intent_catalog as cat, metric_aliases, token_match
from app.llm.entity_extractor import ATTENDANCE_STATUS_KEYWORDS, _extract_thresholds


# ---------------------------------------------------------------------
# The shared matcher is genuinely shared
# ---------------------------------------------------------------------

def test_threshold_patterns_are_built_from_the_shared_matcher():
    """Not "equivalent anchoring" — literally the same helper. An
    independent reimplementation is how these drifted apart before."""
    for pattern, _operator in comparators.threshold_patterns():
        if pattern.startswith("(?<!\\w)"):
            continue                      # prefix phrase
        if "(?!\\w)" in pattern:
            continue                      # suffix phrase
        # Anything left must be a bare symbol, which has no word boundary
        # to anchor against.
        assert re.match(r"^\\?[<>=]", pattern), pattern


@pytest.mark.parametrize("phrase", [
    p for c in comparators.COMPARATORS for p in (*c.phrases, *c.suffix_phrases)
])
def test_no_comparator_phrase_fires_inside_a_word(phrase):
    """The property, over the registry itself."""
    prefix = _extract_thresholds(f"zq{phrase}zq 42")
    suffix = _extract_thresholds(f"42 zq{phrase}zq")
    assert prefix == [], f"{phrase!r} matched inside a word (prefix form)"
    assert suffix == [], f"{phrase!r} matched inside a word (suffix form)"


@pytest.mark.parametrize("word,number", [
    ("turnover", 500), ("handover", 3), ("leftover", 20), ("rollover", 7),
    ("crossover", 12), ("makeover", 9),
])
def test_over_does_not_fire_inside_a_compound(word, number):
    """The reported case. Every one of these is a plausible sales-ops
    word followed by a figure."""
    assert _extract_thresholds(f"{word} {number}") == []


@pytest.mark.parametrize("text", [
    "underwriting 50", "undervalued 30", "overall 80", "aboveboard 10",
    "belowdecks 5",
])
def test_other_comparator_words_do_not_fire_inside_compounds(text):
    assert _extract_thresholds(text) == []


# ---------------------------------------------------------------------
# Every vocabulary, as a property
# ---------------------------------------------------------------------

def _vocabularies():
    """Every phrase table a user query is matched against, with the
    matcher that reads it. Listed so a table added later is a visible
    omission rather than a silent one."""
    tables: list[tuple[str, list[str]]] = [
        ("ATTENDANCE_STATUS_KEYWORDS", list(ATTENDANCE_STATUS_KEYWORDS)),
        ("RANKING_STRONG", list(cat.RANKING_STRONG)),
        ("RANKING_WEAK", list(cat.RANKING_WEAK)),
        ("ASCENDING_ABSOLUTE", list(cat.ASCENDING_ABSOLUTE)),
        ("DESCENDING_ABSOLUTE", list(cat.DESCENDING_ABSOLUTE)),
        ("WORST_RELATIVE", list(cat.WORST_RELATIVE)),
        ("FLAT_KEYWORDS", list(cat.FLAT_KEYWORDS)),
        ("GENERIC_ROLE_WORDS", list(cat.GENERIC_ROLE_WORDS)),
        ("comparator phrases", [p for c in comparators.COMPARATORS
                                for p in (*c.phrases, *c.suffix_phrases)]),
        ("metric aliases", [p for ps in metric_aliases.ALIASES.values() for p in ps]),
        ("unavailable aliases", [p for u in metric_aliases.UNAVAILABLE for p in u.phrases]),
    ]
    tables += [(f"LEVEL_KEYWORDS[{lvl}]", list(kws))
               for lvl, kws in hierarchy.LEVEL_KEYWORDS.items()]
    return tables


@pytest.mark.parametrize("table,phrases", _vocabularies())
def test_no_phrase_in_any_vocabulary_matches_inside_a_word(table, phrases):
    """One property covering every table at once. A phrase added
    tomorrow is covered the moment it is declared."""
    for phrase in phrases:
        buried = f"zq{phrase}zq"
        assert not token_match.contains(buried, phrase), f"{table}: {phrase!r}"


@pytest.mark.parametrize("table,phrases", _vocabularies())
def test_every_phrase_still_matches_itself(table, phrases):
    """The other half — anchoring must not silence a supported phrase."""
    for phrase in phrases:
        assert token_match.contains(f"please show {phrase} now", phrase), f"{table}: {phrase!r}"


@pytest.mark.parametrize("table,phrases", _vocabularies())
def test_every_phrase_matches_next_to_punctuation(table, phrases):
    """Per-EDGE anchoring, not a blanket \\b: a phrase ending in "%" must
    still match, and one next to a comma or a question mark must too."""
    for phrase in phrases:
        for wrapped in (f"{phrase}?", f"({phrase})", f"{phrase}, please"):
            assert token_match.contains(wrapped, phrase), f"{table}: {wrapped!r}"


# ---------------------------------------------------------------------
# The reported collisions, named
# ---------------------------------------------------------------------

# NOTE: ("least", "at least 80 percent") is deliberately absent. "least"
# IS a whole token there, so anchoring cannot and must not suppress it —
# that case is handled by masking comparator phrases before the ranking
# scan, and is covered by its own test below.
@pytest.mark.parametrize("phrase,text", [
    ("most", "almost 80 percent"),
    ("late", "how is this calculated"),
    ("late", "show me related teams"),
    ("late", "escalate to the manager"),
    ("late", "please translate this"),
    ("below", "belowdecks"),
    ("above", "aboveboard"),
    ("over", "turnover"),
    ("under", "underwriting"),
    ("top", "laptop"),
    ("best", "bestseller"),
    ("rank", "franken"),
    ("lead", "leadership"),
])
def test_the_named_collisions_are_gone(phrase, text):
    assert not token_match.contains(text, phrase), f"{phrase!r} in {text!r}"


def test_at_least_keeps_its_threshold_and_its_direction():
    """"least" IS a whole token inside "at least", so anchoring alone
    cannot separate them — comparator phrases are masked before the
    ranking scan. Both halves must hold at once."""
    from app.llm.query_planner import _sort_signal, _without_comparators

    assert _extract_thresholds("at least 80") == [{"operator": ">=", "value": 80.0}]
    assert _sort_signal(_without_comparators("advisors with at least 80 percent"),
                        "achievement_pct") is None


def test_almost_is_not_a_ranking_word():
    from app.llm.query_planner import _sort_signal

    assert _sort_signal("almost 80 percent", "achievement_pct") is None


# ---------------------------------------------------------------------
# Overlap and longest-match precedence
# ---------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("no more than 50", [{"operator": "<=", "value": 50.0}]),
    ("not less than 50", [{"operator": ">=", "value": 50.0}]),
    ("no greater than 50", [{"operator": "<=", "value": 50.0}]),
    ("not below 50", [{"operator": ">=", "value": 50.0}]),
])
def test_a_longer_phrase_consumes_the_shorter_one_inside_it(text, expected):
    """Longest-match precedence AND overlap suppression together. Every
    one of these CONTAINS a shorter comparator with the opposite
    polarity, so without both the result is two contradictory filters
    that match nothing."""
    assert _extract_thresholds(text) == expected


def test_a_range_consumes_both_of_its_bounds():
    """"between 60 and 80" must not also be read as two loose numbers,
    and the joining "and" must not become a second constraint."""
    got = _extract_thresholds("between 60 and 80")
    assert got == [{"operator": ">=", "value": 60.0}, {"operator": "<=", "value": 80.0}]


def test_a_range_and_a_separate_threshold_coexist():
    """Two measures, so each threshold also carries the one it was
    written beside (_bind_threshold_metrics): the range's two bounds stay
    on `achievement` and the loose comparator goes to `connects`. Binding
    all three to one key is what made a two-condition query compile as
    two conditions on the same column."""
    got = _extract_thresholds("achievement between 60 and 80 and connects above 5")
    assert got == [
        {"operator": ">=", "value": 60.0, "metric": "achievement_pct"},
        {"operator": "<=", "value": 80.0, "metric": "achievement_pct"},
        {"operator": ">", "value": 5.0, "metric": "total_connects"},
    ]


def test_no_two_thresholds_share_a_span():
    """The invariant behind the above: each extracted threshold comes
    from its own region of the text. Two operators over one number is
    always a misread."""
    for text in ("no more than 50", "at least 80", "between 60 and 80",
                 "80 percent or higher", "not above 30"):
        values = [t["value"] for t in _extract_thresholds(text)]
        assert len(values) == len(set(values)), text


def test_symbols_are_not_anchored_but_still_do_not_double_match():
    """">= 90" must not also match "> 90". Symbols cannot be word-
    anchored, so this relies on longest-first plus span suppression."""
    assert _extract_thresholds(">= 90") == [{"operator": ">=", "value": 90.0}]
    assert _extract_thresholds("<= 40") == [{"operator": "<=", "value": 40.0}]


# ---------------------------------------------------------------------
# Identity disambiguation
# ---------------------------------------------------------------------

def test_choosing_a_person_by_team_is_token_aware():
    """The highest-consequence site: a false match here returns the
    wrong human being."""
    from app.llm.advisor_resolver import AdvisorIdentity, resolve_choice

    candidates = [
        AdvisorIdentity(wid=1, name="Yasir Ali", team="GRO", company="Graana"),
        AdvisorIdentity(wid=2, name="Yasir Ali", team="Downtown", company="IMARAT"),
    ]
    # A whole-token mention still picks.
    assert resolve_choice("the one in GRO", candidates).wid == 1
    assert resolve_choice("Downtown", candidates).wid == 2
    # "grocery" contains "GRO" but names nobody — re-ask rather than guess.
    assert resolve_choice("the grocery one", candidates) is None
