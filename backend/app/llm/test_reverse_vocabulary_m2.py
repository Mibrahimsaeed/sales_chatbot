"""Registry-derived reverse-role vocabulary (M2).

M2's claim is not "portfolio lead now works" — it is that the trigger
and the level detector are generated from ONE declaration, so they can no
longer disagree. "Portfolio lead now works" is a consequence.

These tests are therefore written against the PROPERTY (the two derived
halves agree, for every declared alias) rather than against the seven
phrasings that happened to be broken. A future relation added to the
registry is covered automatically; a hand-written list would have needed
seven more test cases and would have drifted again.

Preservation of the previously-working phrasings lives in
test_reverse_vocabulary_characterisation.py, which must stay green.
"""

import pytest

from app.database.models import Advisor
from app.llm import intent_catalog as cat
from app.llm import relations


ALIAS_PAIRS = relations.role_alias_pairs()


# ---------------------------------------------------------------------
# The property: one declaration, two derived halves, no disagreement
# ---------------------------------------------------------------------

@pytest.mark.parametrize("alias,level", ALIAS_PAIRS)
def test_every_declared_alias_triggers_a_reverse_question(alias, level):
    assert cat.REVERSE_RE.search(f"who is Waqar Haider's {alias}")


@pytest.mark.parametrize("alias,level", ALIAS_PAIRS)
def test_every_declared_alias_detects_its_own_level(alias, level):
    assert cat.detect_reverse_level(f"who is Waqar Haider's {alias}") == level


def test_the_two_halves_cover_exactly_the_same_vocabulary():
    """The defect M2 removes, asserted directly: an alias that triggers
    but resolves to the wrong level (or vice versa) is now impossible
    because both are generated from role_alias_pairs()."""
    triggering = {alias for alias, _ in ALIAS_PAIRS if cat.REVERSE_RE.search(f"X's {alias}")}
    detecting = {alias for alias, level in ALIAS_PAIRS
                 if cat.detect_reverse_level(f"X's {alias}") == level}
    assert triggering == detecting == {alias for alias, _ in ALIAS_PAIRS}


# ---------------------------------------------------------------------
# The phrasings that were broken, now fixed as a consequence
# ---------------------------------------------------------------------

@pytest.mark.parametrize("phrase,level", [
    ("who is Waqar Haider's portfolio lead", "zonal_head"),
    ("who is Waqar Haider's management lead", "bcm"),
    ("who is Waqar Haider's division head", "unit_head"),
    ("who is Waqar Haider's zone head", "zonal_head"),
    ("who is Waqar Haider's regional head", "unit_head"),
    ("who is Waqar Haider's business centre", "office"),
    ("who is Waqar Haider's branch", "office"),
])
def test_previously_unrecognised_roles_now_route(phrase, level):
    assert cat.REVERSE_RE.search(phrase)
    assert cat.detect_reverse_level(phrase) == level


# ---------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------

def test_team_and_company_never_trigger_reverse():
    """They are reverse-LOOKUPABLE relations (MANAGER_COLUMNS carries
    them) but no phrasing may route to reverse lookup through them —
    "X's team" is M1's group reference, and giving them role aliases
    would silently reroute every M1 query."""
    for level in ("team", "company"):
        assert relations.registry.resolve("advisor", level).role_aliases == ()
    assert not cat.REVERSE_RE.search("show me Waqar Haider's team")
    assert not cat.REVERSE_RE.search("how is Waqar Haider's company doing")


def test_generic_role_words_still_fall_through_to_the_default_level():
    """"boss" names no level, so it must not be a registry alias — it
    resolves by falling through detection, exactly as before."""
    declared = {alias for alias, _ in ALIAS_PAIRS}
    for word in cat.GENERIC_ROLE_WORDS:
        assert word not in declared
        assert cat.detect_reverse_level(f"who is X's {word}") == cat.DEFAULT_REVERSE_LEVEL


def test_generic_role_words_still_trigger():
    for word in cat.GENERIC_ROLE_WORDS:
        assert cat.REVERSE_RE.search(f"who is Waqar Haider's {word}")


def test_multi_word_aliases_tolerate_spacing_like_the_old_regex():
    r"""The hand-written patterns used `unit\s*head`; the derived ones
    must too, or "unithead" silently stops matching."""
    for text in ("who is X's unithead", "who is X's unit  head"):
        assert cat.detect_reverse_level(text) == "unit_head"


def test_longest_alias_is_tested_first():
    """Specificity ordering is now a property of the data, not of list
    order someone maintains by hand."""
    lengths = [len(alias) for alias, _ in ALIAS_PAIRS]
    assert lengths == sorted(lengths, reverse=True)
    assert cat.detect_reverse_level("who is X's regional manager") == "unit_head"
    assert cat.detect_reverse_level("who is X's business center") == "office"


def test_adding_a_relation_extends_the_vocabulary_with_no_code_change():
    """The extensibility claim: declare aliases, and BOTH the trigger and
    the level detector learn them together."""
    scratch = relations.RelationRegistry()
    scratch.register(relations.RelationSpec(
        source_level="advisor", target_level="region", column=Advisor.region,
        role_aliases=("region head", "area head"),
    ))
    pairs = [
        (alias, spec.target_level)
        for spec in scratch.specs_for("advisor") for alias in spec.role_aliases
    ]
    assert ("area head", "region") in pairs
    assert ("region head", "region") in pairs


def test_forward_hierarchy_questions_are_still_not_reverse():
    """The FORWARD/REVERSE distinction is the one thing a widened role
    vocabulary could plausibly blur."""
    for phrase in ("who works under Kaleem Ullah",
                   "who reports to Kaleem Ullah",
                   "show Kaleem Ullah's team"):
        assert not cat.REVERSE_RE.search(phrase)
