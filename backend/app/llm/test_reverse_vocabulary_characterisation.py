"""Characterisation of the reverse-role vocabulary (written BEFORE M2).

NOT a specification — a recording of what `REVERSE_RE` and
`detect_reverse_level()` do TODAY, captured before M2 replaces two
hand-written role lists with one derived from the M0 relation registry.

Reverse lookup ("who is X's BM") is the only advisor->X capability that
already works in production. The design's risk R3 is that regenerating
its vocabulary quietly drops a phrasing. Every phrase below is one this
file asserts must keep behaving identically; the new vocabulary must be a
strict SUPERSET, so anything here that stops working is a regression, and
these tests are what will say so.

The `NOT_REVERSE_TODAY` cases are equally deliberate: they record
phrasings the current implementation does NOT recognise. M2 is expected
to start recognising some of them (that is the point), so those are
listed separately and re-asserted from the other direction afterwards.
"""

import pytest

from app.llm import intent_catalog as cat


# ---------------------------------------------------------------------
# Trigger: is this a reverse question at all?
# ---------------------------------------------------------------------

REVERSE_PHRASES = [
    # possessive + role
    "who is Waqar Haider's bm",
    "who is Waqar Haider's zm",
    "who is Waqar Haider's rm",
    "who is Waqar Haider's manager",
    "who is Waqar Haider's boss",
    "who is Waqar Haider's supervisor",
    "who is Waqar Haider's lead",
    "who is Waqar Haider's unit head",
    "who is Waqar Haider's zonal head",
    "who is Waqar Haider's regional manager",
    "who is Waqar Haider's region manager",
    "who is Waqar Haider's business center",
    # bare possessive, no "who is"
    "Waqar Haider's bm",
    "Waqar Haider's unit head",
    # pronoun forms
    "his bm",
    "her manager",
    "their zonal head",
    "its business center",
    # report-to forms
    "who does Waqar Haider report to",
    "who is Waqar Haider reporting to",
    "who is Waqar Haider under",
    "who is Waqar Haider managed by",
    # role-of forms
    "manager of Waqar Haider",
    "bm of Waqar Haider",
    "unit head for Waqar Haider",
]


@pytest.mark.parametrize("phrase", REVERSE_PHRASES)
def test_phrase_is_recognised_as_reverse(phrase):
    assert cat.REVERSE_RE.search(phrase), f"regression: {phrase!r} stopped being reverse"


NOT_REVERSE = [
    "tell me about Waqar Haider",
    "how is Blue Area doing",
    "top 5 advisors by revenue",
    "who works under Kaleem Ullah",     # FORWARD — the people under someone
    "who reports to Kaleem Ullah",      # FORWARD
    "show me Waqar Haider's team",      # a group reference, not a role
]


@pytest.mark.parametrize("phrase", NOT_REVERSE)
def test_phrase_is_not_reverse(phrase):
    assert not cat.REVERSE_RE.search(phrase), f"{phrase!r} must not be a reverse question"


# ---------------------------------------------------------------------
# Level detection: WHICH manager is being asked for?
# ---------------------------------------------------------------------

LEVEL_CASES = [
    ("who is X's zm", "zm"),
    ("who is X's zonal head", "zonal_head"),
    ("who is X's zone head", "zonal_head"),
    ("who is X's rm", "unit_head"),
    ("who is X's regional manager", "unit_head"),
    ("who is X's region manager", "unit_head"),
    ("who is X's regional head", "unit_head"),
    ("who is X's business center", "office"),
    ("who is X's business centre", "office"),
    ("who is X's branch", "office"),
    ("who is X's portfolio lead", "zonal_head"),
    ("who is X's management lead", "bcm"),
    ("who is X's bm", "bm"),
    ("who is X's unit head", "unit_head"),
    ("who is X's division head", "unit_head"),
    # unspecific role words fall through to the default level
    ("who is X's manager", "unit_head"),
    ("who is X's boss", "unit_head"),
    ("who does X report to", "unit_head"),
]


@pytest.mark.parametrize("phrase,level", LEVEL_CASES)
def test_reverse_level_detection(phrase, level):
    assert cat.detect_reverse_level(phrase) == level


def test_default_reverse_level_is_unit_head():
    assert cat.DEFAULT_REVERSE_LEVEL == "unit_head"


# ---------------------------------------------------------------------
# RETIRED (M2 landed).
#
# This file also recorded the pre-M2 state of seven phrasings that
# `detect_reverse_level` already understood while REVERSE_RE did not —
# "portfolio lead", "management lead", "division head", "zone head",
# "regional head", "business centre", "branch". Two hand-written lists
# that had to agree, with nothing forcing them to.
#
# Those assertions were written to FAIL once the vocabulary was unified,
# and they did. They now live in test_reverse_vocabulary_m2.py, asserted
# from the other direction: the same phrases must be recognised. Keeping
# both would mean asserting a bug is still present.
# ---------------------------------------------------------------------
