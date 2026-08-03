"""
Token-aware keyword matching. THE matcher for every keyword table in the
NLU layer.

WHY IT EXISTS. Nine call sites across four modules asked "is this keyword
in the query?" with plain substring containment — `keyword in q`. That
matches inside unrelated words, and the tables it was applied to hold
short, common English fragments:

    "late"    is inside calcuLATEd, reLATEd, transLATE, escaLATE, beLATEd
    "present" is inside rePRESENTative, PRESENTation
    "most"    is inside alMOST, MOSTly
    "least"   is inside "at LEAST"          <- a normal threshold phrase
    "top"     is inside sTOPped, lapTOP, TOPic

Two of those landed on high-weight signals, so the effect was not a
slightly-off score — the query was routed somewhere else and answered
confidently:

  - "How is the answered calls % calculated?" extracted
    attendance_status=Late, which scores 0.98 (W_SPECIFIC_CONSTRAINT) and
    is in _RULE_BASED_ACTIONS, so it won AND returned before the semantic
    parser ran. The user got a list of late advisors.
  - "advisors with at least 80% achievement" reversed the sort direction
    and manufactured `ranking_strong` evidence, so the reply led with the
    LOWEST achievers above the threshold.

WHAT "TOKEN-AWARE" MEANS HERE. A keyword matches only where its edges are
not inside a word. Boundaries are applied per-edge rather than by wrapping
the whole pattern in `\\b`, because `\\b` is defined against word
CHARACTERS: on a phrase like "80%" or "(late)" a blanket `\\b` asserts a
boundary next to punctuation and fails to match text it should. Each edge
gets a boundary only when that edge is itself alphanumeric.

Runs of whitespace inside a phrase relax to `\\s+`, so "not  marked"
matches "not marked". That is the one deliberate widening; it cannot
introduce a false positive because it still requires whitespace to be
present. Notably NOT `\\s*`, which would let "at least" match "atleast"
and would be a behaviour change rather than a bug fix.

This module does no NLU. It answers one question — does this text contain
this phrase as a whole token — so that a new entry in any keyword table
gets the safe behaviour without its author having to know about any of
the above.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Optional

# Matching is case-insensitive, so callers keep passing whatever casing
# they already had. Compiled patterns are cached because these tables are
# scanned on every request and the set of phrases is small and fixed.
_CACHE_SIZE = 2048


@lru_cache(maxsize=_CACHE_SIZE)
def bounded(phrase: str) -> Optional[str]:
    """`phrase` as regex SOURCE, boundary-anchored and whitespace-relaxed.

    The composable half of this module. `contains()` compiles this and
    searches; callers that need the phrase as PART of a larger pattern —
    comparators building "<phrase> <number>" — use this instead, so a
    threshold regex is anchored exactly the way a keyword scan is.

    That sharing is the point. `comparators.threshold_patterns()` used to
    build its own `re.escape(phrase) + r"\\s+" + NUMBER` with no anchor at
    all, so "over" matched inside "turnover": "turnover 500" extracted a
    `> 500` filter that nobody asked for. Same defect class as the
    keyword tables, in the one detector that had its own pattern builder.

    Returns None for an empty phrase.
    """
    phrase = phrase.strip()
    if not phrase:
        return None

    # Escaped per word, rejoined with a flexible separator, so a phrase
    # containing regex metacharacters ("p1+p2", "80%") is literal and
    # "not  marked" still matches "not marked".
    body = r"\s+".join(re.escape(part) for part in phrase.split())

    # Per-EDGE boundaries. `\b` is defined against word characters, so
    # wrapping a phrase like "80%" or "(late)" in it asserts a boundary
    # next to punctuation and fails to match text it should. Each edge
    # gets one only when that edge is itself alphanumeric.
    left = r"(?<!\w)" if phrase[0].isalnum() or phrase[0] == "_" else ""
    right = r"(?!\w)" if phrase[-1].isalnum() or phrase[-1] == "_" else ""
    return f"{left}{body}{right}"


@lru_cache(maxsize=_CACHE_SIZE)
def _pattern(phrase: str) -> Optional[re.Pattern]:
    """The compiled whole-token pattern for `phrase`, or None if the
    phrase is empty (an empty keyword would otherwise match everything —
    the failure mode this module exists to prevent, so it matches
    nothing instead)."""
    source = bounded(phrase)
    if source is None:
        return None
    return re.compile(source, re.IGNORECASE)


def contains(text: str, phrase: str) -> bool:
    """Does `text` contain `phrase` as a whole token?"""
    pattern = _pattern(phrase)
    return bool(pattern and pattern.search(text))


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    """The `any(kw in q for kw in table)` replacement."""
    return any(contains(text, phrase) for phrase in phrases)


def mask(text: str, phrases: Iterable[str]) -> str:
    """`text` with every occurrence of `phrases` blanked to spaces.

    For the case token-awareness cannot solve: a phrase whose meaning
    differs from that of a word genuinely inside it. "at least 80%"
    contains the whole token "least", but it is a comparator, not a
    request for the minimum — so the ranking scan runs against text with
    the comparator phrases masked out.

    Blanked to spaces of equal length rather than removed so nothing
    either side is accidentally joined into a new word ("at least"
    removed from "at leastmost" would create one).
    """
    for phrase in phrases:
        pattern = _pattern(phrase)
        if pattern is not None:
            text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    return text


def first_match(text: str, phrases: Iterable[str]) -> Optional[str]:
    """The first phrase present, in the caller's iteration order — the
    `for kw in table: if kw in q: ...; break` replacement.

    Order is the caller's responsibility and is load-bearing for at least
    one table: ATTENDANCE_STATUS_KEYWORDS lists "not marked" before
    "late" so that "not marked" wins on a query containing both.
    """
    for phrase in phrases:
        if contains(text, phrase):
            return phrase
    return None
