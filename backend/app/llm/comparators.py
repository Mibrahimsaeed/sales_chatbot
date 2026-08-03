"""
Comparator vocabulary — the single source of truth for "above 80",
"at least 5", ">= 90".

WHY THIS MODULE EXISTS. The vocabulary used to live in two places that
had drifted apart: entity_extractor._THRESHOLD_PATTERNS (deterministic
regexes) and entity_extractor._COMPARATOR_EXEMPLARS (embedding
exemplars). "above" appeared only in the semantic list, so it resolved
only when an embedding call was available while its mirror image
"below" resolved deterministically — the same query worked or silently
dropped its constraint depending on whether the LLM provider was
reachable. Two lists that must agree, with nothing forcing them to.

One declaration per operator now carries BOTH: the phrases that parse
deterministically and the paraphrases that fall back to semantic
matching. Adding a comparator is one entry here and nothing else.

The number pattern is written once. It used to be repeated verbatim in
twelve regexes, which is how "above" came to be missing from one of
them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A number, optionally followed by a percent marker in any of its written
# forms: "80", "80%", "80 percent", "12.5", "80 pct".
#
# The word forms matter for SUFFIX comparators. "above 80 percent" parses
# with or without them because the number ends the pattern — but in "80
# percent or higher" the word sits BETWEEN the number and the phrase, so
# a pattern that cannot consume it never matches at all.
NUMBER = r"(\d+(?:\.\d+)?)\s*(?:%|percent|pct)?"


@dataclass(frozen=True)
class Comparator:
    """One SQL operator and every way a user says it.

    - `phrases` parse deterministically and are the contract: they work
      with no LLM, no network, no quota.
    - `symbols` are the literal forms ("&gt;=") — matched before their
      shorter cousins so ">= 5" never reads as "> 5".
    - `exemplars` are paraphrases handled by embedding similarity when
      no phrase matched. They are a WIDENING, never a replacement, and
      deliberately do not repeat anything in `phrases`.
    """

    operator: str
    phrases: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    # Phrases that follow the number instead of preceding it: "80 or
    # higher". Same contract as `phrases` (deterministic, no LLM), just a
    # different word order, so they need their own pattern shape rather
    # than being crammed into the prefix list.
    suffix_phrases: tuple[str, ...] = ()
    exemplars: tuple[str, ...] = field(default_factory=tuple)


# Declaration order is not significant — pattern ordering is derived
# below, longest-first, so a longer phrase always wins over a shorter one
# it contains ("no more than" before "more than").
COMPARATORS: tuple[Comparator, ...] = (
    Comparator(
        operator=">",
        phrases=("more than", "greater than", "over", "above"),
        symbols=(">",),
        exemplars=("north of", "in excess of"),
    ),
    Comparator(
        operator=">=",
        # POLARITY FIX. "no less than" and friends used to sit in
        # `exemplars`, which only resolve through an embedding call — so
        # with no LLM reachable they never matched, and the shorter
        # "less than" inside them did: "no less than 50" parsed as
        # "< 50", the exact complement of what was asked. A negation must
        # parse deterministically, because getting it wrong doesn't
        # return fewer rows, it returns the WRONG rows.
        phrases=("at least", "no less than", "not less than",
                 "no lower than", "not lower than", "not below", "not under"),
        symbols=(">=",),
        suffix_phrases=("or higher", "or more", "or above", "or greater", "or over"),
        exemplars=("upwards of", "at minimum"),
    ),
    Comparator(
        operator="<",
        phrases=("less than", "below", "under"),
        symbols=("<",),
        exemplars=("south of", "shy of", "a bit under", "just below"),
    ),
    Comparator(
        operator="<=",
        # Same fix, mirror image: "no more than 50" parsed as "> 50".
        phrases=("at most", "no more than", "not more than",
                 "no greater than", "not greater than", "not above", "not over"),
        symbols=("<=",),
        suffix_phrases=("or lower", "or less", "or below", "or under", "or fewer"),
        exemplars=("at maximum", "capped at"),
    ),
)

# "between 60 and 80" — a RANGE, which is two bounds rather than one
# comparator, so it cannot be a Comparator entry.
#
# It compiles to two AND-combined filters (>= low, <= high) rather than a
# new "between" operator. QueryIR.filters is already AND-combined by
# design, so a range needs no new IR shape, no new SQL operator, and no
# change to the LLM's schema — and "between 60 and 80" and "above 60 and
# below 80" produce the same thing, which is what a reader expects.
#
# Inclusive on both ends: "teams between 60 and 80" is normally read as
# including a team on exactly 60.
RANGE_PATTERN = rf"\bbetween\s+{NUMBER}\s+and\s+{NUMBER}"
RANGE_OPERATORS: tuple[str, str] = (">=", "<=")


def threshold_patterns() -> list[tuple[str, str]]:
    """(regex, operator) pairs for deterministic extraction.

    Ordered longest-token-first so that ">=" is tried before ">" and
    "no more than" before "more than". Word phrases come before bare
    symbols: a symbol can appear inside prose, and preferring the
    explicit wording keeps the match anchored where the user wrote it.

    Phrases are anchored with token_match.bounded() — the SAME matcher
    the keyword tables use. Built with a bare re.escape() and no anchor,
    "over" matched inside "turnover" and "turnover 500" produced a
    `> 500` filter out of nothing. Symbols are deliberately not anchored:
    ">" has no word boundary to speak of.
    """
    from app.llm import token_match

    worded: list[tuple[str, str]] = []
    symbolic: list[tuple[str, str]] = []
    for comparator in COMPARATORS:
        for phrase in comparator.phrases:
            worded.append((rf"{token_match.bounded(phrase)}\s+{NUMBER}", comparator.operator))
        for phrase in comparator.suffix_phrases:
            # The number comes FIRST here, so the capture group is still
            # group(1) and callers need no special case.
            worded.append((rf"{NUMBER}\s+{token_match.bounded(phrase)}", comparator.operator))
        for symbol in comparator.symbols:
            symbolic.append((rf"{re.escape(symbol)}\s*{NUMBER}", comparator.operator))
    worded.sort(key=lambda pair: -len(pair[0]))
    symbolic.sort(key=lambda pair: -len(pair[0]))
    return worded + symbolic


def range_pattern() -> tuple[str, tuple[str, str]]:
    """(regex, (low_operator, high_operator)) for "between X and Y".

    Two capture groups, mapping to two AND-combined filters. Matched
    BEFORE the single-comparator patterns so the "and" between the bounds
    is never read as two separate constraints.
    """
    return RANGE_PATTERN, RANGE_OPERATORS


def semantic_exemplars() -> list[tuple[str, str]]:
    """(paraphrase, operator) pairs for the embedding fallback."""
    return [
        (exemplar, comparator.operator)
        for comparator in COMPARATORS
        for exemplar in comparator.exemplars
    ]


def operators() -> tuple[str, ...]:
    return tuple(comparator.operator for comparator in COMPARATORS)


def phrases() -> tuple[str, ...]:
    """Every comparator phrase, longest first.

    Read by query_planner to MASK comparator phrases before it scans for
    ranking vocabulary. Two of these — "at least" and "at most" — contain
    a sort-direction word as a genuine whole token, so token-aware
    matching alone cannot separate them: in "at least 80%" the word
    "least" really is there, it just isn't asking for the minimum.

    Deriving the mask from this registry rather than hardcoding the two
    known collisions means a future comparator phrase ("no more than",
    "at best") is shadowed the day it is declared, instead of silently
    reintroducing the bug. Longest first so a phrase that contains a
    shorter one is masked whole.
    """
    return tuple(sorted(
        (phrase for comparator in COMPARATORS for phrase in comparator.phrases),
        key=len,
        reverse=True,
    ))
