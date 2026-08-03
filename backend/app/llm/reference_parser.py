"""
Relational reference parsing (M1 of the Relationship Inference Engine).

WHAT THIS IS. A pure text parser that answers one question: does this
query refer to an entity THROUGH another entity — "Waqar Haider's team"
— and if so, which level is being referred to? It resolves nothing,
reads no database, and has no idea what intent the query will become.

WHY IT RUNS ON RAW TEXT. The architectural audit's debt item D1: by the
time identity resolution has run, the relationship is gone.
`advisor_resolver.extract_name_spans` deliberately deletes "team", "s",
"his", "under", "manager" and the rest of the org vocabulary, because
over-stripping breaks name matching — a trade-off that is correct for
its purpose and load-bearing. So this parser must see the text BEFORE
that happens. `test_name_span_characterisation.py` locks the stripping
behaviour precisely so this module can be built beside it rather than
inside it.

VOCABULARY IS DERIVED, NOT WRITTEN. The levels this recognises come from
the M0 relation declarations intersected with `hierarchy.LEVEL_KEYWORDS`.
Declaring a new relationship therefore makes its phrasings parseable with
no edit here — the extensibility claim of the design, made structural
rather than promised.

WHAT IT DELIBERATELY DOES NOT MATCH:

- A bare level word with no possessive ("team performance", "company
  revenue"). Those name a topic, not a relationship.
- Reverse-manager questions are not special-cased here; they are excluded
  downstream by which levels are ENABLED (M1 enables team/company only,
  and no reverse role word is a team or a company). Keeping that policy
  out of the parser is deliberate — the parser reports what the text
  says, the caller decides what to act on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.llm import hierarchy, relations

# A source span is at most this many tokens, mirroring
# advisor_resolver._MAX_SPAN_TOKENS — the same names are being described.
_MAX_SOURCE_TOKENS = 5

_TOKEN_RE = re.compile(r"[A-Za-z0-9'\-\.]+")


# How the source of a reference is identified.
#   NAMED   — the text names it: "Waqar Haider's team"      (M1)
#   PRONOUN — the text points at an earlier subject: "his team" (M4)
# The kinds are separate because they resolve from different places, and
# a caller that can only satisfy one of them must be able to ignore the
# other rather than mis-resolve it.
NAMED = "named"
PRONOUN = "pronoun"

# Personal pronouns that stand in for a person already under discussion.
# Matches nlu_pipeline._PERSON_FOLLOWUP_RE's vocabulary — the same
# linguistic fact, and they must not drift apart.
_PRONOUN_RE = re.compile(
    r"\b(he|him|his|she|her|hers|they|them|their|theirs|its|"
    r"this person|that person)\b",
    re.I,
)


@dataclass(frozen=True)
class ReferenceRequest:
    """A reference to an entity reached THROUGH another entity.

    `source_span` is the text naming the entity being referred FROM. For
    a NAMED reference it is informational in M1: the caller resolves the
    source through the identity resolution that already ran over the
    whole query. It is captured because compound references ("X's team
    vs Y's team", milestone M6) need per-reference sources, and a parser
    that discarded them would have to be rewritten rather than extended.

    For a PRONOUN reference it is empty by construction — the source is
    not in this message at all, and finding it is the caller's job.
    """

    source_span: str
    target_level: str
    matched_text: str
    kind: str = NAMED


def _keyword_to_level() -> dict[str, str]:
    """Level keywords for declared relation targets, longest first.

    Longest-first matters: "business center" must be tried before
    "center", or the shorter keyword claims the phrase first.
    """
    mapping: dict[str, str] = {}
    for target in relations.registry.targets_for("advisor"):
        for keyword in hierarchy.LEVEL_KEYWORDS.get(target, []):
            mapping[keyword.lower()] = target
    return dict(sorted(mapping.items(), key=lambda kv: -len(kv[0])))


def _pattern() -> re.Pattern:
    keywords = "|".join(re.escape(k) for k in _keyword_to_level())
    # "'s" or a trailing "s'" possessive, optionally "the", then a level
    # keyword. \b at the end stops "team" matching inside "teamwork".
    return re.compile(rf"(?:'s|s')\s+(?:the\s+)?({keywords})\b", re.I)


def _source_span(text: str, end: int, floor: int = 0) -> str:
    """The up-to-5 word tokens preceding the possessive, never reaching
    back past `floor`.

    `floor` is the end of the PREVIOUS reference's match, and it is what
    keeps two references independent. Without it, the second source span
    in "compare Waqar Haider's team with Sana Tariq's team" scoops up the
    first reference's tail — "Haider's team with Sana Tariq" — and the
    two references stop describing two different people. One reference's
    source cannot lie inside another's; bounding the search is how that
    is expressed rather than assumed.
    """
    tokens = _TOKEN_RE.findall(text[floor:end])
    return " ".join(tokens[-_MAX_SOURCE_TOKENS:]) if tokens else ""


def parse(text: str) -> list[ReferenceRequest]:
    """Every NAMED relational reference in `text`, in order of appearance.

    Pure and total: any string in, a (possibly empty) list out. Pronoun
    references are deliberately NOT included here — see parse_pronoun().
    Keeping them out of `parse()` means every existing caller keeps
    exactly the results it had before cross-turn resolution existed.
    """
    if not text:
        return []

    keyword_map = _keyword_to_level()
    found: list[ReferenceRequest] = []
    previous_end = 0
    for match in _pattern().finditer(text):
        level = keyword_map.get(match.group(1).lower())
        if level is None:
            continue
        # The possessive marker sits at the start of the match; the source
        # is whatever preceded it, back to where the last reference ended.
        found.append(
            ReferenceRequest(
                source_span=_source_span(text, match.start(), floor=previous_end),
                target_level=level,
                matched_text=match.group(0).strip(),
                kind=NAMED,
            )
        )
        previous_end = match.end()
    return found


def has_pronoun(text: str) -> bool:
    return bool(_PRONOUN_RE.search(text or ""))


def _mentions_role(text: str, level: str) -> bool:
    """Does the text name this relation as somebody's ROLE ("unit head")
    rather than as a group ("unit")? Read from the M2 role_aliases
    declarations, with the same \\s* spacing tolerance the reverse-role
    patterns use."""
    spec = relations.registry.resolve("advisor", level)
    if spec is None:
        return False
    for alias in spec.role_aliases:
        pattern = r"\s*".join(re.escape(word) for word in alias.split())
        if re.search(rf"\b{pattern}\b", text, re.I):
            return True
    return False


def parse_pronoun(text: str) -> list[ReferenceRequest]:
    """Levels referred to through a pronoun — "how is HIS TEAM doing",
    "what company does HE work for".

    Detection is CO-OCCURRENCE (a personal pronoun anywhere in the
    message plus a level keyword) rather than adjacency, because English
    separates them freely: "his team" is adjacent, "what company does he
    work for" is not, and both mean the same thing. Requiring adjacency
    would have handled the first and quietly missed the second, which is
    the sort of gap that becomes a hand-written special case later.

    The looseness is safe only because the CALLER gates it: cross-turn
    resolution fires just when the message names no advisor of its own
    and memory holds one. This function reports what the text could mean;
    it never decides that it does.

    Levels are deduplicated and returned in declaration order — with no
    possessive to anchor them, "his team and company" has no meaningful
    per-reference ordering to preserve.

    A keyword that is also that relation's ROLE ALIAS is skipped: "who is
    his unit head" asks WHO the unit head is — a reverse-role question
    the pipeline already answers — whereas "how is his unit doing" asks
    about the group. The distinction is read from the M2 role_aliases
    declarations rather than hand-listed, so it stays correct as
    relations are added, and it is the same role/group split M3 recorded
    for "business center" versus "centre".
    """
    if not text or not has_pronoun(text):
        return []

    found: list[ReferenceRequest] = []
    seen: set[str] = set()
    for keyword, level in _keyword_to_level().items():
        if level in seen:
            continue
        # A role phrasing anywhere in the message disqualifies the WHOLE
        # level, not just the matching keyword. "who is his unit head"
        # also contains the bare group word "unit", so excluding only the
        # alias would let the shorter keyword bind the level anyway and
        # turn a reverse-role question into a group breakdown. Marking the
        # level `seen` is what stops that.
        if _mentions_role(text, level):
            seen.add(level)
            continue
        if re.search(rf"\b{re.escape(keyword)}\b", text, re.I):
            seen.add(level)
            found.append(
                ReferenceRequest(
                    source_span="",
                    target_level=level,
                    matched_text=keyword,
                    kind=PRONOUN,
                )
            )
    return found


def references_to(text: str, levels) -> list[ReferenceRequest]:
    """`parse()` narrowed to the levels a caller is willing to act on."""
    allowed = set(levels or ())
    return [r for r in parse(text) if r.target_level in allowed]
