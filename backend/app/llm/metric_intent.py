"""
Did the user NAME a measure, and did it resolve?

WHY THIS EXISTS (audit finding F6). `DEFAULT_RANKING_METRIC` fires
whenever a strong ranking word is present and `resolve_metric` returned
nothing. That test cannot tell two very different situations apart:

  "top 5 advisors"                    named no measure  -> a default is fine
  "which BCM has the highest CR%"     named one, unresolved -> a default is a
                                      confident wrong answer

Both looked identical to the planner, so the second silently became a
revenue leaderboard. The user asked about client registrations and got a
ranking by cleared revenue, correctly formatted, with a header naming a
metric they never mentioned.

The comment defending the default said it is "disclosed, not silent:
the reply header always names the metric it ranked by". That is true and
sufficient for "top 5 advisors" — the user stated no preference, so
naming the choice is a complete account of it. It is NOT sufficient when
the user stated a measure: the header then contradicts the question, and
a reader who asked for CR% does not read "MTD Revenue Cleared" as a
correction of their own words.

WHAT COUNTS AS NAMING A MEASURE. A metric SLOT — the position in the
sentence where a measure goes ("by X", "highest X", "X leaderboard").
Whatever fills the slot is stripped of everything that is demonstrably
not a measure: level words ("advisors", "teams"), grounded entity names,
ranking words, period words, numbers and stopwords. If anything
substantive survives and still does not resolve, the user named
something this system cannot answer.

This is deliberately conservative in one direction: when in doubt it
concludes NO measure was named, which preserves today's default
behaviour. A false "unresolved" would refuse a query that used to work;
a false "not named" only leaves the pre-existing bug in place for that
phrasing. Given the failure being fixed is a wrong ANSWER and the
failure mode of over-triggering is a wrong REFUSAL, that asymmetry is
the right way round.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dataclass_field
from typing import Optional

from app.llm import hierarchy, metric_aliases, token_match
from app.llm.metric_ontology import resolve_metric

# The slot patterns. Each captures the phrase that should name a measure.
#
# "by X" is the explicit form. The ranking-word form covers "highest CR%"
# / "best answered-call rate", and stops at a preposition or a time word
# so the entity and period don't bleed into the captured phrase.
_STOP = r"(?:\s+(?:in|for|under|at|from|within|across|this|last|per|among|between|over)\b|[?.,!]|$)"
_SLOT_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(rf"\bby\s+(?P<slot>.+?){_STOP}", re.I),
    re.compile(rf"\branked\s+on\s+(?P<slot>.+?){_STOP}", re.I),
    re.compile(
        r"\b(?:highest|lowest|best|worst|most|least|top|bottom)\s+"
        rf"(?:\d+\s+)?(?P<slot>.+?){_STOP}",
        re.I,
    ),
    re.compile(r"\b(?P<slot>[\w%\-\s]+?)\s+leaderboard\b", re.I),
)

# Words that can fill a slot without naming a measure. Ranking and period
# vocabulary, plus filler. Level words and entity names are removed
# separately because they are data, not language.
_NOT_A_MEASURE = frozenset({
    "the", "a", "an", "of", "for", "by", "on", "with", "and", "or", "to",
    "is", "are", "has", "have", "had", "does", "do", "did", "was", "were",
    "who", "what", "which", "whose", "whom", "that", "there", "their",
    "me", "my", "our", "us", "you", "your", "it", "its",
    "show", "give", "list", "tell", "find", "get", "see", "display",
    "top", "bottom", "best", "worst", "highest", "lowest", "most", "least",
    "rank", "ranked", "ranking", "leaderboard", "leaderboards", "performing",
    "this", "last", "current", "today", "yesterday", "now", "month", "months",
    "year", "years", "quarter", "quarters", "week", "weeks", "day", "days",
    "date", "mtd", "ytd", "3m", "period", "so", "far", "since",
    "overall", "total", "all", "any", "some", "each", "every", "one",
    "group", "groups", "people", "person", "staff", "member", "members",
})

# Every word that names a level ("advisors", "teams", "unit heads", ...).
# Read from the registry so a level added later is covered.
_LEVEL_WORDS = frozenset(
    word
    for keywords in hierarchy.LEVEL_KEYWORDS.values()
    for keyword in keywords
    for word in keyword.replace("-", " ").split()
)


@dataclass(frozen=True)
class MetricIntent:
    """What the query says about which measure it wants."""

    key: Optional[str]
    """The resolved metric, or None."""

    named_text: Optional[str]
    """The phrase that appears to name a measure, when one could not be
    resolved. Feeds the clarification so it can quote the user's words
    back rather than asking a generic question."""

    reason: Optional[str] = None
    """Why it cannot be answered, when the registry KNOWS the measure but
    cannot compute it ("answered calls %" needs a working-day calendar).
    None for a phrase nothing recognises. The two deserve different
    replies: one can name the missing ingredient and offer the closest
    available measure, the other can only list what exists."""

    keys: list[str] = dataclass_field(default_factory=list)
    """EVERY measure the query named, in the order it named them.

    `key` above stays the primary one and every existing reader keeps
    using it, so single-measure behaviour is untouched. This exists
    because "connects and answered calls" names two and the pipeline had
    nowhere to say so — the second was discarded here, at detection,
    before the planner or the IR could have represented it.

    Holds one entry for a single-measure query, so a caller can read it
    uniformly, and is EMPTY only when nothing resolved.
    """

    @property
    def is_multi(self) -> bool:
        """Did the query name more than one measure?"""
        return len(self.keys) > 1

    @property
    def resolved(self) -> bool:
        return self.key is not None

    @property
    def unresolved(self) -> bool:
        """The user named a measure and this system cannot answer it.
        A default metric here would answer a different question."""
        return self.key is None and self.named_text is not None

    @property
    def may_default(self) -> bool:
        """No measure named at all, so choosing one is filling a gap the
        user left open rather than overriding something they said."""
        return self.key is None and self.named_text is None


def _entity_words(entities: dict) -> frozenset[str]:
    """Words belonging to entities the extractor already grounded. "Blue
    Area" in "top advisors in Blue Area by revenue" is a subject, not a
    measure, and must not look like an unresolvable metric."""
    words: set[str] = set()
    for key in ("advisor_name", *hierarchy.LEVEL_ENTITY_KEYS.values(), *hierarchy.GROUP_LEVELS):
        value = entities.get(key)
        if isinstance(value, str):
            words.update(value.lower().split())
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    words.update(item.lower().split())
    return frozenset(words)


def _residue(slot: str, entities: dict) -> str:
    """What is left of a slot phrase once everything that cannot be a
    measure is removed."""
    entity_words = _entity_words(entities)
    kept = []
    for word in re.split(r"[\s/]+", slot.strip().lower()):
        word = word.strip("\"'()[[].,!?;:")
        if not word or word in _NOT_A_MEASURE or word in _LEVEL_WORDS or word in entity_words:
            continue
        # A bare number is a limit or a threshold, never a measure name.
        if re.fullmatch(r"[\d.,%]+", word):
            continue
        kept.append(word)
    return " ".join(kept)


def _widen(residue: str) -> Optional[str]:
    """Exact -> fuzzy -> semantic, cheapest first, each fail-soft.

    Imported lazily: semantic_retrieval pulls in the embedding client,
    and this module is imported by the planner on every request.
    """
    exact = resolve_metric(residue)
    if exact:
        return exact

    from app.llm.fallback_reasoning import fuzzy_resolve_metric

    fuzzy = fuzzy_resolve_metric(residue)
    if fuzzy:
        return fuzzy

    try:
        from app.llm import semantic_retrieval

        return semantic_retrieval.retrieve_metric(residue)
    except Exception:
        # Widening is a bonus tier; its failure must never be the reason
        # a request errors. Falling through means "unresolved", which is
        # a clarification rather than a wrong answer.
        return None


def _all_named(text: str, primary: str) -> list[str]:
    """Every measure `text` names, with `primary` guaranteed present.

    `primary` is whatever this module already resolved, and it stays
    authoritative: it may come from `entities["metric"]` (a filled
    clarification slot, an ir_patcher carry) or from a fuzzy/embedding
    widening, neither of which the alias scan can see. When the scan and
    the primary disagree about what was named, the scan is the one that
    is wrong about this query, so the list collapses to the primary
    alone — a single-measure answer, exactly as before.

    That guard is what keeps this additive: a caller reading `keys` can
    never be routed somewhere `key` would not have gone.
    """
    named = [match.metric for match in metric_aliases.resolve_all(text) if match.metric]
    if primary not in named:
        return [primary]
    return named


def detect(text: str, entities: dict) -> MetricIntent:
    """The measure this query asks for, and whether asking was possible.

    `entities["metric"]` is honoured first — a caller that already
    resolved one (ir_patcher, a filled clarification slot) has better
    information than this module can recover from the raw string.
    """
    key = entities.get("metric") or resolve_metric(text)
    if key:
        return MetricIntent(key=key, named_text=None, keys=_all_named(text, key))

    # A measure the registry KNOWS and cannot compute. Checked before any
    # widening: fuzzy matching would otherwise resolve "cr %" back to the
    # client-registration COUNT sitting inside it, which is exactly the
    # count-for-rate substitution the registry exists to prevent.
    declared = metric_aliases.resolve(text)
    if declared is not None and not declared.available:
        return MetricIntent(
            key=None,
            named_text=declared.phrase,
            reason=metric_aliases.explain(declared),
        )

    for pattern in _SLOT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        residue = _residue(match.group("slot"), entities)
        if not residue:
            continue
        # Before concluding "cannot answer", run the SAME widening tiers
        # the pipeline already had: exact match on the residue alone (the
        # slot may carry words that defeated whole-string matching), then
        # fuzzy synonym matching for typos ("revnue"), then embedding
        # retrieval for paraphrases with no lexical overlap.
        #
        # These used to sit downstream in semantic_parser._rule_based_ir,
        # reachable only when plan.action was "unresolved". Refusing here
        # without them would turn a typo into a refusal — trading one
        # wrong outcome for another.
        widened = _widen(residue)
        if widened:
            # A widened match came from a typo or a paraphrase, which the
            # alias scan by definition did not see — so _all_named
            # collapses to this one key. Passed through it anyway so
            # `keys` is never empty for a resolved intent.
            return MetricIntent(key=widened, named_text=None,
                                keys=_all_named(text, widened))
        return MetricIntent(key=None, named_text=residue)

    return MetricIntent(key=None, named_text=None)


def clarification(named_text: str, reason: str | None = None) -> str:
    """The reply for a measure this system cannot answer.

    Quotes the user's own words and lists what IS available, so the next
    message can be an answer rather than another guess. Built here so the
    planner and the pipeline cannot word it differently.

    When the registry knows WHY (a declared-but-uncomputable rate), say
    that instead — "I need a working-day calendar, here is the count" is
    actionable where a list of every metric is not.
    """
    from app.llm.metric_ontology import describe_available_metrics

    if reason:
        return reason

    return (
        f"I don't have a metric for \"{named_text}\" — I don't want to rank by "
        f"something else and pass it off as an answer. I can rank by: "
        f"{describe_available_metrics()}."
    )
