"""THE owner of "which intent wins".

Phase 7. The Phase 6 audit found that four separate defects were one
architectural fact: **a grounded subject is not evidence in intent
selection.** Entity resolution identified the subject correctly, the
scorers ignored it, and any competing signal discarded it uncontested.

    "Top revenue for Omar Farooq"   -> a leaderboard of everyone,
                                       headlined with a DIFFERENT advisor
    "Blue Area and Downtown revenue" -> Blue Area only; Downtown dropped
    "advisors in Blue Area by revenue" -> an alphabetical roster
    "Blue Area revenue"             -> leaderboard (correct answer, but
                                       only because two downstream
                                       components compensated)

WHAT WAS WRONG WITH THE OLD DESIGN. Scorers did two jobs: they proposed
an intent AND they suppressed rival intents. `_score_advisor_metric`
returned None whenever a ranking word, a comparison phrase, a reverse
role or a relation appeared — so "top revenue for Omar Farooq" produced
exactly one candidate, and the ranking was never a contest. A scorer that
declines cannot lose a comparison, and a candidate that is never proposed
cannot be explained.

THE SPLIT. Scorers now only PROPOSE, with structured `Evidence` saying
why. This module RANKS. The two jobs are separate and the second one is
here, once, in a table that reads as English:

  comparison     >  two or more grounded subjects with a measure
  advisor_metric >  one named person with a measure
  group_metric   >  one named group with a measure
  leaderboard    >  a measure with no named subject, or a ranking over
                    a group's members
  roster         >  "who is in X", with no measure
  profile        >  a named person with no measure

WHY RANKING WORDS BEHAVE DIFFERENTLY FOR A PERSON AND A GROUP, which is
the one asymmetry in the table and not a special case: an advisor is a
LEAF. There is nothing inside a person to rank, so "top revenue for Omar
Farooq" can only mean his figure. A group CONTAINS members, so a ranking
word over a group means enumerate them — the same reading
subject_level.decide() already applies (Phase 1), asked here rather than
restated.

Numeric scores still exist and still break ties WITHIN a precedence tier.
What they no longer do is decide across tiers, which is what let a 0.95
roster beat a 0.48 leaderboard for a query that named a measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class Evidence:
    """What the query actually contains, as facts rather than a score.

    Populated once per query from the resolvers that already own each
    fact — entity extraction for the subjects, metric_intent for the
    measure, intent_catalog for the phrases. Nothing here is re-derived.
    """

    named_advisors: int = 0
    named_groups: int = 0
    metric: bool = False
    ranking_phrase: bool = False
    comparison_phrase: bool = False
    roster_phrase: bool = False
    relation_phrase: bool = False
    reverse_phrase: bool = False
    # A named subject that identity resolution could not pin to ONE
    # person. The subject exists — so this is not "no subject named" —
    # but nothing may be reported about it until the user picks, which is
    # clarify_person's job. Distinguishing the two matters: treating an
    # ambiguous name as absent triggers the no-subject leaderboard rule
    # and answers about everybody instead of asking which person.
    ambiguous_subject: bool = False
    # An explicit level WORD ("advisors", "by team"). Distinct from a
    # named group: "advisors in Blue Area" names a group AND asks for the
    # level below it.
    level_word: Optional[str] = None
    # The level of the single grounded group, when there is exactly one.
    group_level: Optional[str] = None

    @property
    def subjects(self) -> int:
        return self.named_advisors + self.named_groups

    def describe(self) -> str:
        """The evidence as a phrase, for the trace."""
        parts = []
        if self.named_advisors:
            parts.append(f"{self.named_advisors} named advisor"
                         + ("s" if self.named_advisors > 1 else ""))
        if self.named_groups:
            parts.append(f"{self.named_groups} named group"
                         + ("s" if self.named_groups > 1 else ""))
        if self.metric:
            parts.append("a measure")
        if self.ranking_phrase:
            parts.append("a ranking phrase")
        if self.comparison_phrase:
            parts.append("a comparison phrase")
        if self.roster_phrase:
            parts.append("a roster phrase")
        if self.level_word:
            parts.append(f"the level word {self.level_word!r}")
        return ", ".join(parts) or "nothing specific"


@dataclass(frozen=True)
class Rule:
    """One precedence rule: when `applies`, `intent` outranks everything
    below it. `why` is written for the trace, in the user's terms."""

    intent: str
    applies: Callable[[Evidence], bool]
    why: str


def _ranking_over_a_group(e: Evidence) -> bool:
    """A ranking word with a named group means "rank the things INSIDE
    it". The group is a scope, not the answer — the same reading
    subject_level.decide() applies, and the reason a group and a person
    behave differently under a ranking word."""
    return e.ranking_phrase and e.named_groups >= 1 and e.named_advisors == 0


def _asks_for_the_level_below(e: Evidence) -> bool:
    """"advisors in Blue Area" names a group AND the level to enumerate
    inside it. Without this the group's own figure would win, and the
    query asked for its members."""
    return bool(e.level_word) and e.level_word != e.group_level


# THE precedence table, highest first. First rule whose evidence matches
# selects the winning intent; ties inside a tier fall back to the scorer's
# numeric score. Order is the specification — read it top to bottom.
PRECEDENCE: tuple[Rule, ...] = (
    Rule(
        "comparison",
        lambda e: e.subjects >= 2 and (e.metric or e.comparison_phrase),
        "two or more subjects were named with a measure, which is a "
        "two-sided question however it was phrased",
    ),
    Rule(
        "comparison",
        lambda e: e.comparison_phrase and e.subjects >= 2,
        "the query asked for a comparison and named both sides",
    ),
    Rule(
        "advisor_metric",
        lambda e: (e.named_advisors == 1 and e.named_groups == 0 and e.metric
                   and not e.ambiguous_subject
                   and not e.comparison_phrase
                   and not e.relation_phrase and not e.reverse_phrase),
        "one person and one measure — a person is a leaf, so there is "
        "nothing inside them to rank and this is their figure",
    ),
    Rule(
        "group_metric",
        # A comparison PHRASE with only one side grounded is an
        # incomplete comparison, not this group's figure: the query named
        # a second subject that did not resolve, and answering about the
        # one that did is the wrong answer to the question asked.
        lambda e: (e.named_groups == 1 and e.named_advisors == 0 and e.metric
                   and not e.comparison_phrase
                   and not _ranking_over_a_group(e)
                   and not _asks_for_the_level_below(e)
                   and not e.relation_phrase and not e.reverse_phrase),
        "one group and one measure, with nothing asking to look inside it "
        "— this is the group's own figure",
    ),
    Rule(
        "leaderboard",
        lambda e: (e.metric and not e.ambiguous_subject
                   and (e.ranking_phrase or _asks_for_the_level_below(e)
                        or e.subjects == 0)),
        "a measure to rank by, and either no named subject or an explicit "
        "request to rank what is inside one",
    ),
    Rule(
        "roster",
        lambda e: e.roster_phrase and not e.metric and not e.ranking_phrase,
        "the query asks who is in a group, with no measure and no ranking "
        "word to order them by",
    ),
    Rule(
        "advisor_profile",
        lambda e: e.named_advisors == 1 and not e.metric and not e.ambiguous_subject,
        "a person with no measure — the question is about them, not about "
        "one of their numbers",
    ),
)


# The intents this table arbitrates between: the SUBJECT/MEASURE family,
# where "who is this about and what are we measuring" is the whole
# question and the Phase 6 defects all lived.
#
# Everything else — reverse_hierarchy, hierarchy, ancestry, trend,
# attendance_filter, clarify_person, clarify_ambiguous — answers a
# DIFFERENT kind of question and keeps its existing scored behaviour
# untouched. Those intents carry evidence this table has no opinion about
# ("who is X's unit head" is not a weaker version of "X's revenue"), and
# an early version that ranked across all of them promoted
# advisor_profile over reverse_hierarchy for every manager lookup.
#
# Scoping the table is what makes it safe to state as flat precedence:
# within the family the order is total, and outside it the table stays
# silent rather than guessing.
GOVERNED = frozenset({
    "comparison", "advisor_metric", "group_metric", "leaderboard",
    "roster", "advisor_profile", "entity_summary",
})


@dataclass
class Ranking:
    """The winner, and every candidate that lost with the reason."""

    winner: object = None
    rule: Optional[Rule] = None
    rejected: list[tuple[str, str]] = field(default_factory=list)

    def trace(self) -> str:
        if self.winner is None:
            return "no candidate proposed"
        head = f"{self.winner.intent}"
        if self.rule is not None:
            head += f" (precedence: {self.rule.why})"
        else:
            head += f" (highest score {self.winner.score:.2f}; no precedence rule applied)"
        losers = " | ".join(f"not {i}: {r}" for i, r in self.rejected)
        return f"{head}{' | ' + losers if losers else ''}"


def rank(candidates: list, evidence: Evidence) -> Ranking:
    """Choose among proposed candidates using explicit precedence.

    `candidates` are already sorted by score (the scorers' own ordering).
    A precedence rule that matches BOTH the evidence and a proposed
    candidate wins outright; otherwise the highest score wins, which is
    the pre-Phase-7 behaviour and keeps every intent this table does not
    mention working exactly as before.

    Never invents an intent: a rule can only promote a candidate some
    scorer actually proposed, so precedence cannot conjure a plan that
    nothing knows how to build.
    """
    result = Ranking()
    if not candidates:
        return result

    # A specialised intent outscoring the family means this is not a
    # subject/measure question at all — leave it alone. See GOVERNED.
    if candidates[0].intent not in GOVERNED:
        result.winner = candidates[0]
        result.rejected = [
            (c.intent, f"scored {c.score:.2f} below {candidates[0].score:.2f}")
            for c in candidates[1:]
        ]
        return result

    by_intent = {c.intent: c for c in candidates}

    for rule in PRECEDENCE:
        if not rule.applies(evidence):
            continue
        candidate = by_intent.get(rule.intent)
        if candidate is None:
            continue
        result.winner = candidate
        result.rule = rule
        result.rejected = [
            (c.intent, f"{rule.intent} outranks it — {rule.why}")
            for c in candidates if c is not candidate
        ]
        return result

    # No rule applied: score order stands.
    result.winner = candidates[0]
    result.rejected = [
        (c.intent, f"scored {c.score:.2f} below {candidates[0].score:.2f}")
        for c in candidates[1:]
    ]
    return result
