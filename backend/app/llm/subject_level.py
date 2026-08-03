"""THE owner of "what is this question about?".

Phase 2. The behavioural audit found that a correctly understood query
still answered the wrong question 12% of the time: entity resolution
grounded "Downtown" as a team, the metric and period resolved perfectly,
and the planner then answered with a list of advisors because
`pipeline_value.primary_level` is "advisor".

    "What is Downtown's pipeline value this month?"
      -> "Shehryar Abbasi has 3,500 MTD Open Pipeline, ranking 1st of 2
          advisors shown filtered by team = Downtown."

The scope was right and the number was right. The GRANULARITY answered a
different question than the one asked.

A controlled experiment isolated the cause — same query shape, same named
group, only the metric changed:

    Blue Area's achievement %   primary_level=team     -> team      (right)
    Blue Area's revenue         primary_level=advisor  -> advisor   (wrong)

So the level came from the METRIC, never from the SUBJECT the user named.
`metric.primary_level` is the correct answer for an unscoped question
("what is revenue?") and the wrong answer whenever the query named a
subject — but it was consulted first, and the grounded entity was demoted
to a filter.

THE PRECEDENCE, highest first:

  1. An explicit level WORD.  "by team", "which advisor", "top 3 teams".
     The user said what they want enumerated; nothing outranks that.

  2. The GROUNDED ENTITY's own level.  "Downtown" is a team, "Graana" is
     a company. The subject the user named owns the level.

     ONE exception, and it is not a special case so much as the same
     rule read correctly: when the query carries a strong RANKING signal,
     the named group is a SCOPE rather than the subject. "Top 5 in Blue
     Area" asks to enumerate the people IN Blue Area — ranking a single
     team against itself is not a question anyone asks. So a ranking
     hands the decision down to the metric default, which enumerates
     members.

  3. A RELATION level ("his team", "Ahmed's unit"). Passed in by the
     caller when a relation resolver already established it; this module
     does not re-derive relations.

  4. `metric.primary_level` — the default, reached only when the query
     named no subject at all.

WHY A MODULE. Four call sites decided this independently before Phase 2:
query_planner's leaderboard scorer, llm_planner's `_leaderboard_level`,
ir_validator (which RE-decided it after the planner had already chosen),
and query_ir's transfer. Three of the four consulted `primary_level`
first, so fixing one left the others wrong — the recurring defect in this
codebase is the same rule written down more than once with nothing
forcing the copies to agree. This module is that forcing mechanism: the
decision is made here, and `Decision.why`/`.rejected` carry enough
context that no caller needs to re-derive it to explain it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.llm import intent_catalog as cat


@dataclass(frozen=True)
class Candidate:
    """One claimant on the subject level, and where it came from."""

    source: str          # "level_word" | "entity" | "relation" | "metric_default"
    level: str
    detail: str = ""     # the evidence — the word matched, the entity grounded


@dataclass(frozen=True)
class Decision:
    """The chosen level, plus every claimant that lost and why.

    The audit needed three things the old trace could not answer: which
    claimant won, what the others proposed, and why they lost. Recording
    only the winner cannot explain what it displaced, so the losers ride
    along.
    """

    level: str
    source: str
    why: str
    rejected: tuple[tuple[str, str], ...] = ()   # (source, reason) pairs

    def trace(self) -> str:
        """One line per claimant, winner first — for the routing trace."""
        parts = [f"{self.source}={self.level} (chosen: {self.why})"]
        parts += [f"{src} lost: {reason}" for src, reason in self.rejected]
        return " | ".join(parts)


def decide(
    *,
    level_word: Optional[str] = None,
    entity_level: Optional[str] = None,
    entity_value: Optional[str] = None,
    relation_level: Optional[str] = None,
    metric_default: Optional[str] = None,
    has_ranking: bool = False,
) -> Decision:
    """Resolve the subject level from every available signal.

    Pure: no database, no entity dict, no metric lookup. Callers pass what
    they already resolved, which keeps this testable in isolation and
    means it cannot disagree with the resolvers by re-reading their input.
    """
    rejected: list[tuple[str, str]] = []

    def _note(source: str, level: Optional[str], reason: str) -> None:
        if level:
            rejected.append((f"{source}={level}", reason))

    # ---- 1. explicit level word ------------------------------------
    if level_word:
        _note("entity", entity_level, "an explicit level word outranks the named subject")
        _note("relation", relation_level, "an explicit level word outranks the relation")
        _note("metric_default", metric_default, "the query said which level it wants")
        return Decision(
            level=level_word, source="level_word",
            why="the query named the level explicitly",
            rejected=tuple(rejected),
        )

    # ---- 2. the grounded subject -----------------------------------
    # A ranking turns the named group into a scope: "top 5 in Blue Area"
    # enumerates its members, it does not rank Blue Area against itself.
    if entity_level and not has_ranking:
        _note("relation", relation_level, "the query named a subject directly")
        _note("metric_default", metric_default,
              "a subject was named, so the metric's default does not apply")
        named = f" ({entity_value})" if entity_value else ""
        return Decision(
            level=entity_level, source="entity",
            why=f"the query's subject{named} is a {entity_level}",
            rejected=tuple(rejected),
        )

    if entity_level and has_ranking:
        rejected.append((
            f"entity={entity_level}",
            "a ranking makes the named group a SCOPE, not the subject — "
            "the members inside it are what gets enumerated",
        ))

    # ---- 3. a relation already resolved ----------------------------
    if relation_level:
        _note("metric_default", metric_default, "a relation established the subject")
        return Decision(
            level=relation_level, source="relation",
            why="a relation in the query established the subject",
            rejected=tuple(rejected),
        )

    # ---- 4. the metric's default -----------------------------------
    if metric_default:
        return Decision(
            level=metric_default, source="metric_default",
            why="the query named no subject, so the metric's own level applies",
            rejected=tuple(rejected),
        )

    return Decision(
        level="advisor", source="fallback",
        why="no level word, no subject, and no metric default",
        rejected=tuple(rejected),
    )


def entity_level_from(entities: dict) -> tuple[Optional[str], Optional[str]]:
    """The most granular grounded GROUP entity, as (level, value).

    Reads GROUP_LEVEL_ORDER so "Blue Area in Graana" resolves to the team
    rather than the company — the narrower subject is the one the user is
    asking about. Advisors are deliberately excluded: a named advisor is
    handled by the advisor_metric/profile intents long before a
    leaderboard level is needed, and treating one as a leaderboard subject
    would rank a single person against nobody.
    """
    for level in cat.GROUP_LEVEL_ORDER:
        value = entities.get(level)
        if value:
            return level, value
    return None, None
