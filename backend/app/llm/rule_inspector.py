"""
Read-only reconstruction of a rule-based planning decision (diagnostics).

WHY THIS EXISTS — the prompt audit established that person-centric
queries never reach the LLM: nlu_pipeline short-circuits `lookup`,
`summary`, `roster`, `reverse_hierarchy`, `breakdown`, `comparison` and
`attendance_filter` to the rule-based planner before semantic_parser is
ever called. For those queries "why did it answer that?" is entirely a
question about keyword signals and intent scores, and neither was
readable after the fact: the request trace records the WINNING score and
the runner-up, but not the intents that never became candidates at all —
which is exactly where a lost requirement hides. "It answered with the
advisor's profile instead of their team" is not explained by the profile
intent's score; it is explained by the hierarchy intent never scoring.

This module reconstructs the full picture from the same inputs the
planner saw:

- `signals()` reports every keyword class in the intent catalog, whether
  it fired, and the literal text that matched it. A NOT-detected signal
  is reported as prominently as a detected one — the absent keyword is
  usually the answer.
- `planner_decision()` calls query_planner.score_intents(), which the
  planner itself exposes for exactly this purpose, and reports EVERY
  scorer: those that produced a candidate (with score and evidence) and
  those that declined.

PURITY. score_intents() is a pure function of (text, entities) — no I/O,
no DB, no clock, no randomness — so re-running it here reproduces the
decision the planner made rather than approximating it. That is what
makes this a deterministic trace and not a guess. The audit still prints
the plan's OWN recorded action alongside the reconstructed winner, so a
divergence (which would mean this assumption had broken) is visible
instead of silently papered over.

Nothing here plans, executes, or alters anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.llm import intent_catalog as cat


@dataclass
class Signal:
    """One keyword class from the catalog, and whether this query fired it."""
    name: str
    detected: bool
    matched: list[str] = field(default_factory=list)
    means: str = ""

    def line(self) -> str:
        if self.detected:
            return f"{self.name}: DETECTED {self.matched} — {self.means}"
        return f"{self.name}: not detected — {self.means}"


@dataclass
class PlannerDecision:
    candidates: list[dict] = field(default_factory=list)
    declined: list[str] = field(default_factory=list)
    winner: dict | None = None
    error: str | None = None


def _find(text: str, words) -> list[str]:
    low = (text or "").lower()
    return [w for w in words if w in low]


def _search(text: str, pattern) -> list[str]:
    try:
        match = pattern.search(text or "")
    except Exception:
        return []
    return [match.group(0)] if match else []


def signals(text: str) -> list[Signal]:
    """Every trigger class the planner consults, detected or not.

    The `means` text states what the signal STEERS TOWARD, so a reader
    can connect an absent keyword to the intent that consequently never
    scored — the "keyword 'team' not detected, so hierarchy never became
    a candidate" chain."""
    ranking_strong = _find(text, cat.RANKING_STRONG)
    ranking_weak = _find(text, cat.RANKING_WEAK)
    flat = _find(text, cat.FLAT_KEYWORDS)
    roster = _search(text, cat.ROSTER_RE)
    comparison = _search(text, cat.COMPARISON_RE)
    relational = _search(text, cat.RELATIONAL_RE)
    reverse = _search(text, cat.REVERSE_RE)

    return [
        Signal("ranking_strong", bool(ranking_strong), ranking_strong,
               "steers toward leaderboard (top/best/worst)"),
        Signal("ranking_weak", bool(ranking_weak), ranking_weak,
               "weak ranking hint (show me/give me) — barely evidence"),
        Signal("relational", bool(relational), relational,
               "steers toward hierarchy: the people UNDER someone, grouped by team"),
        Signal("reverse", bool(reverse), reverse,
               "steers toward reverse_hierarchy: the one person ABOVE someone"),
        Signal("roster", bool(roster), roster,
               "steers toward roster: a flat enumeration of people"),
        Signal("comparison", bool(comparison), comparison,
               "steers toward comparison: two or more named entities"),
        Signal("flat", bool(flat), flat,
               "requests an ungrouped list instead of nested-by-team"),
    ]


def planner_decision(text: str, entities: dict) -> PlannerDecision:
    """Re-derives the intent scoring for this query.

    Reports the scorers that DECLINED as well as those that scored: an
    intent that never became a candidate is invisible in the plan itself,
    and is the single most common place a query's real requirement is
    dropped."""
    try:
        from app.llm.query_planner import _SCORERS, score_intents

        ctx, candidates = score_intents(text, entities)
        scored = [
            {
                "intent": c.intent,
                "score": round(c.score, 3),
                "evidence": list(c.evidence),
            }
            for c in candidates
        ]
        # Which scorers declined is established by ASKING each scorer with
        # the very ctx score_intents built, not by matching function names
        # against candidate intents — those names diverge (_score_attendance
        # produces intent "attendance_filter"), and a name-matched list
        # would confidently report a scorer as declined while it had in
        # fact won. Scorers are pure functions of ctx, so re-calling them
        # observes the same answer the planner got.
        declined = [scorer.__name__ for scorer in _SCORERS if scorer(ctx) is None]
        return PlannerDecision(
            candidates=scored,
            declined=declined,
            winner=scored[0] if scored else None,
        )
    except Exception as e:  # diagnostics must never raise into a request
        return PlannerDecision(error=f"{type(e).__name__}: {e}")


def why_lines(decision: PlannerDecision, signal_list: list[Signal]) -> list[str]:
    """The plain-language "why" for the intent choice.

    Deliberately conservative: it states what WAS scored and what was
    NOT, and never invents a reason a particular scorer declined. The
    absent-signal list is printed alongside so the reader can draw the
    connection from evidence rather than from a guess this module made.
    """
    lines: list[str] = []
    if decision.error:
        return [f"Intent reconstruction unavailable: {decision.error}"]
    if not decision.candidates:
        lines.append("No intent scored at all — the planner fell through to its fallback.")
    else:
        winner = decision.candidates[0]
        lines.append(
            f"Selected intent '{winner['intent']}' (score {winner['score']}) "
            f"on evidence {winner['evidence'] or '[]'}."
        )
        if len(decision.candidates) > 1:
            runner = decision.candidates[1]
            lines.append(
                f"Chosen over '{runner['intent']}' (score {runner['score']}, "
                f"evidence {runner['evidence'] or '[]'})."
            )
        else:
            lines.append("No other intent scored — this was the only candidate.")

    if decision.declined:
        lines.append(
            "Never became candidates (their required signals/entities were absent): "
            + ", ".join(decision.declined)
        )
    absent = [s.name for s in signal_list if not s.detected]
    if absent:
        lines.append("Keyword signals NOT detected: " + ", ".join(absent))
    return lines
