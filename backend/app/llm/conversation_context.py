"""THE owner of cross-turn context.

Phase 2 (conversation context). The behavioural audit found that the
pipeline understands each turn correctly and then loses the conversation
between turns:

    "Top advisors by revenue" -> "only Graana" -> "top 3"
      the company filter silently vanishes on turn 3

    "Blue Area revenue" -> "what about pipeline?"
      the team scope silently vanishes on turn 2

THE ROOT CAUSE is that no merge step existed. Carry-over was
all-or-nothing: `ir_patcher.try_patch` either cloned the prior IR whole,
or the pipeline built a fresh IR from this turn's words with ZERO
inheritance. Every failing conversation needs the third option — inherit
some fields, override others — and there was nowhere to express it.

Two supporting causes:

  * Ellipsis was inferred from `plan.action`, which answers "what would
    the planner DO with this message", not "does this message need the
    previous one". Those are different questions and the answers diverge
    exactly on short follow-ups.

  * Two carry-over mechanisms with different contracts: the structured
    patch above, and semantic_parser serialising the prior IR into the
    LLM PROMPT, where the model decides what to keep. Behaviour therefore
    depended on nlu_mode.

WHAT THIS MODULE OWNS

  1. `specified()`  — what THIS turn's words actually pinned down. Asked
     of the existing owners (metric_aliases, subject_level,
     intent_catalog, the entity dict), never re-derived here, so this
     module cannot drift from them.

  2. `ellipsis()`   — the explicit is-this-incomplete decision that
     replaces the plan.action proxy.

  3. `merge()`      — field-by-field inheritance with one deterministic
     owner per field, and a record of what was inherited, overridden and
     discarded WITH REASONS.

DEFINITION OF ELLIPSIS, without matching phrases. A turn stands alone
when it names a MEASURE and says what to measure it over — a subject, an
explicit level word, or a ranking. Anything else is incomplete and needs
the previous turn.

    "Blue Area revenue"        metric + subject          -> standalone
    "Top advisors by revenue"  metric + level word       -> standalone
    "what about pipeline?"     metric, nothing to scope  -> ELLIPTICAL
    "only Graana"              subject, no metric        -> ELLIPTICAL
    "top 3"                    neither                   -> ELLIPTICAL

Deriving it this way is what keeps the rule free of per-phrase handling:
"and revenue?", "what about pipeline", "pipeline" and "now pipeline" all
reach the same decision because they all specify the same things.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.llm import intent_catalog as cat


# ---------------------------------------------------------------------
# 1. What this turn specified
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TurnSpec:
    """The fields THIS turn's words pinned down.

    Every attribute is answered by the component that already owns that
    decision. This dataclass is a view over those owners, never a second
    implementation of them.
    """

    metric: bool = False
    subject: bool = False
    level_word: bool = False
    period: bool = False
    limit: bool = False
    ranking: bool = False
    comparison: bool = False

    @property
    def stands_alone(self) -> bool:
        """Can this turn be answered without the previous one?

        A measure plus something to measure it over. The three ways of
        supplying the second half are equivalent for this purpose — all
        three tell the compiler what rows to produce.
        """
        return self.metric and (self.subject or self.level_word or self.ranking)


def specified(text: str, entities: dict) -> TurnSpec:
    """Ask each owner what this turn pinned down."""
    from app.llm import metric_aliases, subject_level

    entity_level, _ = subject_level.entity_level_from(entities)
    return TurnSpec(
        # metric resolution's owner
        metric=metric_aliases.resolve(text) is not None,
        # subject-level's owner (Phase 1), plus a named person
        subject=bool(entity_level) or bool(entities.get("advisor_wids")),
        level_word=cat.detect_level(text) is not None,
        # period resolution's owner writes this into the entity dict
        period=entities.get("period") is not None,
        limit=entities.get("limit") is not None,
        ranking=cat.token_match.contains_any(text.lower(), cat.RANKING_STRONG),
        comparison=bool(entities.get("comparison_targets"))
        or len(entities.get("teams") or []) > 1
        or len(entities.get("companies") or []) > 1,
    )


# A closed class of discourse connectives: words that link a turn to the
# previous one and carry no query content of their own. "now only
# IMARAT" and "only IMARAT" ask the same thing, and "and revenue?" is
# "revenue?".
#
# This is normalisation, not phrase matching — the list is finite,
# content-free and checked only at the START of a follow-up, so it cannot
# grow into a table of supported phrasings. It lives here because the
# ellipsis decision and the patcher must strip the same words; stripping
# in one and not the other is how "now only IMARAT" became a new
# question while "only IMARAT" narrowed the previous one.
_OPENERS = frozenset({"now", "and", "ok", "okay", "so", "then", "also",
                      "what", "about", "how", "well", "but"})


def strip_openers(text: str) -> str:
    """Drop leading discourse connectives. Never empties the string."""
    words = text.strip().split()
    while len(words) > 1 and words[0].lower().strip(",") in _OPENERS:
        words = words[1:]
    return " ".join(words)


# ---------------------------------------------------------------------
# 2. Ellipsis
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Ellipsis:
    is_elliptical: bool
    why: str
    confidence: str = "high"


def ellipsis(spec: TurnSpec, has_prior: bool) -> Ellipsis:
    """Does this turn need the previous one?

    Explicitly NOT derived from plan.action — see the module docstring.
    """
    if not has_prior:
        return Ellipsis(False, "no previous turn to inherit from", "high")
    if spec.stands_alone:
        return Ellipsis(
            False,
            "the turn names a measure and what to measure it over, so it "
            "stands as its own question",
            "high",
        )
    missing = []
    if not spec.metric:
        missing.append("no measure")
    if not (spec.subject or spec.level_word or spec.ranking):
        missing.append("nothing to scope it to")
    return Ellipsis(
        True,
        f"incomplete on its own ({', '.join(missing)}) — the previous turn supplies the rest",
        "high",
    )


# ---------------------------------------------------------------------
# 3. The merge
# ---------------------------------------------------------------------

# Which entity fields count as SCOPE. A scope filter names who the
# question is about; a threshold filter constrains a measure. They expire
# differently: naming a new subject replaces the scope, while a new
# measure does not.
_SCOPE_FIELDS = frozenset(
    ["team", "company", "office", "region", "bcm", "zonal_head", "unit_head", "unit"]
)


@dataclass
class MergeResult:
    ir: object
    inherited: list[tuple[str, str]] = field(default_factory=list)
    overridden: list[tuple[str, str]] = field(default_factory=list)
    discarded: list[tuple[str, str]] = field(default_factory=list)
    decision: Optional[Ellipsis] = None

    def trace(self) -> str:
        """One line for the routing trace."""
        parts = []
        if self.inherited:
            parts.append("inherited " + ", ".join(f for f, _ in self.inherited))
        if self.overridden:
            parts.append("overridden " + ", ".join(f for f, _ in self.overridden))
        if self.discarded:
            parts.append("discarded " + ", ".join(f for f, _ in self.discarded))
        return "; ".join(parts) or "nothing to merge"

    def detail(self) -> str:
        """Every field with the reason it survived, was replaced, or was
        dropped — the observability requirement."""
        lines = []
        for name, items in (("inherited", self.inherited),
                            ("overridden", self.overridden),
                            ("discarded", self.discarded)):
            for f, why in items:
                lines.append(f"{name}: {f} — {why}")
        return "\n".join(lines) or "nothing to merge"


def merge(prior, current, spec: TurnSpec, decision: Ellipsis) -> MergeResult:
    """Fill this turn's IR from the previous one where this turn was silent.

    FIELD OWNERSHIP — one owner each, and the rule is the same for all of
    them: the CURRENT turn owns any field it specified, the PREVIOUS turn
    owns the rest. What differs per field is only how "specified" is read,
    and that reading comes from TurnSpec.

        intent         current, always (this turn's question shape)
        metric/sort    current if it named a measure
        period         current if it named a window
        subject_level  current if it named a subject or a level word
        scope filters  current if it named a subject
        threshold      current if it stated one
        limit          current if it stated one

    Mutates and returns a copy of `current`; `prior` is never modified.
    """
    result = MergeResult(ir=current, decision=decision)

    if prior is None:
        result.discarded.append(("all", "no previous turn"))
        return result

    if not decision.is_elliptical:
        result.discarded.append(
            ("all", f"this turn stands alone — {decision.why}")
        )
        return result

    # ---- metric + sort ---------------------------------------------
    if spec.metric:
        result.overridden.append(
            ("metric", "this turn named a measure, which replaces the previous one")
        )
    elif prior.metric is not None:
        current.metric = prior.metric.model_copy(deep=True)
        if current.sort is not None and prior.sort is not None:
            current.sort.metric = prior.sort.metric
        result.inherited.append(
            ("metric", f"this turn named no measure, so {prior.metric.key!r} carries forward")
        )

    # ---- period -----------------------------------------------------
    if spec.period:
        result.overridden.append(
            ("period", "this turn named a time window, which replaces the previous one")
        )
    elif prior.time_range is not None and current.time_range is not None:
        current.time_range.period = prior.time_range.period
        result.inherited.append(
            ("period", f"this turn named no window, so {prior.time_range.period} carries forward")
        )

    # ---- subject level ----------------------------------------------
    if spec.subject or spec.level_word:
        result.overridden.append(
            ("subject_level", "this turn named its own subject")
        )
    else:
        current.subject_level = prior.subject_level
        result.inherited.append(
            ("subject_level", f"this turn named no subject, so {prior.subject_level!r} carries forward")
        )

    # ---- scope filters ----------------------------------------------
    # Per LEVEL, not wholesale. A subject named at a level the previous
    # turn already scoped is a CORRECTION and replaces it; a subject at a
    # different level is a NARROWING and joins it.
    #
    #   "Blue Area revenue" -> "only Downtown"     team replaces team
    #   "Downtown pipeline" -> "now only IMARAT"   company joins team
    #
    # Reading it wholesale gets one of the two wrong whichever way it is
    # written, and both readings are silent: an unwanted intersection
    # returns no rows, a dropped scope returns someone else's numbers.
    current_fields = {f.field for f in current.filters if f.field in _SCOPE_FIELDS}
    inheritable = [
        f for f in prior.filters
        if f.field in _SCOPE_FIELDS and f.field not in current_fields
    ]
    replaced = current_fields & {f.field for f in prior.filters}

    if inheritable:
        for f in inheritable:
            current.filters.append(f.model_copy(deep=True))
        result.inherited.append(
            ("filters",
             "this turn named no subject at "
             + ", ".join(sorted({f.field for f in inheritable}))
             + ", so the previous scope ("
             + ", ".join(f"{f.field}={f.value}" for f in inheritable)
             + ") still applies")
        )
    if replaced:
        result.overridden.append(
            ("filters",
             "this turn named its own "
             + ", ".join(sorted(replaced))
             + ", which corrects the previous scope at that level")
        )

    # ---- comparison subjects ----------------------------------------
    # A comparison's sides live in `subjects`, not in filters. Phase 5B
    # put comparison on the IR path, and without this a follow-up that
    # names a new measure ("what about overdue?") inherited the metric
    # rules above and then dropped both sides, degrading a two-sided
    # question into a ranking of everything.
    #
    # Same ownership rule as every other field: this turn's subjects win
    # when it named any, otherwise the previous turn's carry.
    if getattr(current, "subjects", None):
        result.overridden.append(
            ("subjects", "this turn named its own subjects to compare")
        )
    elif getattr(prior, "subjects", None):
        current.subjects = [s.model_copy(deep=True) for s in prior.subjects]
        if current.intent != "comparison":
            current.intent = prior.intent
            result.inherited.append(
                ("intent", "the previous turn was a comparison and this turn "
                           "named no new subjects, so it stays one")
            )
        result.inherited.append(
            ("subjects",
             "this turn named no subjects, so the previous comparison's sides ("
             + ", ".join(s.value for s in prior.subjects) + ") carry forward")
        )

    # ---- limit ------------------------------------------------------
    if spec.limit:
        result.overridden.append(("limit", "this turn stated how many rows it wants"))
    elif prior.limit is not None:
        current.limit = prior.limit
        result.inherited.append(
            ("limit", f"this turn stated no count, so {prior.limit} carries forward")
        )

    return result
