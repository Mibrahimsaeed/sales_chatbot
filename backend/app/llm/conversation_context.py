from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.llm import hierarchy
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
        # A comparison's sides ARE the level its answer is grouped at —
        # the rule plan_to_ir applies when it builds one. Without it a
        # narrowing follow-up ("only Graana" over a comparison of two
        # TEAMS) overrode subject_level to `company` in the block above,
        # and the compiler then grouped two teams by company: one row,
        # named after the filter, presented as the comparison.
        if current.subject_level != prior.subjects[0].type:
            current.subject_level = prior.subjects[0].type
            result.inherited.append(
                ("subject_level",
                 f"the comparison's sides are at {prior.subjects[0].type!r}, which is "
                 "the level its answer is grouped at")
            )
        if current.intent != "comparison":
            current.intent = prior.intent
            # BOTH NAMES, OR THEY DISAGREE. `operation` and `intent` are
            # the same fact under the old and new vocabularies, and
            # resolved_operation() prefers `operation` — so carrying the
            # intent alone left a follow-up reading "comparison" by one
            # field and "leaderboard" by the other, and answering as the
            # second. Carried together here because this is the one place
            # that rewrites either.
            current.operation = prior.resolved_operation()
            result.inherited.append(
                ("intent", "the previous turn was a comparison and this turn "
                           "named no new subjects, so it stays one")
            )
        result.inherited.append(
            ("subjects",
             "this turn named no subjects, so the previous comparison's sides ("
             + ", ".join(s.value for s in prior.subjects) + ") carry forward")
        )

    # A comparison's sides ARE its scope at their own level, so a scope
    # filter inherited at that level intersects the sides instead of
    # narrowing them — "Blue Area vs Downtown" AND team=Blue Area matches
    # one side and silently answers half the question.
    #
    # Per LEVEL, exactly as the filter block above: a company filter over
    # a comparison of two TEAMS is a real narrowing ("compare them within
    # Graana") and must survive. plan_to_ir applies the same rule when it
    # builds a comparison from scratch; this is where an inherited filter
    # meets subjects that arrived separately, which is the only other way
    # the pair can come together.
    subject_levels = {s.type for s in (getattr(current, "subjects", None) or [])}
    if subject_levels:
        intersecting = [f for f in current.filters if f.field in subject_levels]
        if intersecting:
            current.filters = [f for f in current.filters if f.field not in subject_levels]
            result.discarded.append(
                ("filters",
                 ", ".join(f"{f.field}={f.value}" for f in intersecting)
                 + " — this turn compares subjects at that level, and a filter "
                   "there would intersect the sides rather than narrow them")
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


# ---------------------------------------------------------------------
# 4. The carry — inheritance one step EARLIER, into the planner
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Carry:
    """What the previous turn hands to THIS turn's planner.

    `entities` is an overlay on the extracted entity dict, in exactly the
    shape extraction itself produces, so the planner cannot tell an
    inherited value from a spoken one — which is the point. `reasons`
    explains each key for the trace.
    """

    entities: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.entities)


def _prior_scope(prior) -> list[tuple[str, str]]:
    """The (level, value) pairs the previous turn was scoped to.

    Both places a subject can live: `filters` for an ordinary scoped
    query, `subjects` for a comparison. Read together because a follow-up
    does not know or care which shape established the subject.
    """
    scope = [
        (f.field, f.value) for f in prior.filters
        if f.field in _SCOPE_FIELDS and isinstance(f.value, str)
    ]
    scope += [(s.type, s.value) for s in (getattr(prior, "subjects", None) or [])]
    return scope


def carry_into_plan(prior, entities: dict, spec: TurnSpec, decision: Ellipsis,
                    *, needs_second_subject: bool = False) -> Carry:
    """What the previous turn supplies to this turn BEFORE it is planned.

    `merge()` above repairs an IR after the fact, which is enough only
    when the turn reached the IR path at all. A turn that names a subject
    and no measure does not: the planner sees a bare entity mention and
    answers it as a standalone entity summary, so "Blue Area revenue" ->
    "what about Downtown?" replied with a generic Downtown card and the
    measure the user asked about one message earlier was simply gone.

    Handing the missing piece to the PLANNER instead fixes that at the
    point the shape of the answer is decided, and — because the planner
    then re-scores the turn with the full picture — it is the intent
    precedence table that decides what the turn now is, not this module.
    This is deliberately the same mechanism the metric-inheritance branch
    in nlu_pipeline already used for "now only IMARAT", generalised from
    one plan action to the field-level rule that motivated it.

    TWO FIELDS travel this way, and both follow merge()'s ownership rule
    — the current turn owns what it named, the previous turn owns the
    rest:

      metric   when this turn named a subject (or a level word) and no
               measure. Restricted to subject-bearing turns on purpose: a
               turn naming NEITHER is a bare modifier ("top 5", "year to
               date"), which ir_patcher already resolves against the
               prior IR directly and which would only be confused by
               being re-planned as a fresh question.

      subject  when this turn asked for a comparison and only one side of
               it grounded. The other side is the subject the
               conversation has already established, exactly as
               merge()'s `subjects` block carries both sides forward once
               a comparison IS one.

    `needs_second_subject` is passed in rather than derived from a plan
    action, so this module stays free of planner vocabulary — the
    ellipsis decision must not become a function of what the planner
    would DO with the turn (see the module docstring).
    """
    if prior is None or not decision.is_elliptical:
        return Carry()

    carried: dict = {}
    reasons: list[str] = []

    if (not spec.metric and (spec.subject or spec.level_word)
            and prior.metric is not None):
        carried["metric"] = prior.metric.key
        reasons.append(
            f"metric={prior.metric.key} — this turn named a subject but no "
            "measure, so the previous turn's carries forward"
        )

    if needs_second_subject:
        added: list[str] = []
        for level, value in _prior_scope(prior):
            key = hierarchy.LEVEL_ENTITY_KEYS.get(level, f"{level}s")
            existing = list(carried.get(key) or entities.get(key) or [])
            if any(v.lower() == value.lower() for v in existing):
                continue
            carried[key] = existing + [value]
            added.append(f"{level}={value}")
        if added:
            reasons.append(
                ", ".join(added) + " — this turn asked for a comparison and "
                "grounded only one side; the other is what the conversation "
                "had already established"
            )

    return Carry(entities=carried, reasons=reasons)


# ---------------------------------------------------------------------
# 5. Describing a merge the patcher performed
# ---------------------------------------------------------------------


def summarise(ir) -> str:
    """One line naming the fields that decide what a query returns.

    Used by the trace so `previous_context` and `merged_context` read the
    same on every path.
    """
    if ir is None:
        return "none"
    parts = [f"intent={ir.intent}", f"level={ir.subject_level}"]
    if ir.metric is not None:
        parts.append(f"metric={ir.metric.key}")
    parts.append(f"period={ir.time_range.period}")
    for f in ir.filters:
        parts.append(f"{f.field}{f.operator}{f.value}")
    for s in getattr(ir, "subjects", None) or []:
        parts.append(f"subject:{s.type}={s.value}")
    parts.append(f"limit={ir.limit}")
    return " ".join(parts)


# The IR fields a patch can change, and how to read the changed value.
# Listed once so the trace below names every field ir_patcher touches
# and cannot silently omit one it grows a rule for.
_PATCHABLE = {
    "metric": lambda ir: ir.metric.key if ir.metric else None,
    "period": lambda ir: ir.time_range.period,
    "limit": lambda ir: ir.limit,
    "sort": lambda ir: (ir.sort.metric, ir.sort.direction),
    "filters": lambda ir: sorted(
        (f.field, str(f.value)) for f in ir.filters
    ),
    "subjects": lambda ir: sorted(
        (s.type, s.value) for s in (getattr(ir, "subjects", None) or [])
    ),
}


def describe_patch(prior, patched, decision: Ellipsis) -> MergeResult:
    """The patcher's work, in merge()'s vocabulary.

    ir_patcher performs a merge — it copies the previous IR and changes
    what this turn asked to change — but it is not this module, so its
    turns produced no Context step and the most implicit inheritance in
    the system ("top 5", "year to date") was the least visible in the
    trace. Rather than move the patching here (it is the cheap
    deterministic path and belongs where it is) or duplicate the
    reasoning, this DIFFS the two IRs: every field the patch left alone
    was inherited, every field it changed was overridden.

    A diff, not a second policy — it reports what the patcher did and
    cannot disagree with it.
    """
    result = MergeResult(ir=patched, decision=decision)
    for name, read in _PATCHABLE.items():
        before, after = read(prior), read(patched)
        if before == after:
            # An empty field carried forward is not inheritance — saying
            # "inherited filters" when there were none reads as a scope
            # that survived, which is the exact confusion this trace
            # exists to remove.
            if after not in (None, [], ()):
                result.inherited.append(
                    (name, f"this turn did not change it, so {after!r} carries forward")
                )
        else:
            result.overridden.append(
                (name, f"{before!r} -> {after!r} — this turn's modifier changed it")
            )
    return result
