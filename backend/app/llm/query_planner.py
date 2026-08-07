"""
Turns (text, entities) into a QueryPlan.

INTENT SELECTION IS SCORED, NOT ORDERED. Every intent that could apply
becomes a candidate with a score built from named evidence (see
intent_catalog.py), and the highest score wins. The previous design was
an ordered list of `_plan_*` functions where the first to return a plan
won, which had two problems this replaces:

  1. Priority was implicit in source order. Reading one function told you
     nothing about whether it would ever be reached.
  2. Every conflict was fixed by hand-coding a guard into whichever
     function ran first ("decline if metric and ranking", "decline if
     relational"). Each guard was locally correct and globally invisible,
     so the same class of misrouting kept resurfacing in new phrasings —
     which is how "all advisors in Blue Area" ended up answered with
     aggregate metrics.

Now a conflict is a score comparison. "top 5 advisors in Blue Area by
revenue" matches roster phrasing AND ranking phrasing; leaderboard wins
because a strong ranking word plus a resolved metric outweighs a roster
phrase, and that is stated once as a weight instead of as a guard in
two other branches.

Every decision carries its evidence. `QueryPlan.intent_score` and
`.intent_evidence` are recorded in the request trace, so "why did this
route here, and what came second?" is answerable from a log line.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
from typing import Callable

from app.llm import (
    comparators, hierarchy, intent_catalog as cat, intent_precedence,
    metric_intent, routing, subject_level, token_match,
)
from app.llm.metric_intent import MetricIntent
from app.llm.metric_ontology import (
    METRICS, lower_is_better, resolve_metric, resolve_metric_evidence,
)
from app.llm.query_compiler import is_answerable

# Re-exported for callers and tests that referenced these from here
# before the catalog split.
RANKING_KEYWORDS = list(cat.RANKING_STRONG)
FLAT_KEYWORDS = list(cat.FLAT_KEYWORDS)
LEVEL_KEYWORDS = cat.LEVEL_KEYWORDS
RELATIONAL_RE = cat.RELATIONAL_RE
REVERSE_RE = cat.REVERSE_RE
ROSTER_RE = cat.ROSTER_RE
COMPARISON_RE = cat.COMPARISON_RE


@dataclass
class QueryPlan:
    action: str
    level: str | None = None
    entity_value: str | None = None
    entity_wid: int | None = None
    metric: str | None = None
    limit: int = 10
    # None = the user named no direction, so the metric's own polarity
    # decides (see query_compiler.default_direction). True/False mean the
    # user was explicit.
    ascending: bool | None = None
    # F4. "MTD" | "YTD" | "3M" when the user named a window, None when
    # they did not — the same "None means unstated" convention as
    # `ascending` above, and for the same reason: the two cases must stay
    # distinguishable downstream. plan_to_ir falls back to the metric's
    # own period only when this is None, so a stated period can no longer
    # be overwritten by the measure that happened to resolve.
    #
    # The struct having nowhere to put this IS the F4 defect: extraction
    # set entities["period"]="YTD" correctly, and the value died here.
    period: str | None = None
    # F8. [{"operator": ">", "value": 80.0}, ...] — comparator/value
    # pairs from entity extraction, applied to `metric`. A threshold has
    # no field of its own because the user names the measure once ("...
    # with achievement above 80 percent"), so it binds to whatever this
    # plan already resolved. Same story as `period`: extracted correctly,
    # then dropped for want of a field, and the reply listed everybody.
    thresholds: list[dict] = dataclass_field(default_factory=list)
    # Phase 5.4 — reverse_hierarchy only. `level` is the level being ASKED
    # FOR (the manager); this is the level of the subject being asked
    # ABOUT. They were the same thing while only advisors could be
    # subjects, and are not once a BCM can be one.
    subject_level: str | None = None
    flat: bool = False
    ambiguous: dict | None = None
    person_candidates: list = dataclass_field(default_factory=list)
    # comparison action only — [(level, value), ...] preserving each
    # entity's own type, since a comparison can span levels.
    comparison_targets: list = dataclass_field(default_factory=list)
    # EVERY measure the query named, in the order it named them, with
    # `metric` above remaining the primary one.
    #
    # A plan carries one metric because almost every query names one, and
    # that stays true — this list holds a single entry for those, so
    # nothing reading `metric` changes. It exists so "connects and
    # answered calls" can reach dispatch as the two measures it is
    # instead of as whichever alias string happened to be longest.
    #
    # Deliberately on the PLAN and not on QueryIR: the two shapes that
    # need it — one person with several measures, and a comparison over
    # several measures — are both served on the plan path, the second by
    # comparison_service, which has taken a tuple of KPI keys since it
    # was written. Widening QueryIR would mean teaching the compiler,
    # validator, context merge and formatter about metric lists to reach
    # the same answers.
    metrics: list[str] = dataclass_field(default_factory=list)
    reason: str = ""
    # How the intent was chosen — recorded in the request trace so a
    # misroute is diagnosable without re-running the planner by hand.
    intent_score: float = 0.0
    intent_evidence: list[str] = dataclass_field(default_factory=list)
    runner_up: str | None = None


@dataclass
class _Intent:
    """One shared reading of the query, computed once. Each scorer is a
    pure function of this, which is what makes the scores comparable."""
    text: str
    q: str
    entities: dict
    metric: str | None
    # F6: whether a measure was NAMED, not just whether one resolved.
    # Computed once here so every scorer reads the same verdict.
    metric_intent: "MetricIntent"
    # `q` with the matched metric PHRASE blanked out, for level
    # detection. A metric name can contain a level word — "1 unit ratio"
    # contains "unit", which names unit_head — so scanning the raw text
    # reads the measure's own name as the grouping level and ranks unit
    # heads for a team board. Same shape as the comparator masking in
    # _without_comparators.
    level_q: str 
    has_ranking_strong: bool
    has_ranking_weak: bool
    is_flat: bool
    is_relational: bool
    is_reverse: bool
    is_roster: bool
    is_comparison: bool

    @property
    def is_ranking(self) -> bool:
        """Kept for the leaderboard/summary distinction, which historically
        treated weak cues as ranking evidence."""
        return self.has_ranking_strong or self.has_ranking_weak

    def group_entity(self) -> tuple[str, str] | None:
        """The most granular grounded group entity, as (level, value)."""
        for level in cat.GROUP_LEVEL_ORDER:
            value = self.entities.get(level)
            if value:
                return level, value
        return None

    def comparison_targets(self) -> list[tuple[str, str]]:
        """The entities a comparison should set side by side.

        M6: this is `all_group_entities()` plus the NAMED ADVISOR, when
        the query named exactly one. "How does Waqar Haider compare to
        his team" has two sides — a person and a group — and reading only
        group entities saw one, producing "I could only find Blue Area"
        for a question that named both.

        Nothing downstream needed changing to support it: `advisor` is
        already a hierarchy level backed by Advisor.name, so
        comparison_service treats ("advisor", "Waqar Haider") like any
        other target and QueryIR needed no new shape. The gap was only
        that the planner never built such a target.

        Kept separate from all_group_entities() rather than folded into
        it because the other four callers ask "which GROUP is this query
        about" — a question an advisor is not an answer to.
        """
        targets = self.all_group_entities()
        name = self.entities.get("advisor_name")
        wids = self.entities.get("advisor_wids") or []

        # Phase 5B: SEVERAL named advisors are several sides of the
        # comparison. The single-advisor branch below requires exactly one
        # wid, so "compare Yasir Ali and Sana Tariq" — with both people
        # resolved — produced ZERO targets and fell through to a
        # leaderboard. Advisor-vs-advisor is the one shape every group
        # level supported and the advisor level did not.
        #
        # Group targets take precedence when both are present: naming two
        # people to reach two groups ("compare X's team with Y's team")
        # is a comparison of the groups, which is what the _reference_
        # sources check below encodes for the single-advisor case.
        sources_all = self.entities.get("_reference_sources") or []
        multi = self.entities.get("advisor_multi") or []
        # A possessive relation makes the named people SOURCES, not
        # subjects: "compare Waqar Haider's team with Sana Tariq's team"
        # compares two teams. `_reference_sources` records that only once
        # relation inference has actually bound the groups, so it is
        # empty when RELATION_INFERENCE_ENABLED is off — and without this
        # check the query silently became a comparison of the two PEOPLE,
        # which is a different question with a confident answer.
        # reference_parser owns the possessive pattern (Phase 1 used the
        # same signal for the unresolved-subject gate).
        from app.llm import reference_parser

        names_a_relation = bool(reference_parser.parse(self.q))
        if len(multi) >= 2 and not targets and not sources_all and not names_a_relation:
            # advisor_multi is ordered as the names appear in the text.
            return [("advisor", person["name"]) for person in multi]

        names = self.entities.get("advisor_names") or []
        if len(wids) >= 2 and not targets and not sources_all and not names_a_relation:
            return [("advisor", value) for value in names]
        # A person named only to REACH a group is not one of the things
        # being compared: "compare X's team with Y's team" is about two
        # teams, and adding X would make it a three-way comparison of a
        # person against two groups. Entity extraction records which wids
        # were consumed that way.
        sources = self.entities.get("_reference_sources") or []
        # names_a_relation guards this branch too: with relation
        # inference disabled, `sources` is empty even though the query
        # said "X's team", and the person would become a comparison
        # SUBJECT — turning a question about two teams into a question
        # about two people, answered confidently.
        if name and len(wids) == 1 and wids[0] not in sources and not names_a_relation and not any(
            level == "advisor" for level, _value in targets
        ):
            # The person leads: "how does X compare to his team" asks
            # about X first.
            return [("advisor", name), *targets]
        return targets

    def all_group_entities(self) -> list[tuple[str, str]]:
        """EVERY grounded group entity, as (level, value), preserving each
        one's type — a comparison can span levels ("compare Blue Area with
        Graana"), so collapsing them to a single level would lose exactly
        the information the comparison needs.

        Reads the plural keys the extractor emits (entities["companies"],
        ["teams"], ...) rather than the backward-compatible singulars,
        because a comparison is precisely the case where the second and
        later matches matter. Deduplicated on (level, value), order
        preserved so the reply lists them as the user said them."""
        by_level: dict[str, list[str]] = {}
        for level in cat.GROUP_LEVEL_ORDER:
            plural_key = hierarchy.LEVEL_ENTITY_KEYS.get(level)
            values = list(self.entities.get(plural_key) or [])
            if not values and self.entities.get(level):
                values = [self.entities[level]]
            if values:
                by_level[level] = values

        # If ONE level accounts for two or more of the named entities,
        # that level is the comparison. Necessary because a single name
        # can ground at several levels — production has a company
        # "Graana" (341 advisors) AND an office literally named "Graana"
        # (1), so naively collecting across levels turned "compare Graana
        # and Agency21" into a THREE-way comparison with Graana against
        # itself. Preferring the level that covers the most targets picks
        # the reading the user meant without needing a database lookup.
        for level in cat.GROUP_LEVEL_ORDER:
            values = by_level.get(level, [])
            if len(values) >= 2:
                return [(level, v) for v in values]

        # No single level covers it — a genuine cross-level comparison
        # ("compare Blue Area with Graana"), so combine, one per level.
        found: list[tuple[str, str]] = []
        seen: set[str] = set()
        for level in cat.GROUP_LEVEL_ORDER:
            for value in by_level.get(level, []):
                if value.lower() not in seen:
                    seen.add(value.lower())
                    found.append((level, value))
        return found


@dataclass
class _Candidate:
    intent: str
    score: float
    evidence: list[str]
    build: Callable[[], QueryPlan]


def _advisor_plan(entities: dict, action: str = "lookup", level: str | None = None,
                  metric: str | None = None) -> QueryPlan:
    """A plan targeting ONE person. Shared by the profile lookup, the
    reverse-hierarchy lookup and the single-metric lookup, since all
    three need the same "an ambiguous name must ask, never pick" guard —
    including the metric one: "connects of Yasir Ali" must ask which
    Yasir Ali before reporting anybody's number."""
    if entities.get("advisor_ambiguous"):
        resolution = entities.get("advisor_resolution")
        return QueryPlan(
            action="clarify_person",
            level="advisor",
            entity_value=entities.get("advisor_name"),
            person_candidates=list(resolution.candidates) if resolution else [],
        )
    return QueryPlan(
        action=action,
        level=level or "advisor",
        entity_value=entities.get("advisor_name"),
        entity_wid=entities.get("advisor_wid"),
        metric=metric,
    )


# =====================================================================
# Scorers. Each returns a _Candidate or None. A scorer returning None
# means "this intent does not apply", NOT "something else is better" —
# comparison is the selector's job.
# =====================================================================

def _score_clarify_ambiguous(ctx: _Intent) -> _Candidate | None:
    """A name matching several hierarchy levels. A HARD GATE: answering
    any reading would be a guess, so this outscores everything.

    Two phrasings resolve the ambiguity by themselves and are handled
    here by pruning the losing entity and declining, letting the normal
    scorers run against a now-unambiguous query."""
    ambiguous = ctx.entities.get("ambiguous_entity")
    if not ambiguous:
        return None

    # A REVERSE question is about a PERSON: "who is Ali Murtaza's unit
    # head?" asks for the manager OF Ali Murtaza, so "unit head" names the
    # role being asked for, not the subject. Checked before any keyword
    # evidence, or the level-keyword scan below concludes the subject is
    # the unit_head reading and answers the opposite question.
    if ctx.is_reverse and "advisor" in ambiguous["levels"]:
        ctx.entities = {
            k: v for k, v in ctx.entities.items()
            if k not in [lvl for lvl in ambiguous["levels"] if lvl != "advisor"]
        }
        return None

    named_levels = [
        level for level in ambiguous["levels"]
        if token_match.contains_any(ctx.q, hierarchy.LEVEL_KEYWORDS.get(level, []))
    ]

    # A relational phrasing likewise settles it: only a manager HAS a team.
    if len(named_levels) != 1 and ctx.is_relational:
        managerial = [lvl for lvl in ambiguous["levels"] if lvl in hierarchy.NEW_GROUP_LEVELS]
        if len(managerial) == 1:
            named_levels = managerial

    if len(named_levels) == 1:
        explicit = named_levels[0]
        pruned = dict(ctx.entities)
        for level in ambiguous["levels"]:
            if level != explicit:
                pruned.pop("advisor_name" if level == "advisor" else level, None)
        ctx.entities = pruned
        return None

    return _Candidate(
        intent="clarify_ambiguous",
        score=cat.W_HARD_GATE,
        evidence=[f"ambiguous_across_levels:{','.join(ambiguous['levels'])}"],
        build=lambda: QueryPlan(action="clarify_ambiguous", ambiguous=ambiguous),
    )


def _score_comparison(ctx: _Intent) -> _Candidate | None:
    """"Compare Graana and Agency21" — two entities side by side.

    Requires BOTH a comparison phrase and at least two grounded entities.
    Demanding two entities is what keeps a bare "compare" (with nothing
    to compare) from claiming the query — it falls through to whatever
    else applies rather than producing an empty comparison.

    Outscores the leaderboard even when a metric is named, because
    "compare A and B by revenue" is a two-sided question and a ranking
    answers it with one list that drops the pairing."""
    # M6: includes the named advisor, so a person can be compared against
    # a group — see _Intent.comparison_targets().
    targets = ctx.comparison_targets()

    # Phase 7: TWO GROUNDED SUBJECTS PLUS A MEASURE IS A COMPARISON,
    # however it was phrased. This used to require a comparison PHRASE,
    # so "Blue Area and Downtown revenue" proposed only a leaderboard,
    # plan_to_ir kept one entity filter, and the reply answered about
    # Blue Area with Downtown silently dropped — half the question, with
    # no signal that half was missing.
    #
    # The subjects are the evidence. A phrase is one way to ask for a
    # comparison; naming two things and a measure is another, and the
    # audit's rule is that a grounded subject must participate in intent
    # selection rather than being discarded by whatever else matched.
    if not ctx.is_comparison:
        if len(targets) < 2 or not ctx.metric:
            return None
        return _Candidate(
            intent="comparison",
            score=cat.PRIOR["comparison"] + cat.W_ENTITY,
            evidence=[f"targets:{len(targets)}", f"metric:{ctx.metric}",
                      "no_comparison_phrase"],
            build=lambda: QueryPlan(
                action="comparison", metric=ctx.metric,
                comparison_targets=targets,
                level=targets[0][0], entity_value=targets[0][1],
            ),
        )

    if len(targets) == 1:
        # A comparison was clearly asked for but only ONE side grounded —
        # typically the other name doesn't exist ("compare Blue Area and
        # DHA", where no team called DHA exists). Silently answering
        # about the one that resolved is the wrong answer to the question
        # asked, so say which side is missing. Scored above leaderboard
        # so it beats the "just rank the one entity" reading.
        found = targets[0][1]
        return _Candidate(
            intent="comparison",
            score=cat.PRIOR["comparison"] + cat.W_EXPLICIT_PHRASE,
            evidence=["comparison_phrase", "targets:1", "incomplete"],
            build=lambda: QueryPlan(
                action="comparison_incomplete",
                level=targets[0][0],
                entity_value=found,
                comparison_targets=list(targets),
            ),
        )
    if len(targets) < 2:
        return None

    score = cat.PRIOR["comparison"] + cat.W_EXPLICIT_PHRASE + cat.W_COMPARISON_PAIR
    evidence = ["comparison_phrase", f"targets:{len(targets)}"]
    if ctx.metric:
        score += cat.W_METRIC
        evidence.append(f"metric:{ctx.metric}")

    metric = ctx.metric
    return _Candidate(
        intent="comparison", score=score, evidence=evidence,
        build=lambda: QueryPlan(
            action="comparison",
            level=targets[0][0],
            entity_value=targets[0][1],
            metric=metric,
            comparison_targets=list(targets),
        ),
    )


def _score_roster(ctx: _Intent) -> _Candidate | None:
    """"All advisors in Blue Area" — enumerate the people."""
    if not ctx.is_roster:
        return None
    group = ctx.group_entity()
    if not group:
        return None
    level, value = group

    score = cat.PRIOR["roster"] + cat.W_EXPLICIT_PHRASE + cat.W_ENTITY
    evidence = ["roster_phrase", f"group_entity:{level}"]
    return _Candidate(
        intent="roster", score=score, evidence=evidence,
        build=lambda: QueryPlan(action="roster", level=level, entity_value=value),
    )


def _score_hierarchy(ctx: _Intent) -> _Candidate | None:
    """"X's team" / "who reports to X" — the group under someone."""
    if not ctx.is_relational:
        return None

    group = ctx.group_entity()
    if group and group[0] in hierarchy.NEW_GROUP_LEVELS:
        level, value = group
        return _Candidate(
            intent="hierarchy",
            score=cat.PRIOR["hierarchy"] + cat.W_EXPLICIT_PHRASE + cat.W_ENTITY,
            evidence=["relational_phrase", f"group_entity:{level}"],
            build=lambda: QueryPlan(
                action="breakdown", level=level, entity_value=value, flat=ctx.is_flat
            ),
        )
    if ctx.entities.get("team"):
        team = ctx.entities["team"]
        return _Candidate(
            intent="hierarchy",
            score=cat.PRIOR["hierarchy"] + cat.W_EXPLICIT_PHRASE + cat.W_ENTITY,
            evidence=["relational_phrase", "group_entity:team"],
            build=lambda: QueryPlan(action="summary", level="team", entity_value=team),
        )
    return None


def _score_reverse_hierarchy(ctx: _Intent) -> _Candidate | None:
    """"Who is X's BM?" — the person above someone.

    Phase 5.4: the subject may be a MANAGER, not only an advisor.
    "Which zonal head oversees BCM Usman Ghani" used to fall through here
    (no advisor_name) and be answered by a breakdown of that BCM's own
    advisors — a confident list that answers a different question. The
    chain has always supported it; nothing exposed it.
    """
    if not ctx.is_reverse:
        return None
    if not ctx.entities.get("advisor_name"):
        return _score_group_reverse_hierarchy(ctx)
    level = cat.detect_reverse_level(ctx.q)
    score = cat.PRIOR["reverse_hierarchy"] + cat.W_EXPLICIT_PHRASE
    evidence = ["reverse_phrase", f"manager_level:{level}"]
    if ctx.entities.get("advisor_wid"):
        score += cat.W_IDENTITY
        evidence.append("identity_resolved")
    return _Candidate(
        intent="reverse_hierarchy", score=score, evidence=evidence,
        build=lambda: _advisor_plan(ctx.entities, action="reverse_hierarchy", level=level),
    )


def _score_group_reverse_hierarchy(ctx: _Intent) -> _Candidate | None:
    """A MANAGER's manager: "who is BCM X's zonal head".

    The target level is whatever role the question names, and failing
    that the subject's own parent in the chain — so "who is above X"
    needs no role word at all. Both come from hierarchy, so neither
    names a level here.
    """
    group = ctx.group_entity()
    if group is None:
        return None
    subject_level, subject_value = group
    if not hierarchy.is_chain_level(subject_level):
        return None

    named = cat.detect_reverse_level(ctx.q) if _names_a_role(ctx.q) else None
    target = named or hierarchy.parent_of(subject_level)
    if target is None or target == subject_level:
        # A team has no parent — the chain's root. Declining lets the
        # normal readings answer instead of inventing a manager.
        return None

    score = cat.PRIOR["reverse_hierarchy"] + cat.W_EXPLICIT_PHRASE + cat.W_ENTITY
    return _Candidate(
        intent="reverse_hierarchy",
        score=score,
        evidence=["reverse_phrase", f"subject:{subject_level}", f"manager_level:{target}"],
        build=lambda: QueryPlan(
            action="reverse_hierarchy",
            level=target,
            subject_level=subject_level,
            entity_value=subject_value,
        ),
    )


def _names_a_role(q: str) -> bool:
    """Did the question name a specific manager level, or just ask who is
    above? "who is above X" must use the chain's parent rather than
    detect_reverse_level's fallback, which would answer about unit_head
    regardless of where X sits."""
    return any(pattern.search(q) for _level, pattern in cat.REVERSE_LEVEL_PATTERNS)


def _score_trend(ctx: _Intent) -> _Candidate | None:
    """"Show me the trend of revenue", "is Yasir Ali improving?"

    Recognised so the system can REFUSE it honestly. The data model
    stores only the current row per period, so there is no earlier value
    to diff against — ir_validator._UNSUPPORTED_INTENTS carries that
    reason, and this scorer is what makes it reachable. Until this
    existed, a trend question scored as a leaderboard and answered with a
    point-in-time ranking, which is a different question wearing the
    right words.

    Scored at W_HARD_GATE: a trend word is an unambiguous statement about
    the SHAPE of the answer wanted, and no snapshot reading of the same
    message is preferable to saying we cannot do it yet.
    """
    if not cat.TREND_RE.search(ctx.q):
        return None

    return _Candidate(
        intent="trend",
        score=cat.W_HARD_GATE,
        evidence=["trend_vocabulary"],
        build=lambda: QueryPlan(
            action="trend",
            metric=ctx.metric,
            level=ctx.entities.get("level"),
        ),
    )


def _score_ancestry(ctx: _Intent) -> _Candidate | None:
    """"Show me the full hierarchy above X" — every level up, at once.

    Distinct from reverse_hierarchy, which answers about ONE level. A
    chain of four managers is a different answer shape from a single
    name, and collapsing them would drop three of them.
    """
    if not cat.ANCESTRY_RE.search(ctx.q):
        return None

    subject_level, subject_value = None, None
    if ctx.entities.get("advisor_name"):
        subject_level, subject_value = "advisor", ctx.entities["advisor_name"]
    else:
        group = ctx.group_entity()
        if group and hierarchy.is_chain_level(group[0]):
            subject_level, subject_value = group
    if subject_value is None or not hierarchy.ancestors(subject_level):
        return None

    return _Candidate(
        intent="ancestry",
        score=cat.PRIOR["reverse_hierarchy"] + cat.W_EXPLICIT_PHRASE + cat.W_ENTITY,
        evidence=["ancestry_phrase", f"subject:{subject_level}"],
        build=lambda: QueryPlan(
            action="ancestry", level=subject_level, entity_value=subject_value,
        ),
    )


def _score_advisor_metric(ctx: _Intent) -> _Candidate | None:
    """"connects of Shehryar Abbasi" — ONE metric for ONE person.

    Why this is its own intent rather than a formatting flag on
    advisor_profile: the profile reply is a SUPERSET of the answer. It
    contains the requested number, so it never reads as wrong, and the
    user has to find their metric among team, manager, targets and
    everything else. That is precisely the situation
    W_SPECIFIC_CONSTRAINT exists for — a competing reading that silently
    drops a constraint the user stated and answers with more than was
    asked.

    The discriminator needs no new vocabulary: `ctx.metric` is already
    resolved from the ontology for every query, and it is None for
    "tell me about X" / "who is X" / "show X profile". A named metric IS
    the metric intent; its absence IS the profile intent.
    """
    if not ctx.entities.get("advisor_name") or not ctx.metric:
        return None
    # Phase 7 removed the ranking/comparison suppression here.
    #
    # This used to `return None` whenever a ranking word, a comparison
    # phrase, a reverse role or a relation appeared — a SCORER declining
    # on a RIVAL intent's evidence. "Top revenue for Omar Farooq" then
    # produced exactly one candidate, so the ranking was never a contest
    # and the reply named a different advisor entirely. A candidate that
    # is never proposed cannot lose, and cannot be explained.
    #
    # Proposing is now unconditional on rival evidence;
    # intent_precedence.PRECEDENCE decides who wins, in one table. The
    # relation/reverse cases stay excluded THERE (they are genuinely
    # different questions about a person, not a measure of them), so no
    # behaviour is conceded — only the place the decision is made moves.
    if ctx.is_reverse or ctx.is_relational:
        return None
    # A GROUP was named too ("the connects of zonal head Salman Arshad and
    # his team"), so the question is about that group's people, not about
    # one person's own number.
    if ctx.group_entity() is not None:
        return None
    if ctx.entities.get("advisor_match_score", 1.0) < 0.6:
        return None
    # "performance of X" resolves a metric key but is asking how somebody
    # is doing — see cat.GENERAL_INTEREST_SYNONYMS. The profile answers
    # that; one percentage does not.
    evidence = resolve_metric_evidence(ctx.text)
    if evidence is not None and evidence[1] in cat.GENERAL_INTEREST_SYNONYMS:
        return None

    score = cat.PRIOR["advisor_metric"] + cat.W_SPECIFIC_CONSTRAINT
    evidence = ["advisor_name_present", f"metric:{ctx.metric}"]
    if ctx.entities.get("advisor_wid"):
        score += cat.W_IDENTITY
        evidence.append("identity_resolved")

    metric = ctx.metric
    return _Candidate(
        intent="advisor_metric", score=score, evidence=evidence,
        build=lambda: _advisor_plan(ctx.entities, action="advisor_metric", metric=metric),
    )


def _score_group_metric(ctx: _Intent) -> _Candidate | None:
    """"Blue Area revenue" — one group's own figure for one measure.

    Phase 7 made this a first-class intent. It had none: `advisor_metric`
    expressed "one measure, one person" and nothing expressed "one
    measure, one GROUP", so every such query was proposed only as a
    leaderboard. The ANSWERS were right — Phase 1 gave the group the
    subject level and Phase 3 rendered one row as a metric value — but
    they were right by downstream compensation, which means a change to
    either silently returns the member list the audit found originally.

    Proposes whenever a group and a measure are both present. Whether it
    WINS is intent_precedence's decision: a ranking word or an explicit
    inner level word means the query wants what is inside the group, and
    the table sends those to the leaderboard.
    """
    group = ctx.group_entity()
    if group is None or not ctx.metric:
        return None
    if ctx.entities.get("advisor_name"):
        return None  # a person was named too — not this group's own figure

    level, value = group
    if not is_answerable(ctx.metric, level):
        return None

    score = cat.PRIOR["leaderboard"] + cat.W_SPECIFIC_CONSTRAINT + cat.W_ENTITY
    evidence = [f"group_entity:{level}", f"metric:{ctx.metric}"]
    metric = ctx.metric
    return _Candidate(
        intent="group_metric", score=score, evidence=evidence,
        # Built as a leaderboard scoped to the group: one row, the
        # group's own figure. The PLAN shape is deliberately unchanged —
        # this phase makes the intent explicit, it does not rebuild the
        # execution path that already produces the right number.
        build=lambda: QueryPlan(
            # A first-class ACTION, so "which intent won" is answerable
            # from the plan rather than inferred from a leaderboard that
            # happens to return one row. The execution shape is
            # deliberately identical — plan_to_ir builds the same scoped
            # QueryIR, and the response planner still renders one row as
            # a metric value. This phase makes the classification
            # explicit; it does not rebuild a path that already produces
            # the right number.
            action="group_metric", level=level, entity_value=value, metric=metric,
            limit=ctx.entities.get("limit", 10),
            ascending=_sort_signal(ctx.q, metric),
        ),
    )


def _score_advisor_profile(ctx: _Intent) -> _Candidate | None:
    """"Tell me about X" — one person's own record."""
    if not ctx.entities.get("advisor_name"):
        return None
    if ctx.entities.get("advisor_match_score", 1.0) < 0.6:
        return None

    score = cat.PRIOR["advisor_profile"]
    evidence = ["advisor_name_present"]
    if ctx.entities.get("advisor_wid"):
        score += cat.W_IDENTITY
        evidence.append("identity_resolved")
    # A ranking phrasing argues AGAINST a single-person lookup — "top
    # advisors" is not a request for one profile.
    if ctx.has_ranking_strong:
        return None
    return _Candidate(
        intent="advisor_profile", score=score, evidence=evidence,
        build=lambda: _advisor_plan(ctx.entities),
    )


def _score_attendance(ctx: _Intent) -> _Candidate | None:
    """Advisors filtered by attendance status.

    Carries W_SPECIFIC_CONSTRAINT because every competing reading would
    DROP the status: "late advisors in Blue Area" also matches roster
    phrasing, and a roster answers with all 54 people in the team — a
    superset that reads as authoritative and is simply wrong."""
    status = ctx.entities.get("attendance_status")
    if not status:
        return None
    # A RANKING by an attendance measure is not a filter on a status.
    # "who has the most late arrivals" resolves late_count and asks for
    # it ranked; this scorer's W_SPECIFIC_CONSTRAINT exists for readings
    # that would DROP the status, and a leaderboard on late_count drops
    # nothing — it is the same information, ordered. Without this the
    # filter reading won at 0.98 and answered with an unordered list.
    if ctx.has_ranking_strong and ctx.metric:
        return None

    score = cat.PRIOR["attendance_filter"] + cat.W_EXPLICIT_PHRASE + cat.W_SPECIFIC_CONSTRAINT
    evidence = [f"attendance_status:{status}", "honours_constraint_others_drop"]
    if ctx.entities.get("team"):
        score += cat.W_ENTITY
        evidence.append("group_entity:team")

    return _Candidate(
        intent="attendance_filter", score=score, evidence=evidence,
        build=lambda: QueryPlan(
            action="attendance_filter", level="advisor",
            entity_value=ctx.entities.get("team"), reason=status,
        ),
    )


def _score_entity_summary(ctx: _Intent) -> _Candidate | None:
    """"How is Blue Area doing" — a group's aggregate metrics."""
    if ctx.metric:
        # a resolved metric means the user named a measure; that is
        # leaderboard evidence, and a summary would ignore it
        return None
    group = ctx.group_entity()
    if not group:
        return None
    level, value = group

    score = cat.PRIOR["entity_summary"] + cat.W_ENTITY
    evidence = [f"group_entity:{level}"]

    def _build() -> QueryPlan:
        if level in ("team", "company"):
            return QueryPlan(action="summary", level=level, entity_value=value)
        return QueryPlan(action="breakdown", level=level, entity_value=value, flat=ctx.is_flat)

    return _Candidate(intent="entity_summary", score=score, evidence=evidence, build=_build)


def _without_comparators(q: str) -> str:
    """`q` with comparator phrases blanked out, for the ranking scans.

    Derived from the comparator registry (comparators.phrases()), so a
    newly declared threshold phrase is shadowed automatically instead of
    quietly re-creating this bug for the next word that happens to
    contain a direction term.
    """
    return token_match.mask(q, comparators.phrases())


def _sort_signal(q: str, metric_key: str | None) -> bool | None:
    """Did the user name a sort direction, and which?

    True = ascending, False = descending, None = they didn't say — in
    which case the metric's own polarity decides downstream.

    "worst" is resolved against that polarity here because it is a
    QUALITY word: the worst overdue is the HIGHEST, the worst revenue is
    the LOWEST. Treating it as a synonym for "lowest" (which is what the
    single pre-Phase-2 list did) got one of those backwards.
    """
    # F9, part 1: token_match, not substring containment — "most" is
    # inside "almost".
    #
    # F9, part 2: comparator phrases are masked first. "at least 80%"
    # contains the whole token "least", so token-awareness alone cannot
    # separate it — but it is a threshold, not a request for the
    # minimum. Without this, "advisors with at least 80% achievement"
    # ranked ASCENDING and led with the lowest achievers above the
    # threshold the user had just set.
    q = _without_comparators(q)
    if token_match.contains_any(q, cat.ASCENDING_ABSOLUTE):
        return True
    if token_match.contains_any(q, cat.DESCENDING_ABSOLUTE):
        return False
    if token_match.contains_any(q, cat.WORST_RELATIVE):
        # The BAD end, resolved against polarity. For a higher-is-better
        # metric that is ascending; for overdue or late arrivals it is
        # descending. Reversing the sort instead would make "bottom 5 by
        # overdue" show the advisors with the FEWEST overdue items — the
        # best performers, labelled as the worst.
        return not lower_is_better(metric_key)
    return None


def _score_leaderboard(ctx: _Intent) -> _Candidate | None:
    """"Top 5 advisors by revenue" — a metric ranking.

    A STRONG ranking word with no metric named still ranks: "top 5
    advisors" is unambiguous about wanting five, ranked — only the
    measure is unstated, and it falls back to DEFAULT_RANKING_METRIC.
    Requiring a metric meant the leaderboard wasn't a candidate at all
    for those queries, so "top 5 advisors in Blue Area" lost to the
    roster reading and returned all 54 advisors — dropping both "top"
    and "5", which is the opposite of what was asked."""
    metric_key = ctx.metric
    defaulted = False
    if not metric_key and ctx.metric_intent.unresolved:
        # F6. The user NAMED a measure and it could not be resolved, even
        # after fuzzy and semantic widening. Defaulting here is what made
        # "which BCM has the highest CR%" a revenue leaderboard: a real
        # answer to a question nobody asked, formatted exactly like a
        # right one.
        #
        # Scored at what the leaderboard WOULD have scored, and returned
        # INSTEAD of it, so this replaces the ranking reading without
        # overriding a genuinely different intent — "all advisors in Blue
        # Area by widget velocity" is still a roster, because the roster
        # candidate outscores this the same way it outscored the
        # leaderboard.
        named = ctx.metric_intent.named_text
        reason = ctx.metric_intent.reason
        return _Candidate(
            intent="clarify_metric",
            score=cat.PRIOR["leaderboard"] + cat.W_METRIC + (
                cat.W_RANKING_STRONG + cat.W_RANK_METRIC_COMBO
                if ctx.has_ranking_strong else 0.0
            ),
            evidence=[f"metric_named_but_unresolved:{named}"],
            # `reason` on the plan is the human-readable refusal when the
            # registry knows WHY (a declared-but-uncomputable rate);
            # `entity_value` carries the raw phrase either way.
            build=lambda: QueryPlan(
                action="clarify_metric", entity_value=named, reason=reason or named,
            ),
        )
    if not metric_key and ctx.has_ranking_strong:
        # No measure named at all — the ranking intent is explicit and
        # only the measure is unstated, so filling it is completing the
        # request rather than overriding it.
        metric_key = cat.DEFAULT_RANKING_METRIC
        defaulted = True
    if not metric_key:
        return None

    ctx = replace(ctx, metric=metric_key) if defaulted else ctx
    metric_def = METRICS.get(ctx.metric)
    if not metric_def:
        return _Candidate(
            intent="leaderboard", score=cat.PRIOR["leaderboard"],
            evidence=[f"metric_not_in_ontology:{ctx.metric}"],
            build=lambda: QueryPlan(
                action="unresolved", reason=f"metric '{ctx.metric}' not in ontology"
            ),
        )

    score = cat.PRIOR["leaderboard"] + cat.W_METRIC
    evidence = [f"metric:{ctx.metric}"]

    # A stated THRESHOLD is the same shape of evidence attendance_status
    # already carries: a constraint every competing reading would drop.
    # "advisors in Team 70 with achievement between 60 and 80" matches
    # roster phrasing, and a roster ignores the range and returns the
    # whole team — a superset that reads as authoritative and is simply
    # wrong. Only a metric-bearing reading can honour a threshold at all,
    # since a threshold needs a measure to compare against.
    #
    # W_ENTITY comes with it because this reading honours the named group
    # too (plan_to_ir emits the team/company filter alongside), so it is
    # not conceding that evidence to the roster either.
    if ctx.entities.get("thresholds"):
        score += cat.W_SPECIFIC_CONSTRAINT
        evidence.append("honours_threshold_others_drop")
        if ctx.group_entity() is not None:
            score += cat.W_ENTITY
            evidence.append("group_entity_filtered")
    if defaulted:
        # Full weight deliberately: the RANKING intent is explicit and
        # unambiguous, only the measure was inferred. Recorded as
        # evidence, and the reply header names the metric, so the default
        # is visible rather than silent.
        evidence.append(f"default_metric:{ctx.metric}")
    if ctx.has_ranking_strong:
        score += cat.W_RANKING_STRONG + cat.W_RANK_METRIC_COMBO
        evidence += ["ranking_strong", "ranking+metric"]
    elif ctx.has_ranking_weak:
        score += cat.W_RANKING_WEAK
        evidence.append("ranking_weak")

    # Phase 2: subject_level.decide() owns this. It used to read
    # `detect_level(...) or metric_def.primary_level`, which never
    # consulted the grounded entity — so "Downtown's pipeline value"
    # answered with a list of advisors filtered to Downtown instead of
    # Downtown's figure. The named subject now outranks the metric's
    # default; the default is reached only when no subject was named.
    entity_level, entity_value = subject_level.entity_level_from(ctx.entities)
    decision = subject_level.decide(
        level_word=cat.detect_level(ctx.level_q),
        entity_level=entity_level,
        entity_value=entity_value,
        metric_default=metric_def.primary_level,
        has_ranking=ctx.has_ranking_strong,
    )
    level = decision.level
    if not is_answerable(ctx.metric, level):
        # The chosen level has no resolver for this metric — degrade to
        # the metric's primary level rather than failing. is_answerable
        # (not a bare entity_levels check) because the new hierarchy
        # levels are answerable via the compiler's generic rollup
        # fallback. Recorded as its own evidence so a degraded level is
        # never mistaken for a chosen one.
        evidence.append(f"level_unanswerable:{level}->{metric_def.primary_level}")
        level = metric_def.primary_level
    evidence.append(f"level:{level}")
    routing.decide("Level", level, decision.trace())

    return _Candidate(
        intent="leaderboard", score=score, evidence=evidence,
        build=lambda: QueryPlan(
            action="leaderboard", level=level, metric=ctx.metric,
            limit=ctx.entities.get("limit", 10),
            ascending=_sort_signal(ctx.q, ctx.metric),
        ),
    )


# Declaration order is the deterministic tie-break for EQUAL scores only.
# It is not the priority mechanism — PRIOR in intent_catalog.py is, and
# evidence overrides that.
_SCORERS: tuple[Callable[[_Intent], _Candidate | None], ...] = (
    _score_clarify_ambiguous,
    _score_trend,
    _score_comparison,
    _score_roster,
    _score_ancestry,
    _score_reverse_hierarchy,
    _score_hierarchy,
    _score_attendance,
    # Before advisor_profile: on the equal-score tie that cannot happen
    # today (metric evidence always separates them) declaration order
    # would still favour the more specific reading.
    _score_advisor_metric,
    _score_group_metric,
    _score_advisor_profile,
    _score_entity_summary,
    _score_leaderboard,
)


def _fallback(ctx: _Intent) -> QueryPlan:
    """No intent scored. Answer about whatever entity was found, advisor
    first, then groups — the pre-scoring fallback order, preserved."""
    if ctx.entities.get("advisor_name"):
        return _advisor_plan(ctx.entities)
    group = ctx.group_entity()
    if group:
        level, value = group
        if level in ("team", "company"):
            return QueryPlan(action="summary", level=level, entity_value=value)
        return QueryPlan(action="breakdown", level=level, entity_value=value, flat=ctx.is_flat)
    return QueryPlan(action="unresolved", reason="no metric or entity matched")


def score_intents(text: str, entities: dict) -> tuple[_Intent, list[_Candidate]]:
    """Every applicable intent with its score, best first. Exposed for
    the audit tooling and tests — the planner itself just takes [0]."""
    q = text.lower()
    ranking_q = _without_comparators(q)
    intent = metric_intent.detect(text, entities)
    # Blank the metric's own phrase before level detection — see
    # _Intent.level_q.
    evidence = resolve_metric_evidence(text)
    level_q = token_match.mask(q, [evidence[1]]) if evidence else q
    ctx = _Intent(
        text=text,
        q=q,
        entities=entities,
        metric=intent.key,
        metric_intent=intent,
        level_q=level_q,
        # F9, on the evidence side. Same two problems, same two fixes:
        # "almost" no longer contributes "most", and a comparator phrase
        # no longer contributes the direction word inside it.
        # "at least"/"almost" manufactured `ranking_strong`, worth 0.25
        # plus the 0.25 rank+metric combo — enough to move a threshold
        # query onto a different intent entirely.
        has_ranking_strong=token_match.contains_any(ranking_q, cat.RANKING_STRONG),
        has_ranking_weak=token_match.contains_any(ranking_q, cat.RANKING_WEAK),
        is_flat=token_match.contains_any(q, cat.FLAT_KEYWORDS),
        is_relational=bool(cat.RELATIONAL_RE.search(q)),
        is_reverse=bool(cat.REVERSE_RE.search(q)),
        is_roster=bool(cat.ROSTER_RE.search(q)),
        is_comparison=bool(cat.COMPARISON_RE.search(q)),
    )

    candidates: list[_Candidate] = []
    for scorer in _SCORERS:
        candidate = scorer(ctx)
        if candidate is not None:
            candidates.append(candidate)

    # stable sort by score desc; equal scores keep declaration order
    candidates.sort(key=lambda c: -c.score)
    return ctx, candidates


def _carry_extracted_constraints(plan: QueryPlan, entities: dict, ctx) -> QueryPlan:
    """Copy the constraints the user stated onto whichever plan won.

    Done HERE rather than inside each scorer's build lambda for two
    reasons. It is one place instead of ten, so a constraint cannot be
    carried by the leaderboard and silently dropped by the summary — the
    shape of the F4/F8 bugs. And an intent that grows a period-aware or
    threshold-aware dispatch later inherits the value instead of having
    to re-read `entities`, which is how the two copies drift apart in the
    first place.

    Only ever ADDS information a scorer didn't already set, so no
    existing build is overridden.
    """
    if plan.period is None:
        plan.period = entities.get("period")
    if not plan.metrics:
        # Every measure the query named, from the module that already
        # decided which one is primary — so the list and `plan.metric`
        # cannot disagree about what was asked for. Carried here for the
        # same reason `period` is: one place, so a constraint cannot be
        # kept by one intent and dropped by another.
        plan.metrics = list(ctx.metric_intent.keys)
    if not plan.thresholds:
        # Copied, not aliased: a caller mutating plan.thresholds must not
        # reach back into the extractor's dict.
        plan.thresholds = [dict(t) for t in entities.get("thresholds") or []]
    return plan


def build_query_plan(text: str, entities: dict) -> QueryPlan:
    ctx, candidates = score_intents(text, entities)

    if not candidates:
        return _carry_extracted_constraints(_fallback(ctx), entities, ctx)

    # Phase 7: PROPOSE then RANK. The scorers above only propose; which
    # one wins is intent_precedence's decision, from the evidence the
    # query actually contains. Score order is what it falls back to when
    # no precedence rule applies, so every intent the table does not
    # mention behaves exactly as it did before.
    ranking = intent_precedence.rank(candidates, _evidence_for(ctx))
    winner = ranking.winner

    plan = winner.build()
    plan.intent_score = round(winner.score, 3)
    plan.intent_evidence = list(winner.evidence)
    if ranking.rejected:
        plan.runner_up = ranking.rejected[0][0]
    routing.decide("Intent", winner.intent, ranking.trace())
    return _carry_extracted_constraints(plan, entities, ctx)


def _evidence_for(ctx: _Intent) -> "intent_precedence.Evidence":
    """The query's facts, read from the components that already own them.

    Nothing is re-derived here: subjects come from entity extraction,
    the measure from metric_intent, the phrases from intent_catalog. The
    Phase 6 audit's finding was that these facts existed and were never
    consulted when choosing an intent — this is the consultation.
    """
    group = ctx.group_entity()
    groups = len(ctx.all_group_entities())

    # AMBIGUITY IS NOT MULTIPLICITY. One reference matching several people
    # ("Advisor 20" against a roster of "Advisor 1..40") is not the user
    # naming several subjects — it is one subject we cannot pin down, and
    # clarify_person owns that. Counting it as a named advisor let a
    # spurious grounding promote advisor_metric over a legitimate
    # leaderboard for "top 20 advisors by connects".
    advisors = len(ctx.entities.get("advisor_multi") or []) or (
        1 if ctx.entities.get("advisor_name") else 0
    )
    return intent_precedence.Evidence(
        named_advisors=advisors,
        named_groups=groups,
        metric=ctx.metric is not None,
        ranking_phrase=ctx.has_ranking_strong,
        comparison_phrase=ctx.is_comparison,
        roster_phrase=ctx.is_roster,
        relation_phrase=ctx.is_relational,
        reverse_phrase=ctx.is_reverse,
        level_word=cat.detect_level(ctx.level_q),
        group_level=group[0] if group else None,
        ambiguous_subject=bool(ctx.entities.get("advisor_ambiguous")),
    )
