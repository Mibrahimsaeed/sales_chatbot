"""The routing decision layer — WHERE a query goes, decided once.

Phase 1 (routing pipeline refactor) exists because three defects all had
the same shape: a routing decision was made by a component that could not
yet see the information the decision depended on, and nothing downstream
could undo it.

  P1  classify_intent() ran BEFORE extract_entities() and was handed a
      hardcoded empty dict, so its "is this a generic attendance sweep or
      a specific metric question?" guard had only regex to work with. It
      guessed wrong for every person-scoped attendance/login question,
      and `attendance_rate` / `login_rate` — both fully bound and
      verified computing — were unreachable.

  P2  a declared-but-unavailable measure ("connect %") produced its
      written explanation only on the branch reachable when NO person
      resolved. Name the person in full and the same question degraded to
      an advisor profile card: the better-specified query got the worse
      answer.

  P3  `metric_def.primary_level` was applied whenever no level word
      appeared in the text, which silently conflated "the user named no
      subject" with "the user named a subject we failed to resolve". The
      second case answered about the person's TEAM without saying so.

The fix in all three cases is ordering plus an explicit signal: extract
first, decide second, and let the shortcut handlers run only as a
FALLBACK once everything better has declined. This module holds those
predicates so the decision exists in exactly one place — the recurring
defect in this codebase is the same fact declared twice with nothing
forcing the copies to agree, and a routing rule inlined at its branch is
exactly that kind of second copy.

Every predicate here is pure: it reads text and the entity dict and
returns a decision. Nothing in this module queries the database, calls an
LLM, or mutates its inputs, which is what makes routing reproducible —
the same query with the same entities always routes the same way.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

from app.core import audit
from app.llm import metric_aliases

# ---------------------------------------------------------------------
# Routing trace
# ---------------------------------------------------------------------
#
# audit.decision() already records branch points, but it is gated on
# AUDIT_ENABLED and writes to a log file — it cannot be asserted against
# in a test, and it is off in production by default. The routing trace is
# a always-on, in-memory, ordered record of the routing decisions ONLY,
# cheap enough to leave on (a handful of small tuples per request).
#
# Both sinks are written from decide() rather than from each call site,
# so a new routing decision cannot land in one and be missing from the
# other.


@dataclass(frozen=True)
class Step:
    """One routing decision: what stage, what it chose, and why."""

    stage: str
    chose: str
    why: str = ""


@dataclass
class RoutingTrace:
    query: str = ""
    steps: list[Step] = field(default_factory=list)

    def render(self) -> str:
        """The linear debug view, one decision per line."""
        if not self.steps:
            return "(no routing decisions recorded)"
        lines = [f"Query = {self.query!r}"] if self.query else []
        for step in self.steps:
            line = f"{step.stage} = {step.chose}"
            if step.why:
                line += f"   [{step.why}]"
            lines.append(line)
        return "\n  ↓\n".join(lines)

    def chose(self, stage: str) -> Optional[str]:
        """What `stage` decided, or None if it never ran. Lets a test
        assert on one decision without depending on the trace's shape."""
        for step in self.steps:
            if step.stage == stage:
                return step.chose
        return None


_trace: ContextVar[Optional[RoutingTrace]] = ContextVar("routing_trace", default=None)


def start_trace(query: str = "") -> RoutingTrace:
    """Begin recording. Called once per resolve() at depth 0."""
    trace = RoutingTrace(query=query)
    _trace.set(trace)
    return trace


def current_trace() -> Optional[RoutingTrace]:
    return _trace.get()


def decide(stage: str, chose: str, why: str = "") -> None:
    """Record one routing decision to both sinks.

    Never raises: a trace that fails must not be able to fail the
    request it is describing."""
    trace = _trace.get()
    if trace is not None:
        # A decision can be reached more than once per request: the
        # planner scores candidates in several passes, and on the
        # LLM-disabled path semantic_parser re-plans through
        # build_query_plan to build its degrade IR. Identical (stage,
        # choice, reason) triples carry no information the first one
        # didn't, and they are not necessarily adjacent — a Planner step
        # lands between the two Level steps — so the check is against the
        # whole trace. A CHANGED value still appends: that is a real
        # second decision, and seeing a level move mid-request is exactly
        # what this trace exists for.
        step = Step(stage, chose, why)
        if step not in trace.steps:
            trace.steps.append(step)
    try:
        audit.decision("routing", chose, why or chose)
    except Exception:  # pragma: no cover - audit is best-effort
        pass


# ---------------------------------------------------------------------
# P2 — metric availability
# ---------------------------------------------------------------------


def unavailable_metric(text: str) -> Optional[metric_aliases.AliasMatch]:
    """The declared-but-uncomputable measure this text names, if any.

    Checked BEFORE any other routing decision. `metric_aliases.resolve()`
    is already token-aware and scans the whole string, and it returns an
    AliasMatch with `metric=None` exactly for the UNAVAILABLE set — so
    "I know this measure and cannot compute it" is distinguishable from
    "I have never heard of this measure" (no AliasMatch at all), which is
    the distinction P2 was collapsing.
    """
    match = metric_aliases.resolve(text)
    if match is not None and match.metric is None:
        return match
    return None


def explain_unavailable(match: metric_aliases.AliasMatch) -> str:
    """The user-facing reply for an unavailable measure. Delegates to the
    registry so the wording has one source."""
    return metric_aliases.explain(match)


# ---------------------------------------------------------------------
# P1 — shortcuts are a fallback, never the primary route
# ---------------------------------------------------------------------

# A phrase asking for a RATE rather than a sweep. The distinction matters
# because the bare word "attendance" is itself an `attendance_rate`
# synonym: gating the shortcut on "did any metric resolve" would block
# every generic "show me attendance issues" too, which is the trap
# intent_detector's own docstring documents. Gating on the matched
# PHRASE's shape keeps generic sweeps on the shortcut and sends explicit
# rate questions to the planner.
_RATE_PHRASE = re.compile(r"(%|\bpercent\b|\bpercentage\b|\brate\b|\bratio\b|\bpct\b)")


def names_a_rate(text: str) -> bool:
    """Does this text ask for a percentage/rate form of a measure?

    Read from the matched alias phrase rather than the raw text, so an
    incidental "rate" elsewhere in a sentence cannot flip the decision.
    """
    match = metric_aliases.resolve(text)
    if match is None:
        return False
    return bool(_RATE_PHRASE.search(match.phrase.lower()))


def shortcut_allowed(text: str, entities: dict) -> tuple[bool, str]:
    """May a canned shortcut handler answer this message?

    Returns (allowed, why) — the reason is recorded either way, because
    "the shortcut was skipped and here is what outranked it" is the piece
    that makes a mis-route diagnosable.

    A shortcut may answer only what nothing better can. Two signals mean
    something better exists, and NEITHER was available at the old call
    site: a resolved person (the question is about someone specific), and
    an explicit rate/percentage phrase (the question names a measure the
    canned handler cannot express).
    """
    if entities.get("advisor_wids"):
        name = entities.get("advisor_name") or "an advisor"
        return False, (
            f"entity extraction resolved {name!r} — a person-scoped question "
            "belongs to the planner, which can bind the named metric to that "
            "person; the canned handler can express neither"
        )
    if names_a_rate(text):
        match = metric_aliases.resolve(text)
        phrase = match.phrase if match else text
        return False, (
            f"the text names a rate/percentage measure ({phrase!r}) — the canned "
            "handler answers with a status sweep and cannot express a rate"
        )
    return True, "no advisor resolved and no rate/percentage measure named"


# ---------------------------------------------------------------------
# P3 — "no subject" is not the same as "subject we could not resolve"
# ---------------------------------------------------------------------

# A capitalised possessive is the strong, low-false-positive signal that
# the user named a HUMAN subject: "Ahmed's attendance", "Zainab Malik's
# CR". Deliberately narrow — this decides whether to ask a clarifying
# question, and asking one when the user named nobody would be worse than
# the defect it fixes.
# A capitalised possessive: "Waqar Haider's", "Central Region's".
#
# The trailing group also accepts a PARENTHESISED qualifier, because real
# master-sheet names carry them — "Omer Sandhu (Virtual)". Without it the
# capture stopped at the bracket, matched nothing, and the whole
# unresolved-subject refusal was skipped for every such person: their
# queries fell through to a global ranking instead of "I couldn't find
# anyone called that". Kept to a bracketed word rather than arbitrary
# punctuation so the pattern still ends at a sentence boundary.
_POSSESSIVE = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z/]+)*(?:\s+\([A-Za-z][A-Za-z\s]*\))?)'s\b"
)


def _trim_span(value: str) -> str:
    """Strip the parts of a captured possessive that are not the name.

    The regex walks backwards over every capitalised token, which in real
    queries swallows three things that are not part of a person's name.
    The behavioural audit caught all three in production phrasings:

        "BCM Usman Ghani's group"      -> captured "BCM Usman Ghani"
        "Compare Fawad Hafeez's group" -> captured "Compare Fawad Hafeez"
        "Central Region's 1-unit"      -> captured "Central Region"

    Each produced "I couldn't find anyone called ..." for a query naming
    a perfectly resolvable subject. Role prefixes come from the hierarchy
    registry so a new level is handled here for free; the leading verbs
    are sentence openers that are capitalised only because they start the
    sentence.
    """
    from app.llm import hierarchy

    words = value.split()

    # A leading verb, capitalised only by sentence position.
    while words and words[0].lower() in _SENTENCE_OPENERS:
        words = words[1:]

    # A role prefix: "BCM Usman Ghani", "Unit Head Tariq Mehmood".
    changed = True
    while changed and len(words) > 1:
        changed = False
        for keywords in hierarchy.LEVEL_KEYWORDS.values():
            for keyword in keywords:
                parts = keyword.split()
                if len(words) > len(parts) and [w.lower() for w in words[:len(parts)]] == parts:
                    words = words[len(parts):]
                    changed = True
                    break
            if changed:
                break

    return " ".join(words)


# Capitalised only because they open the sentence. Not names.
_SENTENCE_OPENERS = frozenset({
    "compare", "show", "list", "give", "tell", "what", "who", "which",
    "how", "find", "get", "display", "is", "are", "was", "were",
})


def _any_subject_grounded(entities: dict) -> bool:
    """Did extraction ground ANY entity this query could be about?

    Read from the hierarchy registry rather than a fixed key list, so a
    level added later counts here for free — the same derivation
    _known_non_person below already uses, and for the same reason.

    Deliberately generous: one grounded entity of any kind is enough. The
    question this answers is not "is the traversal correct" but "does the
    traversal have a source at all", and a stricter test would start
    refusing the possessive queries that work today.
    """
    from app.llm import hierarchy

    if entities.get("advisor_wids"):
        return True
    for level in hierarchy.HIERARCHY_LEVELS:
        if entities.get(level):
            return True
        if entities.get(hierarchy.LEVEL_ENTITY_KEYS.get(level, f"{level}s")):
            return True
    return False


def _known_non_person(value: str, entities: dict) -> bool:
    """Is this capitalised possessive something OTHER than a person?

    Checked against the live registries rather than a hardcoded stop
    list, so a hierarchy level or metric added later is excluded here for
    free.
    """
    from app.llm import hierarchy

    lowered = value.lower()

    # Anything the extractor already grounded ("Blue Area's achievement").
    #
    # Every grounded value is scanned, not a fixed list of singular keys.
    # Two reasons, both found by the behavioural audit: the extractor
    # grounds region="Central" while the query says "Central Region's",
    # so the match must be fuzzy at the edges; and a COMPARISON grounds
    # its second subject only in the plural key (zonal_heads = [A, B]
    # with zonal_head = A), so reading singulars alone refused
    # "Compare Fawad Hafeez's group and Adeel Aslam's group" on its
    # second name.
    for key, grounded in entities.items():
        if key.startswith("_") or key.endswith("_matches"):
            continue
        values = grounded if isinstance(grounded, (list, tuple)) else [grounded]
        for value in values:
            if not isinstance(value, str) or not value:
                continue
            g = value.lower()
            if g == lowered or g in lowered or lowered in g:
                return True

    # A hierarchy level word, alone ("the company's target") or trailing
    # a group name ("Central Region's", "GRO Team's").
    for level in hierarchy.HIERARCHY_LEVELS:
        names = {level, level.replace("_", " ")}
        label = hierarchy.label_for(level)
        if label:
            names.add(label.lower())
        for keywords in (hierarchy.LEVEL_KEYWORDS.get(level, []),):
            names.update(k.lower() for k in keywords)
        for name in names:
            if lowered == name or lowered.endswith(" " + name):
                return True

    # A measure ("this month's Revenue").
    if metric_aliases.resolve(lowered) is not None:
        return True

    return False


def unresolved_subject(text: str, entities: dict) -> Optional[str]:
    """The person this query names that identity resolution could not
    ground, or None.

    None covers BOTH "named nobody" and "named someone we resolved" — the
    caller only needs to act on the third case. Returning the name rather
    than a bool lets the clarification quote it back, which is the whole
    point: "I don't know who Ahmed is" is useful, "I couldn't resolve
    that" is not.

    Two guards keep this narrow, both learned from breaking real queries:

      * A MEASURE must be named. The defect being fixed is a metric
        question silently answered at group level; a query naming no
        metric cannot hit it. This is what keeps "Adeel Dogar's advisors"
        (a roster request) out.

      * A RELATION reference disqualifies it. "show Adeel Dogar's team"
        is a hierarchy traversal, and managers are deliberately NOT in
        the advisor gazetteer — they are grounded later, by the planner,
        against the manager columns. Treating "not an advisor" as "not a
        person" here would break every possessive traversal query.
        reference_parser owns that pattern already, so it is asked
        rather than re-implemented.
    """
    from app.llm import reference_parser

    if entities.get("advisor_wids"):
        return None  # resolved (or ambiguous — clarify_person owns that)

    if metric_aliases.resolve(text) is None:
        return None  # no measure named — the primary_level defect can't fire

    # A relation traversal ("X's team") is grounded downstream, by the
    # planner, against the manager columns — but ONLY if its source
    # grounded to something. When it did not, there is no downstream: the
    # person silently disappears, the word "team" is still in the text,
    # and the planner builds a perfectly valid UNFILTERED team
    # leaderboard. "Omer Sandhu (Virtual)'s team pipeline" answered with
    # a ranking of all nine teams, confidently, having dropped the only
    # subject the question had.
    #
    # So the escape now asks whether the traversal has anything to
    # traverse FROM. Nothing grounded means the possessive names someone
    # this system cannot find, which is exactly the case the rest of this
    # function exists to refuse — and refusing it here is the same answer
    # the non-possessive phrasing already gives.
    if reference_parser.parse(text) and _any_subject_grounded(entities):
        return None

    for raw in _POSSESSIVE.findall(text):
        candidate = _trim_span(raw)
        if candidate and not _known_non_person(candidate, entities):
            return candidate
    return None


def validate_route(ir) -> Optional[str]:
    """The last gate before the compiler: is this IR answerable as asked?

    Returns an explanation when it is not, None when it is, and is the
    single place every kind="ir" exit passes through (nlu_pipeline.
    _ir_resolution) so a fourth exit added later cannot bypass it.

    WHO OWNS WHAT — this gate deliberately does NOT re-check any of these,
    because a second copy of a rule that drifts from the first is the
    defect this whole refactor exists to remove:

      advisor resolved      unresolved_subject() above, then the planner's
                            clarify_person branch for a genuine ambiguity.
      metric resolved       ir_validator.validate_ir(), which also owns the
                            confidence floor and the fuzzy recovery of a
                            near-miss key.
      metric computable     unavailable_metric() above, from the UNAVAILABLE
                            registry.
      hierarchy valid       QueryIR.subject_level is a Literal over
                            hierarchy's own level names — pydantic rejects
                            an invalid level at construction.
      period valid          query_compiler._effective_metric(), which maps a
                            requested period onto the metric's period family
                            and returns None when no member covers it; the
                            response layer then explains in the period's own
                            terms ("I don't have year-to-date figures for
                            X — I hold MTD totals").

    That leaves exactly one invariant unowned. ir_validator's metric-key
    check runs only for leaderboard/comparison/filtered_list, so an IR with
    any OTHER intent can still carry a key that is not in the ontology —
    from a patched prior IR, or an LLM parse for an intent outside that
    set. The compiler would find no binding and return an empty result,
    which reads to the user as "no data" rather than "that isn't a
    measure I have".
    """
    from app.llm.ir_validator import _NON_METRIC_FILTER_FIELDS
    from app.llm.metric_ontology import METRICS

    # EVERY measure the IR names, not just the primary two fields. The
    # narrower read let an invented key reach the compiler unchallenged
    # whenever it sat in `metrics[]` or in a filter — the compiler then
    # found no binding and returned an empty result, which reads as "no
    # data" rather than "that isn't a measure I have". That is the same
    # gap this gate exists to close, one field over.
    candidates = list(ir.metric_keys())
    if ir.sort and ir.sort.metric:
        candidates.append(ir.sort.metric)
    candidates.extend(f.field for f in ir.filter_leaves())

    for metric_key in dict.fromkeys(candidates):
        # A filter field may legitimately be an ENTITY level rather than
        # a measure ("team = Blue Area"); those are validated separately
        # by ir_validator against the hierarchy, so only reject a name
        # that is neither.
        if not metric_key or metric_key in METRICS:
            continue
        if metric_key in _NON_METRIC_FILTER_FIELDS:
            continue
        return (
            f"I don't have a measure called \"{metric_key}\". Ask me for one of "
            "the metrics I track by name, and I'll pull it for whoever you need."
        )
    return None


def explain_unresolved_subject(name: str) -> str:
    """Ask who was meant, rather than answering about someone else."""
    return (
        f"I couldn't find anyone called \"{name}\". Check the spelling, or give "
        "me their full name as it appears on the master sheet — I can also "
        "answer for a team, company, or region by name."
    )
