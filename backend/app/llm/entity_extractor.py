"""
Extracts structured entities from free text: advisor names, team names,
company names, metrics, time periods, numeric limits ("top 5"), and now
— every gazetteer match per type instead of only the first one, plus
comparator/threshold tokens ("more than 80%", "at least 5").

Advisor/team/company matching is done against a real gazetteer pulled from
the DB and cached in memory — matching against actual data beats keyword
regex for a domain this specific. The cache is per-process; for horizontal
scaling with multiple workers, swap this for a shared cache (Redis) so all
workers see the same refresh. This stays 100% rule-based on purpose: an
LLM has no advantage over SequenceMatcher for "does this substring match a
real DB value", and would only add latency/cost/hallucination risk.

Root Cause #2 fix: `extract_entities()` used to `break` on the first team/
company hit, silently discarding a second one — "Compare Blue Area with
Downtown" only ever saw "Blue Area". `teams`/`companies` below are now
lists of every gazetteer match found, longest-match-first so a full team
name isn't shadowed by a shorter partial hit contained within it. `team`/
`company` (singular) are kept for backward compatibility with the existing
rule-based query_planner.py, and are simply the first list entry.

Root Cause #4 fix: `thresholds` extracts a small, closed comparator
vocabulary (>, >=, <, <=, "at least", "more than", "over", "under",
"below") paired with a number — fed to the LLM semantic parser as a hint,
and used directly by the rule-based fast path when unambiguous.

Part 12 (semantic retrieval expansion): both the comparator vocabulary and
the attendance-status keyword map now have a semantic widening step for
phrasing the closed lists above don't cover ("north of 80", "showed up
late" without the bare word "late"). Both only run after their
deterministic list finds nothing — see _extract_thresholds() and the
attendance block in extract_entities() — so neither can override an
already-matched deterministic result. The comparator floor is
deliberately high: a wrong operator silently flips a filter's direction,
which is a wrong-answer bug, not a missing-answer one.
"""

import dataclasses
import re
import time
from sqlalchemy.orm import Session
from sqlalchemy import distinct
from app.core import tracing
from app.core.config import settings
from app.database.models import Advisor
from app.llm import (
    advisor_resolver, comparators, cross_turn_resolver, entity_linker, hierarchy,
    reference_parser, relation_resolver, token_match,
)
from app.llm.fuzzy_match import best_match, find_in_text, STRONG_FLOOR
from app.llm.temporal_parser import parse_period

_CACHE_TTL_SECONDS = 300
_cache = {
    "teams": [], "companies": [], "advisor_names": [],
    "offices": [], "portfolio_leads": [], "management_leads": [],
    # New hierarchy levels (Advisor.bm / Advisor.zm / Advisor.office — see
    # hierarchy.py for the single mapping). "business_centers" duplicates
    # the same source column as the pre-existing "offices" cache above —
    # kept as its own cache key so entities come back keyed by the
    # chat-facing hierarchy name ("business_center"), not the legacy
    # "office" entity type, while "offices"/"office" stay untouched for
    # backward compatibility.
    # PHASE 3: the verified chain's levels. `business_centers` and
    # `units` are gone — office is now the single canonical key for
    # Advisor.office, and Advisor.unit had zero production rows.
    "unit_heads": [], "zonal_heads": [], "bcms": [], "regions": [],
    "loaded_at": 0,
}
ATTENDANCE_STATUS_KEYWORDS = {
    "not marked": "Not Marked",
    "late": "Late",
    "present": "Present",
    "absent": "Absent",
}

# Part 12: paraphrases the bare keywords above don't catch. Merged with
# them as a semantic fallback, never a replacement — see extract_entities().
#  None of these literally contain one of ATTENDANCE_STATUS_KEYWORDS'
#  words ("not marked"/"late"/"present"/"absent") — an exemplar that did
#  would never actually exercise this fallback, since the keyword check
#  above would already have matched it first. Every exemplar also contains
#  at least one word _ATTENDANCE_HINT_RE below recognizes, so the cheap
#  pre-filter and the corpus stay in sync.
_ATTENDANCE_STATUS_EXEMPLARS: list[tuple[str, str]] = [
    ("didn't show up", "Absent"), ("no show", "Absent"), ("missed today", "Absent"),
    ("was a no-show", "Absent"),
    ("walked in behind schedule", "Late"), ("clocked in behind schedule", "Late"),
    ("showed up past the scheduled start time", "Late"), ("came in past the scheduled time", "Late"),
    ("showed up on schedule", "Present"), ("showed up today", "Present"), ("clocked in on time", "Present"),
    ("hasn't punched in", "Not Marked"), ("no attendance record", "Not Marked"),
    ("didn't log attendance", "Not Marked"), ("no biometric record", "Not Marked"),
]

entity_linker.register_exemplar_type("attendance_status", lambda: _ATTENDANCE_STATUS_EXEMPLARS)

# Cheap pre-filter for the semantic attendance fallback below — most
# messages have nothing to do with attendance at all and must never pay
# for an embedding call just to find that out.
_ATTENDANCE_HINT_RE = re.compile(
    r"\b(show\w*|attend\w*|present|absent|late|punch\w*|mark\w*|biometric|login|"
    r"walk\w*|clock\w*|schedule\w*|miss\w*)\b", re.I
)

# Phase 2: the comparator vocabulary moved to app/llm/comparators.py, which
# declares each operator's deterministic phrases and its semantic
# paraphrases TOGETHER. Two separate lists lived here and had drifted —
# "above" was only ever an exemplar, so it resolved only when an embedding
# call was available while "below" resolved deterministically. Both names
# below are kept as thin aliases so existing call sites and tests continue
# to work unchanged.
_THRESHOLD_PATTERNS: list[tuple[str, str]] = comparators.threshold_patterns()
_COMPARATOR_EXEMPLARS: list[tuple[str, str]] = comparators.semantic_exemplars()

entity_linker.register_exemplar_type("comparator", lambda: _COMPARATOR_EXEMPLARS)

_SEMANTIC_COMPARATOR_FLOOR = 0.85
_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%?")


def _distinct(db: Session, column, master_only) -> list[str]:
    """Distinct non-null values of one hierarchy column."""
    return [v for (v,) in db.query(distinct(column)).filter(column.isnot(None), master_only).all()]


def _refresh_cache(db: Session):
    now = time.time()
    if _cache["loaded_at"] and now - _cache["loaded_at"] < _CACHE_TTL_SECONDS:
        return
    # in_master_sheet=True only — a team/company/advisor name that only
    # exists because of a raw-data-only WID shouldn't ground a query or
    # get fuzzy-matched against (see models.py's in_master_sheet docstring).
    master_only = Advisor.in_master_sheet.is_(True)
    _cache["teams"] = [t for (t,) in db.query(distinct(Advisor.team)).filter(Advisor.team.isnot(None), master_only).all()]
    _cache["companies"] = [c for (c,) in db.query(distinct(Advisor.company)).filter(Advisor.company.isnot(None), master_only).all()]
    # Distinct names, sourced from advisor_resolver so identity resolution
    # and this gazetteer can never disagree about who exists. Previously a
    # separate non-distinct query, which also meant a name shared by 8
    # people appeared 8 times and skewed fuzzy scans.
    #
    # force=True deliberately: the resolver keeps its own TTL, and two
    # caches expiring independently means the gazetteer can say a name
    # exists while identity resolution says it doesn't (or vice versa).
    # Refreshing here makes THIS function the single invalidation point
    # for both — resetting `_cache["loaded_at"]` now reliably reloads
    # identity too, which is what every caller and test already assumes.
    advisor_resolver.refresh_cache(db, force=True)
    _cache["advisor_names"] = advisor_resolver.known_names(db)
    _cache["offices"] = [o for (o,) in db.query(distinct(Advisor.office)).filter(Advisor.office.isnot(None), master_only).all()]
    _cache["portfolio_leads"] = [p for (p,) in db.query(distinct(Advisor.portfolio_lead)).filter(Advisor.portfolio_lead.isnot(None), master_only).all()]
    _cache["management_leads"] = [m for (m,) in db.query(distinct(Advisor.management_lead)).filter(Advisor.management_lead.isnot(None), master_only).all()]
    # Phase 3: loaded from the columns hierarchy.py declares, so a rebind
    # there cannot leave a gazetteer reading the old column.
    _cache["unit_heads"] = _distinct(db, hierarchy.column_for("unit_head"), master_only)
    _cache["zonal_heads"] = _distinct(db, hierarchy.column_for("zonal_head"), master_only)
    _cache["bcms"] = _distinct(db, hierarchy.column_for("bcm"), master_only)
    _cache["regions"] = _distinct(db, hierarchy.column_for("region"), master_only)
    _cache["loaded_at"] = now


def get_known_teams(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["teams"]


def get_known_companies(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["companies"]


def get_known_advisor_names(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["advisor_names"]


def get_known_offices(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["offices"]


def get_known_portfolio_leads(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["portfolio_leads"]


def get_known_management_leads(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["management_leads"]


def get_known_unit_heads(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["unit_heads"]


def get_known_zonal_heads(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["zonal_heads"]


def get_known_bcms(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["bcms"]


def get_known_regions(db: Session) -> list[str]:
    _refresh_cache(db)
    return _cache["regions"]


# Registered once at import time — the only step required to make a new
# entity type semantically linkable is one more call like these (Part 9).
entity_linker.register_entity_type("advisor", get_known_advisor_names, kind="advisor")
entity_linker.register_entity_type("team", get_known_teams, kind="team")
entity_linker.register_entity_type("company", get_known_companies, kind="company")
entity_linker.register_entity_type("office", get_known_offices, kind="team")
entity_linker.register_entity_type("portfolio_lead", get_known_portfolio_leads, kind="advisor")
entity_linker.register_entity_type("management_lead", get_known_management_leads, kind="advisor")
entity_linker.register_entity_type("unit_head", get_known_unit_heads, kind=hierarchy.match_kind_for("unit_head"))
entity_linker.register_entity_type("zonal_head", get_known_zonal_heads, kind=hierarchy.match_kind_for("zonal_head"))
entity_linker.register_entity_type("bcm", get_known_bcms, kind=hierarchy.match_kind_for("bcm"))
entity_linker.register_entity_type("region", get_known_regions, kind=hierarchy.match_kind_for("region"))


# Built from intent_catalog.LIMIT_RANKING_WORDS so the ranking
# vocabulary and the limit vocabulary cannot drift apart. Word-bounded
# via token_match for the usual reason: "top" sits inside "laptop".
def _limit_pattern() -> str:
    from app.llm import intent_catalog as cat, token_match

    words = "|".join(
        token_match.bounded(word) for word in cat.LIMIT_RANKING_WORDS
    )
    return rf"(?:{words})\s+(\d+)"


def _extract_thresholds(q: str) -> list[dict]:
    thresholds = []
    consumed_spans: list[tuple[int, int]] = []

    def _overlaps(span: tuple[int, int]) -> bool:
        return any(start < span[1] and span[0] < end for start, end in consumed_spans)

    # RANGES FIRST. "between 60 and 80" is two bounds, and the "and"
    # joining them must not be read as two independent constraints.
    # Consuming the whole span also stops the single-comparator patterns
    # from re-reading either number.
    range_pattern, (low_operator, high_operator) = comparators.range_pattern()
    for match in re.finditer(range_pattern, q):
        low, high = float(match.group(1)), float(match.group(2))
        # "between 80 and 60" is the same request said backwards; taking
        # the words literally would emit >= 80 AND <= 60, which matches
        # nothing and looks like "no results" rather than a misreading.
        low, high = min(low, high), max(low, high)
        thresholds.append({"operator": low_operator, "value": low})
        thresholds.append({"operator": high_operator, "value": high})
        consumed_spans.append(match.span())

    for pattern, operator in _THRESHOLD_PATTERNS:
        for match in re.finditer(pattern, q):
            # Patterns are longest-first, so an already-consumed span
            # belongs to a MORE specific phrase. Without this, "no more
            # than 50" matched the negation AND the "more than 50" inside
            # it, emitting <= 50 and > 50 together — a contradiction that
            # silently returns nothing.
            if _overlaps(match.span()):
                continue
            thresholds.append({"operator": operator, "value": float(match.group(1))})
            consumed_spans.append(match.span())

    # Part 12: a number the closed vocabulary above didn't already pair
    # with a comparator might still be one, phrased unusually ("north of
    # 80", "a bit under 30") — classify the words right before it against
    # the comparator exemplars at a HIGH floor (see module docstring on
    # why) before leaving it as just a bare number with no comparator.
    for match in _NUMBER_RE.finditer(q):
        if any(start <= match.start() < end for start, end in consumed_spans):
            continue
        window = q[max(0, match.start() - 25):match.start()].strip()
        if not window:
            continue
        semantic = entity_linker.semantic_classify(window, "comparator", top_k=1, floor=_SEMANTIC_COMPARATOR_FLOOR)
        if semantic:
            thresholds.append({"operator": semantic[0]["value"], "value": float(match.group(1))})

    return thresholds


def _own_vocabulary(entity_type: str) -> list[str]:
    """The words that NAME this level — its keywords plus, when it is a
    relation, its role aliases. Longest first so a phrase is removed
    before one of its own words is."""
    from app.llm import relations

    words = set(hierarchy.LEVEL_KEYWORDS.get(entity_type, ()))
    spec = relations.registry.resolve("advisor", entity_type)
    if spec is not None:
        words |= set(spec.role_aliases)
    return sorted(words, key=len, reverse=True)


def _without_own_vocabulary(q: str, entity_type: str) -> str:
    """`q` with this level's own vocabulary removed, for FUZZY matching
    only.

    A level's name is not one of its values. "who is X's unit head" asks
    about a role, but the unit gazetteer holds values like "Unit 1", and
    RapidFuzz scores "unit head" against "Unit 1" at 0.83 — above the
    0.80 team floor. That fabricated a unit entity for a question that
    named none, and with two units in the data it also produced two
    matches, which trips semantic_parser.looks_compound() and diverts a
    working reverse-lookup question to the LLM.

    Exact substring matching runs BEFORE this and on the untouched text,
    so a value that genuinely contains its level's name ("Unit 1",
    "Team Rashid Majeed") still matches at full confidence. Only the
    typo-tolerant tier sees the reduced text, where a level keyword is
    noise rather than evidence.
    """
    reduced = q
    for phrase in _own_vocabulary(entity_type):
        reduced = re.sub(rf"\b{re.escape(phrase)}\b", " ", reduced, flags=re.I)
    return reduced


def _resolve_gazetteer_field(
    text: str, q: str, values: list[str], kind: str, entity_type: str, db: Session
) -> list[dict]:
    """Exact substring -> RapidFuzz fuzzy -> embedding semantic search
    (Part 9), in that order — each tier only runs if the previous one
    found nothing. Longest-value-first on the substring pass so a full
    name isn't shadowed by a shorter partial hit contained within it
    (Root Cause #2). Returns [] rather than guessing once every tier has
    been tried and none cleared its floor — the caller's job is then to
    ask for clarification, not to assume no match exists.

    The fuzzy floor is kind-dependent: PERSON-valued levels (unit_head,
    zonal_head — kind="advisor") use the stricter advisor_resolver.
    PERSON_FLOOR, because the 0.80 team floor is routinely cleared by
    genuinely different people in this name population — "yasir ali" vs
    "asif ali" scores 0.82, which fabricated a unit_head/zonal_head
    entity for a query that never mentioned Asif Ali, and then triggered
    a clarifying question about the wrong person entirely."""
    ordered = sorted((v for v in values if v), key=len, reverse=True)
    # Token-aware for the same reason as the keyword tables: a short
    # gazetteer value was matching inside longer words, grounding an
    # entity the query never named (a team called "GRO" hit "grocery").
    # token_match applies boundaries per EDGE, so values carrying
    # punctuation — "North/KPK", "P1+P2" — still match; a blanket \b
    # would not.
    matches = [{"value": v, "score": 1.0} for v in ordered if token_match.contains(q, v)]
    if matches:
        return matches

    floor = advisor_resolver.PERSON_FLOOR if kind == "advisor" else STRONG_FLOOR
    matches = [
        {"value": v, "score": s}
        for v, s in find_in_text(_without_own_vocabulary(q, entity_type), values, kind=kind, floor=floor)
    ]
    if matches:
        return matches

    return entity_linker.semantic_candidates(text, entity_type, db)


# The hierarchy-relevant levels a matched value can be ambiguous across.
# portfolio_lead/management_lead/office are intentionally excluded — those
# stay their own separate, non-hierarchy entity types (unchanged, still
# independently pluggable), not part of this cross-level check.
# Derived from the hierarchy so a rebind cannot leave this list checking
# a level that no longer exists. `office` is excluded deliberately: it is
# an attribute, and its values overlap other levels' by design.
_AMBIGUITY_LEVELS = tuple(
    lvl for lvl in [*hierarchy.CHAIN, "company", "region"] if lvl != "office"
)


def _detect_ambiguous_entity(entities: dict) -> dict | None:
    """A name matched under more than one of team/company/unit_head/
    zonal_head/business_center/advisor is genuinely ambiguous — which one
    did the user mean? Without this, query_planner.py would silently pick
    one via whatever priority order it happens to check first (a real
    misresolution, not just a hypothetical one — e.g. an org where a Unit
    Head and an unrelated Advisor happen to share a name). Surfaced here so
    nlu_pipeline can ask instead of guess, fully rule-based (no LLM call,
    so it can't itself introduce a spurious guess)."""
    seen: dict[str, tuple[str, list[str]]] = {}
    for level in _AMBIGUITY_LEVELS:
        key = "advisor_name" if level == "advisor" else level
        value = entities.get(key)
        if not value:
            continue
        value_lower = value.lower()
        if value_lower in seen:
            seen[value_lower][1].append(level)
        else:
            seen[value_lower] = (value, [level])

    for value, levels in seen.values():
        if len(levels) > 1:
            return {"value": value, "levels": levels}
    return None


# ---------------------------------------------------------------------
# Provenance (M0 of the Relationship Inference Engine — audit debt D7)
#
# Every entity this module produces today came from the USER'S OWN WORDS:
# a gazetteer hit, a keyword, a number. Relationship inference (M1) will
# add entities that came from a JOIN instead — an advisor's team, which
# the user never named. Those two are indistinguishable in a bare dict,
# and telling them apart is what makes the feature auditable, scorable
# and safely reversible. So the slot is introduced now, while everything
# in it is still "explicit", rather than alongside the change that first
# makes it interesting.
#
# Keys are META, not entities: `_PROVENANCE_KEY` is underscore-prefixed
# and consumers that serialise the entity dict skip underscore keys (see
# prompt_builder.build_ir_prompt), so this is invisible downstream.
# ---------------------------------------------------------------------
PROVENANCE_KEY = "_provenance"
# M6: wids that were named only to REACH something else ("compare X's
# team with ...") rather than as an entity in their own right. Meta, so
# underscore-prefixed like the provenance map — consumers that serialise
# the entity dict skip these.
REFERENCE_SOURCES_KEY = "_reference_sources"
# Pre-M4 private spelling, kept so existing call sites and tests that
# reference it keep working. Same object, one name is now public because
# cross-turn inference (a separate module) has to write into this slot.
_PROVENANCE_KEY = PROVENANCE_KEY

# The user said it. Every entity extracted from the query text.
PROVENANCE_EXPLICIT = "explicit"


def provenance_of(entities: dict, key: str) -> str | None:
    """How `key` came to be in this entity dict, or None if unrecorded.
    A reader for downstream code so no call site has to know the storage
    layout."""
    return (entities.get(_PROVENANCE_KEY) or {}).get(key)


def _finalize_provenance(entities: dict) -> None:
    """Records provenance for every entity key that doesn't already have
    it, then stores the map under the meta key.

    Filling defaults at the END rather than tagging at each write site is
    deliberate: it is total by construction — a new extraction can never
    be forgotten — and it leaves earlier, more specific marks intact, so
    M1 can tag an inferred entity where it is produced and rely on this
    pass not to overwrite it.
    """
    marks: dict[str, str] = dict(entities.get(_PROVENANCE_KEY) or {})
    for key in entities:
        # Underscore keys are META about the extraction, not entities, so
        # they have no provenance of their own.
        if key.startswith("_"):
            continue
        marks.setdefault(key, PROVENANCE_EXPLICIT)
    entities[_PROVENANCE_KEY] = marks


# ---------------------------------------------------------------------
# Relationship inference (M1)
#
# THE SAFETY RULE, from §2.5 of the design: inference fires ONLY when a
# relational reference was actually parsed from the text. It is never an
# unconditional enrichment.
#
# The distinction is the whole feature. Populating entities["team"] for
# every query that names a person would give `_score_hierarchy` a
# grounded group on "tell me about Waqar Haider", where it would score
# ~0.92 and beat `advisor_profile` at 0.5 — silently converting the
# product's most common query shape into a team breakdown. Gating on a
# parsed reference confines the change to queries that are broken today.
# ---------------------------------------------------------------------

def _infer_related_entities(text: str, entities: dict, db: Session) -> None:
    """Writes inferred group entities into `entities`, in place.

    Emits the SAME key shape as a gazetteer match ({level}_matches,
    plural, singular), because that shape is what the planner, the
    ambiguity check, plan_to_ir and the compiler already read — inference
    fills existing slots rather than introducing a parallel channel that
    every downstream consumer would have to learn about.
    """
    if not settings.relation_inference_enabled:
        return

    levels = settings.relation_inference_level_set
    if not levels:
        return

    references = reference_parser.references_to(text, levels)
    pronouns = [
        reference for reference in reference_parser.parse_pronoun(text)
        if reference.target_level in levels
    ]
    if not references and not pronouns:
        return

    for reference in [*references, *pronouns]:
        resolution = _source_resolution(reference, text, entities, db, len(references))
        related = relation_resolver.resolve_from_resolution(resolution, reference.target_level)
        if related is None:
            continue
        if reference.kind == reference_parser.NAMED:
            # This person was named in order to reach something else —
            # "compare X's team with Y's team" is about two TEAMS. Recorded
            # so the planner can tell a reference SOURCE from a comparison
            # TARGET without knowing anything about parsing. A pronoun
            # source is deliberately not recorded: "how does X compare to
            # his team" names X as one side of the comparison.
            sources = entities.setdefault(REFERENCE_SOURCES_KEY, [])
            if related.source_id not in sources:
                sources.append(related.source_id)
        # One writer, shared with cross-turn inference (M4): a follow-up
        # must produce the same entity shape as the query that
        # established the subject, and two writers would eventually
        # differ by a key. It appends, so two references at one level
        # become two targets rather than one overwriting the other.
        cross_turn_resolver.bind_relation(entities, related, _PROVENANCE_KEY)


def _source_resolution(reference, text: str, entities: dict, db: Session, named_count: int):
    """Which person THIS reference is about.

    M6: resolved per reference rather than once per message. Identity
    resolution over the whole text returns exactly one advisor — the last
    name span wins — so "compare Waqar Haider's team with Sana Tariq's
    team" used to resolve both references to Sana and compare Downtown
    with itself. Each reference now resolves its own source span, which
    the parser bounds so one reference cannot reach into another.

    A PRONOUN reference has no span of its own: within a single message
    it points at the person that message names. When the message names
    nobody, this returns None and cross_turn_resolver (M4) picks it up
    from conversation memory instead — the two paths are disjoint by
    construction, so a pronoun is never resolved twice.
    """
    if reference.kind == reference_parser.PRONOUN:
        # Exactly one named person, or the pronoun is ambiguous and we
        # must not choose.
        #
        # Phase 5B also consults advisor_multi. `advisor_wids` comes from
        # single-identity resolution, which returns ONE person when it
        # can resolve confidently — so "compare Waqar Haider and Sana
        # Tariq to his team" left advisor_wids at length 1 and this gate
        # bound "his" to whichever of the two it had picked. The gate was
        # only ever passing because span extraction used to fail on
        # "compare <name>"; once that was fixed the accidental protection
        # went with it. Two named people make a pronoun ambiguous however
        # many of them identity resolution settled on.
        if len(entities.get("advisor_multi") or []) > 1:
            return None
        if len(entities.get("advisor_wids") or []) != 1:
            return None
        return entities.get("advisor_resolution")

    if reference.source_span:
        resolution = advisor_resolver.resolve_from_text(reference.source_span, db)
        if resolution.status == advisor_resolver.RESOLVED:
            return resolution
        # A span that resolves to nobody falls back to the message-wide
        # resolution ONLY when it is the sole reference — with several,
        # borrowing another reference's person is exactly the confusion
        # this function exists to remove.
        if named_count > 1:
            return None

    return entities.get("advisor_resolution")


def extract_entities(text: str, db: Session) -> dict:
    _refresh_cache(db)
    q = text.lower()
    entities: dict = {}

    # F1: token_match, not `keyword in q`. "late" is inside "calculated",
    # "related", "translate" and "escalate", and "present" is inside
    # "representative" — so "How is the answered calls % calculated?" set
    # attendance_status=Late, which scores 0.98 and returns before the
    # semantic parser ever runs. Dict order is load-bearing and preserved:
    # "not marked" is checked before "late".
    matched_status = token_match.first_match(q, ATTENDANCE_STATUS_KEYWORDS)
    if matched_status is not None:
        entities["attendance_status"] = ATTENDANCE_STATUS_KEYWORDS[matched_status]
    else:
        # Part 12: no exact keyword — try semantic retrieval against
        # paraphrases ("didn't show up", "clocked in late") before
        # concluding no status was mentioned. Gated on a cheap keyword
        # hint so the common case (no attendance language at all) never
        # pays for an embedding call.
        if _ATTENDANCE_HINT_RE.search(q):
            semantic = entity_linker.semantic_classify(q, "attendance_status", top_k=1)
            if semantic:
                entities["attendance_status"] = semantic[0]["value"]

    # Part 8: parse_period() replaces the old bare "month"/"year" keyword
    # map, which used to silently turn "last month" into MTD. Genuinely
    # unsupported windows (last month, yesterday, this week, past N days,
    # custom ranges) are surfaced as period_unsupported instead of a
    # silently wrong period — never set alongside "period".
    temporal = parse_period(q)
    if temporal is not None:
        if temporal.kind == "equivalent":
            entities["period"] = temporal.period
            entities["period_confidence"] = temporal.confidence
        else:
            entities["period_unsupported"] = temporal.reason

    # FIX 1: any ranking word may carry the limit, not just "top".
    # "bottom 5", "worst 3" and "lowest 5" all silently lost their N and
    # fell back to the default 10 — so a request for 5 rows returned 10,
    # on top of being sorted the wrong way.
    limit_match = re.search(_limit_pattern(), q)
    if limit_match:
        entities["limit"] = int(limit_match.group(1))

    entities["thresholds"] = _extract_thresholds(q)

    # companies/teams/offices/portfolio & management leads — exact substring,
    # then fuzzy, then embedding semantic search (Part 9); see
    # _resolve_gazetteer_field. ALL matches collected, not just the first
    # (Root Cause #2). New pluggable entity types (office, portfolio_lead,
    # management_lead) follow the exact same {plural}_matches/{plural}/
    # {singular} shape as the original team/company fields, so nothing
    # downstream that only knows about team/company needs to change.
    for plural, kind, entity_type in (
        ("companies", "company", "company"),
        ("teams", "team", "team"),
        ("offices", "team", "office"),
        ("portfolio_leads", "advisor", "portfolio_lead"),
        ("management_leads", "advisor", "management_lead"),
        ("unit_heads", hierarchy.match_kind_for("unit_head"), "unit_head"),
        ("zonal_heads", hierarchy.match_kind_for("zonal_head"), "zonal_head"),
        ("bcms", hierarchy.match_kind_for("bcm"), "bcm"),
        ("regions", hierarchy.match_kind_for("region"), "region"),
    ):
        matches = _resolve_gazetteer_field(text, q, _cache[plural], kind, entity_type, db)
        if matches:
            entities[f"{entity_type}_matches"] = matches
            entities[plural] = [m["value"] for m in matches]
            entities[entity_type] = matches[0]["value"]   # backward-compat singular

    # ---- advisor IDENTITY (Phase 1 refactor) ----
    # Delegated wholesale to advisor_resolver, which resolves to a WID
    # rather than a name. Three behaviors changed here, all of them causes
    # of wrong-person answers in the audit:
    #
    #  1. No more whole-sentence fuzzy matching. The old code stripped a
    #     few filler phrases and ran best_match() over everything left, so
    #     "show adeel dogar's team" scored "Adeel Mubarik Dogar" at 0.62
    #     and returned that unrelated person's profile. Resolution now
    #     uses candidate-sized token windows (find_in_text) at a 0.90
    #     person floor.
    #  2. No more silent single-answer. A name matching several real
    #     people (238 such groups in production) comes back AMBIGUOUS with
    #     every candidate, for the planner to ask about.
    #  3. `advisor_wids` carries identity downstream. `advisor_name` is
    #     still emitted so existing consumers keep working, but it is now
    #     a DISPLAY value, never an identifier.
    resolution = advisor_resolver.resolve_from_text(text, db)
    if resolution.status == advisor_resolver.NOT_FOUND:
        # last-resort semantic widening (Part 9) — unchanged in spirit,
        # but its hit is re-resolved through the same identity path so it
        # can never bypass the ambiguity check either. The SEMANTIC score
        # is carried onto the resolved identity rather than replaced with
        # resolve_by_name's exact-match 1.0: how confident we are that
        # this is the right person is a property of the semantic hit, and
        # overwriting it would misreport a 0.78 guess as certainty.
        semantic_matches = entity_linker.semantic_candidates(text, "advisor", db)
        if semantic_matches:
            semantic_score = semantic_matches[0]["score"]
            resolution = advisor_resolver.resolve_by_name(semantic_matches[0]["value"], db)
            # replace() rather than a field-by-field rebuild — see
            # advisor_resolver._with_score. Naming fields explicitly here
            # would drop every cached relation added later, so a query
            # that reached the identity through semantic widening would
            # silently lose its team/unit head/zonal head.
            resolution.candidates = [
                dataclasses.replace(c, score=semantic_score)
                for c in resolution.candidates
            ]

    # Phase 7: record what identity resolution decided AND what it was
    # choosing between — the candidate list is the piece that makes a
    # "you gave me the wrong person" report diagnosable after the fact.
    tracing.record_identity(resolution)

    if resolution.candidates:
        entities["advisor_resolution"] = resolution
        entities["advisor_wids"] = [c.wid for c in resolution.candidates]
        entities["advisor_matches"] = [
            {"value": c.name, "score": c.score, "wid": c.wid} for c in resolution.candidates
        ]
        entities["advisor_names"] = list(dict.fromkeys(c.name for c in resolution.candidates))
        entities["advisor_name"] = resolution.candidates[0].name
        entities["advisor_match_score"] = round(resolution.candidates[0].score, 2)
        if resolution.is_resolved:
            entities["advisor_wid"] = resolution.wid
        elif resolution.is_ambiguous:
            entities["advisor_ambiguous"] = True

    # Phase 5B: every DISTINCT person named, for the queries that are
    # about more than one. `advisor_resolution` above answers "who is
    # this about?" and collapses to a single identity, which is correct
    # for a lookup and loses a side of a comparison. Recorded under its
    # own key rather than widening advisor_wids, so nothing that reads
    # the single-identity keys changes behaviour.
    if len(entities.get("advisor_wids") or []) < 2:
        multi = advisor_resolver.resolve_all_from_text(text, db)
        if len(multi) > 1:
            entities["advisor_multi"] = [
                {"wid": i.wid, "name": i.name} for i in multi
            ]

    ambiguous = _detect_ambiguous_entity(entities)
    if ambiguous:
        entities["ambiguous_entity"] = ambiguous

    # Relationship inference (M1) runs AFTER cross-level ambiguity
    # detection, deliberately: that check asks "the user named something
    # that could be two levels — which did they mean?", and an entity the
    # user never named cannot make their wording ambiguous. Running it
    # earlier would let an inferred team collide with an advisor name and
    # raise a clarifying question about a word nobody typed.
    _infer_related_entities(text, entities, db)

    _finalize_provenance(entities)
    return entities
