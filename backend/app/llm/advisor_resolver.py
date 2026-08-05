"""
Advisor identity resolution (Phase 1 of the identity refactor).

THE PROBLEM THIS EXISTS TO SOLVE — from the pipeline audit: the chatbot
identified people by NAME STRING end to end. `Advisor.wid` (the actual
primary key) was read from the database during gazetteer loading, the name
was kept, and the wid was thrown away. Every downstream stage then had to
re-guess which human a name referred to:

  - 238 duplicate-name groups exist in production; 8 different real people
    are named "Yasir Ali". A name is simply NOT an identifier here.
  - advisor_service did `WHERE name ILIKE '%q%' ORDER BY wid LIMIT 1`, so
    "Ahmed Ali" returned "Ahmed Ali Pirzada" and 7 of the 8 Yasir Alis were
    permanently unreachable.

This module makes identity explicit. It owns the (wid, name, team, company)
cache, and every resolution returns one of exactly three outcomes:

  RESOLVED    — exactly one advisor. `.wid` is safe to use.
  AMBIGUOUS   — several real people match. The caller MUST ask which one;
                it must never silently pick. `.candidates` carries enough
                context (team/company) to make the question answerable.
  NOT_FOUND   — nothing cleared the floor. Say so; never fall back to a
                loose substring guess.

Matching policy, deliberately stricter than the team/metric matchers in
fuzzy_match.py: a wrong TEAM filter returns fewer rows, but a wrong PERSON
returns someone else's revenue under someone else's name. So:

  1. exact (case-insensitive) name equality — the only tier that can
     produce a confident single answer on its own
  2. token-window fuzzy match over the QUERY TEXT (never the whole
     sentence — see find_in_text) at PERSON_FLOOR, which is set well above
     the 0.80 team floor because short South-Asian name pairs collide
     ("yasir ali" vs "asif ali" scores 0.82)
  3. a decisiveness margin: if the runner-up scores within
     AMBIGUITY_MARGIN of the winner, that's AMBIGUOUS, not a win

Substring containment ("Ali" matching "Iqra Ali") is deliberately NOT a
resolution tier. It is what produced the wrong-person answers, and a
partial name is exactly the case that should ask rather than guess.
"""

from __future__ import annotations

import dataclasses
import re
import time
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy.orm import Session

from app.database.models import Advisor
from app.llm.fuzzy_match import find_in_text

# A person match must be near-exact. Above the 0.80 STRONG_FLOOR used for
# teams because this name population shares a small component pool (Ali,
# Ahmed, Muhammad, Hassan, Syed) and genuinely different people routinely
# score 0.80-0.85 against each other.
PERSON_FLOOR = 0.90

# If the second-best candidate is within this of the best, the match isn't
# decisive — ask instead of picking. Catches "Ahmed Ali" vs "Ahmad Ali".
AMBIGUITY_MARGIN = 0.05

_CACHE_TTL_SECONDS = 300

RESOLVED = "resolved"
AMBIGUOUS = "ambiguous"
NOT_FOUND = "not_found"

ResolutionStatus = Literal["resolved", "ambiguous", "not_found"]


@dataclass(frozen=True)
class AdvisorIdentity:
    """One real person. `wid` is the identifier every downstream stage
    should use; name/team/company exist to make a disambiguation question
    answerable by a human ("which Yasir Ali — North/KPK or Team Rashid
    Majeed?").

    M3: the hierarchy fields below are named for their RELATION TARGET
    LEVEL, not for the Advisor column they come from (`unit_head` holds
    Advisor.bm, `zonal_head` holds Advisor.zm, `business_center` holds
    Advisor.office). That naming is what lets relationship inference stay
    generic: relation_resolver reads `getattr(identity, target_level)`,
    so a newly cached relation needs no resolver change at all — only a
    field here and `cached=True` on its declaration.

    Every field is populated from the single projection in
    refresh_cache(), which is itself derived from the registry, so this
    list and the `cached` flags cannot drift apart.
    """
    wid: int
    name: str
    team: str | None = None
    company: str | None = None
    # Phase 3: one field per CACHED relation, named for its target level.
    # `bcm` replaced `business_center` when the chain was rebound, and
    # `office` became a groupable attribute rather than a chain level.
    unit_head: str | None = None
    zonal_head: str | None = None
    bcm: str | None = None
    office: str | None = None
    score: float = 1.0

    def label(self) -> str:
        """Human-facing description, used when asking the user to choose."""
        context = " · ".join(p for p in (self.team, self.company) if p)
        return f"{self.name} ({context})" if context else self.name


@dataclass
class ResolvedAdvisor:
    """The Phase 2 resolution contract.

    `wid`/`name`/`confidence` are populated ONLY when the query identifies
    exactly one person. When several match, they stay None/0.0 and
    `candidates` carries all of them — making it structurally impossible
    for a caller to read a single advisor out of an ambiguous result,
    which is how the old lookup silently returned the wrong person.
    """
    status: ResolutionStatus
    candidates: list[AdvisorIdentity] = field(default_factory=list)
    # the raw text span that was matched, for logging/traceability
    matched_text: str | None = None

    @property
    def wid(self) -> int | None:
        return self.candidates[0].wid if self.status == RESOLVED and self.candidates else None

    @property
    def name(self) -> str | None:
        return self.candidates[0].name if self.status == RESOLVED and self.candidates else None

    @property
    def confidence(self) -> float:
        """0.0 for anything that isn't a single confident person — an
        ambiguous match is NOT "confidently one of these"."""
        return self.candidates[0].score if self.status == RESOLVED and self.candidates else 0.0

    @property
    def identity(self) -> AdvisorIdentity | None:
        return self.candidates[0] if self.status == RESOLVED and self.candidates else None

    @property
    def is_resolved(self) -> bool:
        return self.status == RESOLVED

    @property
    def is_ambiguous(self) -> bool:
        return self.status == AMBIGUOUS

    def to_dict(self) -> dict:
        """Serializable form — the {wid, name, confidence, candidates}
        shape used by the API layer and request tracing."""
        return {
            "status": self.status,
            "wid": self.wid,
            "name": self.name,
            "confidence": round(self.confidence, 2),
            "candidates": [
                {"wid": c.wid, "name": c.name, "team": c.team,
                 "company": c.company, "score": round(c.score, 2)}
                for c in self.candidates
            ],
        }


# The pre-Phase-2 name, kept as an alias so existing call sites and tests
# keep working — same object, the contract only gained fields.
AdvisorResolution = ResolvedAdvisor


_cache: dict = {"identities": [], "by_name": {}, "loaded_at": 0.0}


def cached_relation_specs() -> list:
    """The advisor relations whose values ride along on AdvisorIdentity,
    in declaration order. The projection below is built from this, so
    "which relations are free to resolve" is stated once — on the
    declaration — instead of in a column list here and a flag there."""
    from app.llm import relations

    return [spec for spec in relations.registry.specs_for("advisor") if spec.cached]


def refresh_cache(db: Session, force: bool = False) -> None:
    """Loads (wid, name) plus every CACHED relation's column for each
    master-sheet advisor. Mirrors entity_extractor's TTL so identity and
    gazetteer staleness track each other.

    M3 widened this projection from (team, company) to include
    business_center/unit_head/zonal_head. One query, one refresh, one
    invalidation point: every relation on an identity is therefore as
    fresh as every other, which is the property an on-demand read would
    have given up — it would have introduced a second freshness domain
    where a cached team and a live unit head could disagree about the
    same advisor inside one response.
    """
    now = time.time()
    if not force and _cache["loaded_at"] and now - _cache["loaded_at"] < _CACHE_TTL_SECONDS:
        return

    specs = cached_relation_specs()
    rows = (
        db.query(Advisor.wid, Advisor.name, *[spec.column for spec in specs])
        .filter(Advisor.in_master_sheet.is_(True), Advisor.name.isnot(None))
        .all()
    )
    identities = [
        AdvisorIdentity(
            wid=row[0],
            name=row[1],
            **{spec.target_level: value for spec, value in zip(specs, row[2:])},
        )
        for row in rows
    ]
    by_name: dict[str, list[AdvisorIdentity]] = {}
    for identity in identities:
        by_name.setdefault(identity.name.lower(), []).append(identity)

    _cache["identities"] = identities
    _cache["by_name"] = by_name
    _cache["loaded_at"] = now


def identity_for_wid(wid: int | None, db: Session) -> AdvisorIdentity | None:
    """The cached identity for one wid, or None.

    Added for cross-turn reference resolution (M4): a conversation
    remembers WHO it is about by wid, and the relations belonging to that
    person are then read from the current cache rather than from a copy
    frozen when the name was first mentioned. That keeps a remembered
    subject exactly as fresh as a just-named one — a stored identity
    would keep reporting last hour's team after a sync.
    """
    if wid is None:
        return None
    refresh_cache(db)
    for identity in _cache["identities"]:
        if identity.wid == wid:
            return identity
    return None


def known_names(db: Session) -> list[str]:
    """Distinct advisor names, for the fuzzy candidate list and for
    entity_linker's semantic index. Distinct on purpose: the raw identity
    list has one entry per PERSON, so 8 people named "Yasir Ali" would
    otherwise put that name into the gazetteer 8 times and skew every
    fuzzy scan."""
    refresh_cache(db)
    return sorted({i.name for i in _cache["identities"]})


# Tokens that can never be part of a person's name in this domain, so a
# contiguous run of everything EXCEPT these is a candidate name span.
# Deliberately a closed, curated list rather than a general stopword
# corpus: over-stripping is the dangerous direction here — removing a
# token that was actually part of a name breaks resolution outright,
# whereas leaving one extra token in a span merely lowers its fuzzy score
# and gets caught by the 0.90 floor.
_SPAN_STOPWORDS = frozenset({
    # question / filler
    "who", "whom", "whose", "what", "which", "where", "when", "how", "why",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "the", "a", "an", "of", "for", "to", "in", "on", "at", "by", "with",
    "and", "or", "but", "from", "about", "me", "my", "our", "us", "i",
    "you", "your", "please", "can", "could", "would", "should", "tell",
    "show", "give", "get", "find", "list", "display", "see", "look", "up",
    "s", "his", "her", "their", "its", "this", "that", "these", "those",
    "doing", "going", "performing", "performance", "report", "reports",
    "reporting", "details", "detail", "info", "information", "stats",
    "summary", "today", "now", "currently", "much", "many", "any", "all",
    "ok", "okay", "let", "know", "want", "need", "like", "some",
    # hierarchy / org vocabulary — a level keyword is never a name token
    "unit", "head", "heads", "zonal", "zone", "region", "regional",
    "business", "center", "centre", "branch", "division", "team", "teams",
    "advisor", "advisors", "agent", "agents", "company", "companies",
    "manager", "lead", "leads", "bm", "zm", "rm", "under", "works", "work",
    "member", "members", "people", "staff", "belongs", "belong",
    # comparison vocabulary — Phase 5B. "and"/"or" were already here, but
    # the verb and the connectives were not, so "compare yasir ali and
    # sana tariq" produced the span "compare yasir ali" (which resolves
    # to nobody) and "yasir ali vs omar farooq" produced ONE span
    # spanning both people. A comparison therefore saw one side of a
    # two-sided question. Like the hierarchy vocabulary above, none of
    # these can be part of a person's name.
    "compare", "compares", "compared", "comparing", "comparison",
    "vs", "versus", "against", "between", "difference", "differences",
    "more", "less", "fewer", "higher", "lower", "greater", "better",
    "worse", "best", "worst", "than", "each", "other",
})

# A person name in this dataset is 1-5 tokens ("Umer Khatab Abbasi Abbasi").
_MAX_SPAN_TOKENS = 5

_metric_words_cache: frozenset[str] | None = None


def _metric_words() -> frozenset[str]:
    """Single-token metric vocabulary, DERIVED from the alias registry.

    The stopword set above already carries a handful of measure words
    ("performance", "stats"), but a hand-kept list drifts: "cleared" was
    absent, so "compare Yasir Ali and Sana Tariq's cleared" produced the
    span "sana tariq cleared", which resolves to nobody — and the
    comparison saw one side of a two-sided question.

    Reading metric_aliases means a measure added later is excluded here
    for free. Only SINGLE tokens are taken: a multi-word alias cannot
    glue itself to a name the way a bare noun does, and splitting on its
    words would strip tokens that are fine inside a name ("new", "total").
    """
    global _metric_words_cache
    if _metric_words_cache is None:
        from app.llm.metric_aliases import ALIASES

        _metric_words_cache = frozenset(
            phrase.lower()
            for phrases in ALIASES.values()
            for phrase in phrases
            if " " not in phrase and len(phrase) > 2
        )
    return _metric_words_cache
_TOKEN_RE = re.compile(r"[A-Za-z0-9'\-\.]+")


def extract_name_spans(text: str) -> list[str]:
    """Candidate person-name spans in `text`, longest first.

    Splits on stopwords and punctuation, keeping contiguous runs of
    tokens that could plausibly be name parts. "show adeel dogar's team"
    -> ["adeel dogar"]; "who reports to Waqar Haider" -> ["waqar haider"].

    Longest-first because a longer span is more specific: for "adeel
    mubarik dogar" the full three-token span must be tried before its
    two-token prefix, or a different person could match first."""
    if not text:
        return []

    spans: list[str] = []
    current: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        # possessive 's is punctuation-attached in some tokenizations
        token = raw.lower().rstrip(".")
        if token.endswith("'s"):
            token = token[:-2]
        if (not token or token in _SPAN_STOPWORDS or token.isdigit()
                or token in _metric_words()):
            if current:
                spans.append(current)
                current = []
            continue
        current.append(token)
    if current:
        spans.append(current)

    out: list[str] = []
    for tokens in spans:
        if len(tokens) > _MAX_SPAN_TOKENS:
            tokens = tokens[:_MAX_SPAN_TOKENS]
        if tokens:
            out.append(" ".join(tokens))
    # longest first, preserving original order among equal lengths
    return sorted(dict.fromkeys(out), key=lambda s: -len(s.split()))


def resolve_by_name(name: str, db: Session) -> AdvisorResolution:
    """Exact (case-insensitive) name -> identity. The duplicate-name case
    that used to be invisible is surfaced here as AMBIGUOUS: 8 people
    named "Yasir Ali" produce 8 candidates, not an arbitrary pick."""
    refresh_cache(db)
    if not name:
        return AdvisorResolution(status=NOT_FOUND)

    matches = _cache["by_name"].get(name.strip().lower(), [])
    if not matches:
        return AdvisorResolution(status=NOT_FOUND, matched_text=name)
    if len(matches) == 1:
        return AdvisorResolution(status=RESOLVED, candidates=list(matches), matched_text=name)
    return AdvisorResolution(status=AMBIGUOUS, candidates=list(matches), matched_text=name)


def resolve_advisor(query: str, db: Session, floor: float = PERSON_FLOOR) -> ResolvedAdvisor:
    """THE person-resolution entry point (Phase 2). Replaces
    advisor_service.find_advisor_by_name and its
    `ILIKE '%query%' … LIMIT 1`, which returned a different person in two
    ways: substring containment ("Ahmed Ali" -> "Ahmed Ali Pirzada") and
    silent lowest-wid selection among duplicates.

    Tiers, in order — each only runs if the previous found nothing:

      1. case-insensitive EXACT name equality
      2. high-confidence fuzzy match (RapidFuzz, PERSON_FLOOR=0.90),
         which absorbs typos and word-order swaps ("Haider Waqar")

    There is deliberately NO substring tier. A partial name is precisely
    the case that must ask rather than guess — "Ali" is not a request for
    whichever of the 90 matching people happens to sort first.

    Always returns ALL candidates and never selects among several: `wid`
    is populated only when exactly one person matched.
    """
    refresh_cache(db)
    if not query or not query.strip():
        return ResolvedAdvisor(status=NOT_FOUND)

    cleaned = query.strip()

    # ---- tier 1: exact ----
    exact = resolve_by_name(cleaned, db)
    if exact.status != NOT_FOUND:
        return exact

    # ---- tier 2: high-confidence fuzzy over distinct names ----
    scorer = _person_scorer()
    needle = cleaned.lower()
    scored: list[tuple[str, float]] = []
    for name in {i.name for i in _cache["identities"]}:
        score = scorer(needle, name.lower())
        if score >= floor:
            scored.append((name, round(score, 2)))

    if not scored:
        return ResolvedAdvisor(status=NOT_FOUND, matched_text=cleaned)

    scored.sort(key=lambda pair: -pair[1])
    best_name, best_score = scored[0]

    # A near-tie between DIFFERENT names is ambiguous even though each
    # cleared the floor on its own ("Ahmed Ali" vs "Ahmad Ali").
    tied = [(n, s) for n, s in scored if best_score - s <= AMBIGUITY_MARGIN]

    candidates: list[AdvisorIdentity] = []
    for name, score in tied:
        for identity in _cache["by_name"].get(name.lower(), []):
            candidates.append(_with_score(identity, score))

    if not candidates:
        return ResolvedAdvisor(status=NOT_FOUND, matched_text=cleaned)
    if len(candidates) == 1:
        return ResolvedAdvisor(status=RESOLVED, candidates=candidates, matched_text=best_name)
    return ResolvedAdvisor(status=AMBIGUOUS, candidates=candidates, matched_text=best_name)


def _person_scorer():
    """Scorer for RESOLUTION — deliberately NOT the one fuzzy_match.py
    uses for kind="advisor".

    That scorer includes token_set_ratio (discounted to 0.9), which by
    design lets a partial name hit a full name: correct when LOCATING a
    name mentioned inside a sentence, wrong when deciding WHO a query
    means. It scores every bare fragment at exactly 0.90 — "Pirzada" vs
    "Ahmed Ali Pirzada", "Ali" vs "Yasir Ali" — landing precisely on the
    person floor and re-introducing the substring behaviour Phase 2
    removed from SQL.

    ratio + token_sort_ratio keeps what resolution actually needs (typos:
    "Waqar Haidar" 0.92; word-order swaps: "Haider Waqar" 1.00) while
    scoring those same fragments 0.50-0.71, well under the floor."""
    from rapidfuzz import fuzz

    return lambda a, b: max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b)) / 100.0


def resolve_all_from_text(
    text: str, db: Session, floor: float = PERSON_FLOOR
) -> list[AdvisorIdentity]:
    """EVERY distinct person named in `text`, in the order they appear.

    Phase 5B. resolve_from_text() answers "who is this query ABOUT?" — a
    single-identity question — and returns on the first unambiguous span
    (see the RESOLVED short-circuit below). That is right for a lookup
    and wrong for a comparison: "compare Yasir Ali and Sana Tariq"
    resolved Sana Tariq, dropped Yasir Ali, and the planner then saw one
    side of a two-sided question and asked which other name was meant.

    Same two stages and the same tiers, without the short-circuit. Lives
    here rather than in the planner because identity resolution has one
    owner, and a second span-matcher would drift from this one's
    stripping rules the first time either changed.

    Returns [] when fewer than one person resolves; the caller decides
    what too-few means. Ambiguous spans are deliberately skipped: "which
    Ali Raza did you mean" is a question the comparison cannot answer for
    the user, and picking one silently is the failure this whole
    programme has been removing.
    """
    refresh_cache(db)
    if not text or not _cache["identities"]:
        return []

    found: list[AdvisorIdentity] = []
    seen: set[int] = set()
    for span in extract_name_spans(text):
        result = resolve_advisor(span, db, floor=floor)
        if result.status != RESOLVED:
            continue
        identity = result.identity
        if identity is not None and identity.wid not in seen:
            seen.add(identity.wid)
            found.append(identity)
    return found


def resolve_from_text(text: str, db: Session, floor: float = PERSON_FLOOR) -> AdvisorResolution:
    """Find the advisor named in free text and resolve them to a wid.

    Two-stage, per Phase 3: EXTRACT the name span, then match only that
    span. The original code ran best_match() over the whole utterance,
    which is how "show adeel dogar's team" scored "Adeel Mubarik Dogar"
    at 0.62 and returned an unrelated person's profile — every filler
    word was scored as if it were part of the name.

    Stage 1 (extract_name_spans) strips question words, verbs,
    possessives, hierarchy keywords and metric words, leaving contiguous
    runs of unknown tokens — the only things that can plausibly BE a
    name. Stage 2 resolves each span through the same exact->fuzzy tiers
    resolve_advisor uses, so a span gets identical treatment to a name
    typed on its own.

    Window scanning (find_in_text) remains as a fallback for the case
    where stripping removed too much — it is still window-based, never
    whole-sentence, so it cannot reintroduce the original bug."""
    refresh_cache(db)
    identities: list[AdvisorIdentity] = _cache["identities"]
    if not text or not identities:
        return AdvisorResolution(status=NOT_FOUND)

    # ---- stage 1+2: extract spans, resolve each ----
    best_span_result: ResolvedAdvisor | None = None
    for span in extract_name_spans(text):
        result = resolve_advisor(span, db, floor=floor)
        if result.status != NOT_FOUND:
            # record the SPAN that produced the match, not the gazetteer
            # name it matched — this is what makes a resolution auditable
            # ("which words did you think were the name?"), and it is the
            # question that was unanswerable when the whole sentence was
            # fed to the matcher.
            result.matched_text = span
        if result.status == RESOLVED:
            return result
        # an ambiguous span still beats no match — remember the first one
        # and keep looking for something unambiguous
        if result.status == AMBIGUOUS and best_span_result is None:
            best_span_result = result
    if best_span_result is not None:
        return best_span_result

    # ---- fallback: window scan over the raw text ----
    distinct_names = sorted({i.name for i in identities})
    hits = find_in_text(text, distinct_names, kind="advisor", floor=floor)
    if not hits:
        return AdvisorResolution(status=NOT_FOUND, matched_text=text)

    best_name, best_score = hits[0]

    # Decisiveness: a near-tie between two DIFFERENT names is ambiguous,
    # even though each individually cleared the floor.
    rivals = [
        (n, s) for n, s in hits[1:]
        if n.lower() != best_name.lower() and best_score - s <= AMBIGUITY_MARGIN
    ]
    if rivals:
        candidates: list[AdvisorIdentity] = []
        for name, score in [(best_name, best_score), *rivals]:
            for identity in _cache["by_name"].get(name.lower(), []):
                candidates.append(_with_score(identity, score))
        return AdvisorResolution(status=AMBIGUOUS, candidates=candidates, matched_text=best_name)

    # One winning NAME — but that name may still map to several people.
    people = _cache["by_name"].get(best_name.lower(), [])
    scored = [_with_score(p, best_score) for p in people]
    if not scored:
        return AdvisorResolution(status=NOT_FOUND, matched_text=best_name)
    if len(scored) == 1:
        return AdvisorResolution(status=RESOLVED, candidates=scored, matched_text=best_name)
    return AdvisorResolution(status=AMBIGUOUS, candidates=scored, matched_text=best_name)


def _with_score(identity: AdvisorIdentity, score: float) -> AdvisorIdentity:
    """`replace()` rather than a field-by-field rebuild: this used to name
    each field explicitly, which silently dropped any field added later.
    That would have made relationship inference work on exact name
    matches and return None on fuzzy ones — flakiness, not an obvious
    bug. Copying by construction removes the failure mode instead of
    documenting it."""
    return dataclasses.replace(identity, score=score)


def resolve_choice(choice: str, candidates: list[AdvisorIdentity]) -> AdvisorIdentity | None:
    """Match a user's answer to a disambiguation question ("the one in
    North/KPK", "wid 36041") against the candidates we offered. Returns
    None when the answer doesn't clearly pick one — the caller re-asks
    rather than guessing again."""
    if not choice or not candidates:
        return None
    needle = choice.strip().lower()

    for candidate in candidates:
        if needle == str(candidate.wid):
            return candidate
    for candidate in candidates:
        if needle == candidate.label().lower():
            return candidate

    # a distinguishing context word (team or company) that matches exactly
    # one candidate is an unambiguous answer
    #
    # Token-aware: substring containment let a short team name match
    # inside an unrelated word in the user's answer, and the consequence
    # here is picking the WRONG PERSON — the one thing this whole
    # disambiguation exists to avoid.
    from app.llm import token_match

    matched = [
        c for c in candidates
        if (c.team and token_match.contains(needle, c.team))
        or (c.company and token_match.contains(needle, c.company))
    ]
    if len(matched) == 1:
        return matched[0]
    return None


def _reset_for_tests() -> None:
    _cache["identities"] = []
    _cache["by_name"] = {}
    _cache["loaded_at"] = 0.0
