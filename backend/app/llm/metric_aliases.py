"""
THE metric alias registry. Every phrase a user might type for a measure,
and the metric key it means.

WHY IT EXISTS. Aliases were declared inside each MetricDef, which made
them invisible as a set: nobody could see that "answered calls %" and
"answered calls" both landed on the same COUNT, or that "cr rate" did
too. A percentage phrase resolving to a count is not a near miss — it
returns 47 where the user expected 68%, formatted identically to a right
answer.

Two things live here that a per-metric synonym list could not express:

1. RATE vs COUNT is explicit. `answered calls` is a count; `answered
   calls %` is a rate. They are different measures and get different
   entries, so one can never silently stand in for the other.

2. UNAVAILABLE measures are DECLARED. Several of the spec's rate KPIs
   are `value / (teamSize x perDayTarget x workingDays) x 100`, and this
   system has no working-day calendar. Those phrases are registered with
   no metric and an explanation, so asking for one produces "I can't
   compute that yet, here is what I can give you" instead of quietly
   handing back the underlying count. An unknown phrase and a known-but-
   uncomputable one are different situations and now read differently.

WHAT THIS MODULE DOES NOT DO. It maps words to metric KEYS. It never
defines how a metric is computed — that stays in metric_ontology's
MetricDef/ColumnBinding. Adding a phrase here can change which existing
measure a question resolves to; it can never change a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# =====================================================================
# Phrases per metric key
#
# Moved here verbatim from the MetricDef declarations, which now derive
# `synonyms` from this table — so there is exactly one place a phrasing
# is written down.
# =====================================================================

ALIASES: dict[str, tuple[str, ...]] = {
    # ---- performance / revenue ----
    "achievement_pct": (
        "target achievement", "achievement", "hit rate", "on target",
        "achieved target", "target hit", "performance against target",
        "top performer", "performer", "performance",
        # The spec calls this the Performance leaderboard's "performance
        # rate" — pct = Cleared / Target x 100. Same measure, business
        # wording, and the only target-rate KPI whose denominator this
        # system actually stores.
        # One declaration per rate form — expand_percent generates
        # "%", " %", " percent", " percentage" and " pct" for each.
        "performance %", "performance rate", "perf %",
        "achievement %", "achievement rate", "target achievement %",
    ),
    "mtd_cleared": (
        "revenue", "cleared", "sales", "closed revenue", "closed",
        "highest sales", "highest closer", "top closer",
    ),
    "ytd_cleared": (
        "ytd cleared", "ytd revenue", "year to date revenue",
        "year to date cleared", "annual cleared",
    ),
    "three_month_cleared": (
        "3 month cleared", "three month cleared", "quarterly cleared",
        "3m cleared", "quarter revenue",
    ),
    "mtd_target": (
        "mtd target", "monthly target", "this month's target",
        "month target", "target", "targets",
    ),

    # ---- funnel counts ----
    "total_connects": (
        "connects", "connections", "connect", "total connects",
        "all connects", "most connects",
    ),
    "new_connects": ("new connects", "new connect", "fresh connects", "first connects"),
    "followup_connects": (
        "follow-up connects", "followup connects", "follow up connects",
        "repeat connects",
    ),
    # "meetings conducted" moved to the IBD metric below — it names that
    # board specifically, whereas this is the raw new+follow-up count
    # from CCMC. Two different figures from two different tabs.
    "total_meetings": ("meetings", "meeting", "total meetings", "most meetings",
                       "meetings held"),
    # A COUNT of conversions. The rate phrasings deliberately no longer
    # point here — see meeting_to_conversion_rate below; a rate is not a
    # count, and letting one stand in for the other is the whole bug.
    "conversion": ("conversion", "conversions", "raw conversions", "conversion count"),
    "client_registrations": (
        "client registration", "client registrations", "cr", "crs", "cr count",
    ),
    "bookings": (
        "booking", "bookings", "booked units", "booked", "units booked",
        "booking stored", "booking value",
        "cr booked", "cr bookings", "crs booked",
    ),
    "answered_calls": (
        "answered calls", "calls answered", "call answered", "answered call count",
    ),

    # ---- value ----
    "pipeline_value": ("pipeline", "pipeline value", "open pipeline", "open deals"),
    # "overdue amount" folded in — see the note in metric_ontology where
    # the duplicate MetricDef was removed. One sheet column, one measure.
    "overdue": ("overdue", "overdues", "past due", "overdue count",
                "overdue amount", "overdue value", "amount overdue"),
    "portfolio_value": ("portfolio", "portfolio value", "book size"),
    "returned_value": ("returned", "returned value", "returns", "portfolio returned"),

    # ---- 1 Unit, Login, IBD meetings (bound once the ETL imported
    # their sources; each was previously declared uncomputable) ----
    "one_unit_ratio": (
        "1 unit ratio", "one unit ratio", "1-unit ratio", "1 unit",
        "one unit", "1-unit", "1 unit %", "unit ratio", "unit ownership",
        "advisors with units",
    ),
    "login_rate": (
        "login rate", "worksapp login", "worksapp login rate",
        "login on time rate", "on time login rate", "login %",
        "login punctuality",
    ),
    "meetings_planned": (
        "meetings planned", "planned meetings", "meeting planned",
        "meetings scheduled",
    ),
    "meetings_conducted": (
        "meetings conducted", "conducted meetings", "meeting conducted",
        "meetings done",
    ),
    "meeting_conduction_rate": (
        "meeting conduction rate", "conduction rate", "conduction ratio",
        "meetings conducted rate", "meeting conduction %",
        "conducted vs planned", "meetings conducted ratio",
    ),

    # ---- attendance ----
    "late_count": (
        "late count", "how many late", "number of late",
        "late arrival", "late arrivals", "lates", "tardiness", "late",
    ),
    # Working-day scaled rates. These three lived in UNAVAILABLE until
    # working_days.py gave `workingDays` a source; the phrases are
    # carried over unchanged so every wording that used to get the
    # refusal now gets the number.
    "cr_rate": (
        "cr %", "cr percentage", "cr rate", "client registration rate",
        "client registration %", "client registration percentage",
        "registration rate", "cr ratio",
    ),
    "ytd_cr_rate": (
        "ytd cr %", "ytd cr rate", "year to date cr %", "year to date cr rate",
        "ytd client registration rate",
    ),
    "answered_calls_rate": (
        "answered call %", "answered calls %", "answered call rate",
        "answered-call rate", "answered calls rate", "answered %",
        "answered call percentage", "connect rate", "connect %",
        "connect percentage", "connect percent",
    ),
    "meeting_rate": (
        "meeting rate", "meetings rate", "meeting %", "meeting percentage",
        "meetings percentage", "meetings %",
    ),

    "attendance_rate": (
        "attendance rate", "attendance percentage", "attendance %", "attendance",
        "on time rate", "on-time percentage", "punctuality",
    ),

    # ---- funnel ratios (the spec's Connect->CR / CR->Meeting /
    # Meeting->Conversion leaderboards). Every component is a column this
    # system already stores, so unlike the target rates below these are
    # real, computable percentages.
    "connect_to_cr_rate": (
        "connect to cr", "connect-to-cr", "connect to cr ratio",
        "connect to cr rate", "connect to client registration",
        "cr per connect", "cr conversion rate",
    ),
    "cr_to_meeting_rate": (
        "cr to meeting", "cr-to-meeting", "cr to meeting ratio",
        "cr to meeting rate", "meeting per cr", "meetings per cr",
    ),
    "meeting_to_conversion_rate": (
        "meeting to conversion", "meeting-to-conversion",
        "meeting to conversion ratio", "meeting to conversion rate",
        "conversion per meeting", "meeting conversion rate",
        # A bare "conversion rate" lands here, not on the conversion
        # COUNT it used to. In the spec "CR" is Client Registration and
        # "Conversion" is a later, separate stage, so the rate attached
        # to conversions is Conversion / Meetings x 100 — the one
        # conversion percentage whose components this system stores.
        "conversion rate", "conversion %", "conversion percentage",
        "conversion ratio",
    ),
}


# YTD phrasings. Generated from the MTD stems rather than written out,
# because "ytd X" / "year to date X" / "X ytd" is the same three-way
# pattern for every measure and nine hand-written blocks is nine chances
# to mistype one.
#
# These beat their MTD counterparts because the index is longest-first:
# "ytd connects" (12) is matched before "connects" (8).
_YTD_STEMS: dict[str, tuple[str, ...]] = {
    "ytd_connects": ("connects", "connections"),
    "ytd_new_connects": ("new connects",),
    "ytd_followup_connects": ("follow-up connects", "followup connects"),
    "ytd_client_registrations": ("client registrations", "cr", "crs"),
    "ytd_meetings": ("meetings",),
    "ytd_conversion": ("conversions", "conversion"),
    "ytd_bookings": ("bookings",),
    "ytd_pipeline_value": ("pipeline", "pipeline value"),
    "ytd_overdue": ("overdue",),
}

for _key, _stems in _YTD_STEMS.items():
    ALIASES[_key] = tuple(dict.fromkeys(
        phrase
        for stem in _stems
        for phrase in (f"ytd {stem}", f"year to date {stem}", f"{stem} ytd",
                       f"{stem} year to date")
    ))


# DAILY phrasings, for the three measures that have real daily data
# (Phase 12 — `calls.answered_calls_daily` and `calls.connects_daily`).
# Same generated-from-stems shape as the YTD block above and for the same
# reason.
#
# THESE ARE NOT A SECOND PERIOD RESOLVER. "connects today" never reaches
# here as a period at all: temporal_parser reads the window, and
# resolve_metric_for_period swaps `total_connects` for `daily_connects`
# through the shared period_family. What this block adds is only that the
# daily KEY is nameable — which metric_ontology requires of every metric,
# and which the YTD keys have had all along. The two routes are
# convergent by construction: both end at the same key.
_DAILY_STEMS: dict[str, tuple[str, ...]] = {
    "daily_connects": ("connects", "connections"),
    "daily_answered_calls": ("answered calls", "calls answered"),
    "daily_answered_calls_rate": ("answered calls %", "answered call rate",
                                  "connect %", "connect rate"),
}

for _key, _stems in _DAILY_STEMS.items():
    ALIASES[_key] = tuple(dict.fromkeys(
        phrase
        for stem in _stems
        for phrase in (f"daily {stem}", f"today's {stem}", f"{stem} today",
                       f"{stem} daily")
    ))


# =====================================================================
# Declared but not computable
# =====================================================================

@dataclass(frozen=True)
class Unavailable:
    """A measure the business genuinely has, that this system cannot
    compute yet.

    `phrases` resolve to NO metric. `reason` says what is missing and
    `instead` names the closest measure that IS available, so the reply
    can offer something concrete rather than a shrug.

    The alternative — mapping these onto the underlying count — is the
    bug this registry exists to remove: "what is the answered calls %"
    answered with 47 reads as a percentage to anyone skimming.
    """

    key: str
    phrases: tuple[str, ...]
    reason: str
    instead: Optional[str] = None


# The spec's target-rate leaderboards are all
# `value / (teamSize x perDayTarget x workingDays) x 100`. teamSize is
# derivable (aggregation.headcount) but workingDays is not — there is no
# working-day calendar anywhere in this system, and adding one is a
# separate piece of work with its own business rules (holidays, partial
# months, per-region calendars).
UNAVAILABLE: tuple[Unavailable, ...] = (
    Unavailable(
        key="portfolio_rate",
        phrases=("portfolio %", "portfolio rate", "portfolio percentage of target"),
        reason=(
            "portfolio is a raw value with no percentage form — there is no "
            "portfolio target to measure it against"
        ),
        instead="portfolio_value",
    ),
    # NOTE: `one_unit_ratio` used to be declared here as uncomputable
    # ("I don't track which advisors have units yet"). The ETL now
    # imports the "1 Unit" tab, so it is a real metric above and the
    # refusal is retired. Left as a comment because the reasoning — a
    # declared refusal is superseded the moment its data lands — is the
    # thing worth remembering.
)

# =====================================================================
# Lookup
# =====================================================================

@dataclass(frozen=True)
class AliasMatch:
    """What a phrase resolved to.

    `metric` is None for a declared-but-unavailable measure; `reason` and
    `instead` are populated only in that case. A phrase this registry has
    never heard of produces no AliasMatch at all — the caller must be
    able to tell "I know this and can't do it" from "I don't know this".
    """

    phrase: str
    metric: Optional[str]
    reason: Optional[str] = None
    instead: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.metric is not None


# The ways a user writes "percent". Interchangeable in speech, and each
# one has to be spelled out somewhere or the phrase falls through.
_PERCENT_MARKERS: tuple[str, ...] = ("%", " %", " percent", " percentage", " pct")


def expand_percent(phrase: str) -> tuple[str, ...]:
    """`phrase` in every spelling of its percent marker.

    THE CLASS THIS FIXES. A rate phrase only beats the count inside it if
    the rate phrase is declared. Declaring "cr %" but not "cr%" meant a
    single missing space sent the query to the client-registration COUNT
    — the exact percentage-resolves-to-count defect this registry exists
    to prevent, resurfacing on a typing variant. Same for "conversion
    percentage" vs "conversion percent".

    Enumerating spellings by hand across every rate phrase is how that
    keeps happening, so they are generated: declare the phrase once with
    any marker and all five spellings are registered.

    A phrase with no percent marker is returned unchanged.
    """
    for marker in sorted(_PERCENT_MARKERS, key=len, reverse=True):
        if phrase.endswith(marker):
            stem = phrase[: -len(marker)].rstrip()
            return tuple(dict.fromkeys(
                f"{stem}{m}" if m.startswith("%") else f"{stem}{m}"
                for m in _PERCENT_MARKERS
            ))
    return (phrase,)


def _build_index() -> list[tuple[str, AliasMatch]]:
    """(phrase, match) longest-first.

    Longest-first is load-bearing: "answered calls %" must beat "answered
    calls", and "cr rate" must beat "cr". Without it every rate phrase
    collapses onto the count sitting inside it, which is the defect.
    """
    entries: list[tuple[str, AliasMatch]] = []
    seen: set[str] = set()

    def _add(phrase: str, match_for) -> None:
        # Percent spellings are generated, not declared — see
        # expand_percent. First declaration wins, so an explicit entry is
        # never overwritten by another phrase's generated variant.
        for spelling in expand_percent(phrase):
            if spelling in seen:
                continue
            seen.add(spelling)
            entries.append((spelling, match_for(spelling)))

    for key, phrases in ALIASES.items():
        for phrase in phrases:
            _add(phrase, lambda spelling, k=key: AliasMatch(phrase=spelling, metric=k))
    for unavailable in UNAVAILABLE:
        for phrase in unavailable.phrases:
            _add(phrase, lambda spelling, u=unavailable: AliasMatch(
                phrase=spelling, metric=None, reason=u.reason, instead=u.instead,
            ))
    entries.sort(key=lambda pair: -len(pair[0]))
    return entries


_INDEX: list[tuple[str, AliasMatch]] = _build_index()


def resolve(text: str) -> Optional[AliasMatch]:
    """The measure this text names, available or not.

    Token-aware (app/llm/token_match.py) so a short alias like "cr"
    cannot fire inside "across" or "describe".
    """
    from app.llm import token_match

    lowered = text.lower()
    for phrase, match in _INDEX:
        if token_match.contains(lowered, phrase):
            return match
    return None


def resolve_all(text: str) -> list[AliasMatch]:
    """EVERY measure this text names, in the order it names them.

    `resolve()` above returns the first hit in an index sorted
    longest-phrase-first, which for "connects and answered calls" is
    `answered_calls` — so the other measure was discarded before anything
    downstream could know it had been asked for. Order was not even the
    tie-break: "answered calls and connects" resolved to `answered_calls`
    too, because the winner is whichever ALIAS STRING is longer.

    Same index, same matcher, same longest-first precedence — the only
    additions are that the scan continues after a hit and that each hit
    MASKS its own span before the next phrase is tried. Masking is what
    makes the matches non-overlapping: without it "answered calls" would
    be found, and then "calls" would be found again inside the text it
    had already claimed, reporting two measures where the user named one.
    token_match.mask is the same helper the ranking scan already uses to
    keep a comparator phrase from contributing the word inside it.

    Ordering is by POSITION IN THE TEXT, not by alias length, so the list
    reads the way the sentence does. Deduplicated by metric key, so two
    phrasings of one measure ("connects", "total connects") count once.

    Unavailable entries are returned too, with `metric=None` — a caller
    asking "what did they name?" needs to hear about a measure this
    system knows and cannot compute, or that half of the request would
    vanish exactly as the second metric used to.

    `resolve()` is untouched and every existing caller keeps its
    behaviour, including the longest-alias winner.
    """
    from app.llm import token_match

    lowered = text.lower()
    masked = lowered
    hits: list[tuple[int, AliasMatch]] = []
    seen: set = set()

    for phrase, match in _INDEX:
        found = token_match.find(masked, phrase)
        if found is None:
            continue
        masked = token_match.mask(masked, [phrase])
        # Key on the metric for available measures and on the phrase for
        # unavailable ones, which have no key to collide on.
        identity = match.metric or f"unavailable:{match.phrase}"
        if identity in seen:
            continue
        seen.add(identity)
        hits.append((found.start(), match))

    return [match for _, match in sorted(hits, key=lambda pair: pair[0])]


def phrases_for(metric_key: str) -> list[str]:
    """The declared phrasings for one metric. Read by metric_ontology to
    populate MetricDef.synonyms, so the ontology stays the place a metric
    is DEFINED while this stays the place it is NAMED."""
    return list(ALIASES.get(metric_key, ()))


def unavailable_keys() -> tuple[str, ...]:
    return tuple(entry.key for entry in UNAVAILABLE)


def explain(match: AliasMatch) -> str:
    """The reply for a measure we know about but cannot compute."""
    from app.llm.metric_ontology import metric_label

    message = f'I can\'t give you "{match.phrase}" yet — {match.reason}.'
    if match.instead:
        message += f" I can give you {metric_label(match.instead)} instead."
    return message
