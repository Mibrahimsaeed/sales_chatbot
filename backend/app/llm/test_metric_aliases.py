"""One alias registry, and a percentage never resolving to a count.

THE DEFECT. Aliases were declared inside each MetricDef, which made them
invisible as a set. Nobody could see that "answered calls %" and
"answered calls" both landed on the same COUNT, or that "cr rate" and
"conversion rate" did too. A percentage phrase resolving to a count is
not a near miss — it returns 47 where the reader expects 68%, formatted
identically to a right answer.

THE FIX has two halves, and the second is the one that matters:

1. Every phrase now lives in metric_aliases.ALIASES, and MetricDef
   derives `synonyms` from it. One place a phrasing is written down.

2. A rate is a DIFFERENT MEASURE from its count, and gets its own entry.
   Where the components exist, that entry is a real RATIO metric
   (Connect->CR, CR->Meeting, Meeting->Conversion). Where they do not —
   the spec's target rates are
   `value / (teamSize x perDayTarget x workingDays) x 100` and there is
   no working-day calendar — the phrase is registered as UNAVAILABLE with
   the reason and the nearest available measure. Asking for one produces
   "I need a working-day calendar; I can give you the count" instead of
   silently handing back that count.

An unknown phrase and a known-but-uncomputable one are different
situations and now read differently.
"""

import pytest

from app.database.models import Advisor, Calls, Performance, PerformancePeriod, SalesFunnel
from app.llm import entity_extractor, metric_aliases
from app.llm.entity_extractor import extract_entities
from app.llm.metric_ontology import METRICS, Rollup, resolve_metric, resolve_metric_evidence
from app.llm.preprocessing import normalize
from app.llm.query_compiler import compile_and_run
from app.llm.query_ir import MetricRef, QueryIR, Sort
from app.llm.query_planner import build_query_plan


@pytest.fixture()
def funnel(db_session):
    """A funnel with clean ratios so a count/rate mix-up is unmissable.

      Adv A: 100 connects, 200 answered calls, 50 cr, 20 meetings, 5 conversions
      Adv B:  50 connects, 100 answered calls, 10 cr,  5 meetings, 1 conversion

    connect->cr    60/300 = 20.0%      (FIX 3: over ANSWERED CALLS, per
                                        the spec — not over connects,
                                        which would give 60/150 = 40%)
    cr->meeting    25/60  = 41.67%
    meeting->conv   6/25  = 24.0%
    """
    rows = [(1, "Adv A", 100, 200, 50, 20, 5), (2, "Adv B", 50, 100, 10, 5, 1)]
    for wid, name, connects, answered, cr, meetings, conversions in rows:
        db_session.add(Advisor(wid=wid, name=name, team="Blue Area",
                               company="Graana", in_master_sheet=True))
        db_session.add(SalesFunnel(
            wid=wid, mtd_new_connect=connects, mtd_followup_connect=0,
            mtd_cr=cr, mtd_new_meeting=meetings, mtd_followup_meeting=0,
            mtd_conversion=conversions,
        ))
        db_session.add(Calls(wid=wid, answered_calls_mtd=answered, connects_mtd=connects))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=50, pct=50))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    return db_session


def _value(db, metric, level="team"):
    ir = QueryIR(intent="leaderboard", subject_level=level,
                 metric=MetricRef(key=metric), sort=Sort(metric=metric))
    rows = compile_and_run(db, ir)
    return rows[0]["value"] if rows else None


# ---------------------------------------------------------------------
# The reported problems
# ---------------------------------------------------------------------

def test_answered_calls_percentage_does_not_resolve_to_the_count():
    """The headline case. "answered calls %" resolved to answered_calls,
    a raw count.

    RETIRED REFUSAL. This asserted the phrase resolved to NOTHING, with a
    written reason about a missing working-day calendar.
    working_days.py supplies that calendar now, so the rate is computable
    and the phrase resolves to it. The original point — a percentage
    question must not be answered with a raw count — is unchanged and
    still asserted on the first line.
    """
    assert resolve_metric("answered calls %") != "answered_calls"
    assert resolve_metric("answered calls %") == "answered_calls_rate"


def test_answered_calls_without_a_percent_is_still_the_count():
    """The count must keep working — it is a real measure."""
    assert resolve_metric("answered calls") == "answered_calls"


def test_cr_percent_does_not_resolve_to_the_count():
    """"CR%" was unresolved, then (Step 4) resolved to the client
    registration COUNT. Both are wrong for a percentage question."""
    # RETIRED REFUSAL — see the note on answered calls % above. Both
    # phrases now reach the rate; neither reaches the count.
    assert resolve_metric("cr %") == "cr_rate"
    assert resolve_metric("cr rate") == "cr_rate"
    assert resolve_metric("cr %") != "client_registrations"


def test_cr_without_a_percent_is_still_the_count():
    assert resolve_metric("cr") == "client_registrations"
    assert resolve_metric("client registrations") == "client_registrations"


def test_conversion_rate_resolves_to_a_rate_not_a_count():
    """"conversion rate" used to resolve to the conversion COUNT. In the
    spec, Conversion's rate is Conversion / Meetings x 100 — computable,
    so it is computed."""
    assert resolve_metric("conversion rate") == "meeting_to_conversion_rate"
    assert resolve_metric("conversion %") == "meeting_to_conversion_rate"
    assert resolve_metric("conversion") == "conversion"


# ---------------------------------------------------------------------
# Longest-first is what separates a rate from its count
# ---------------------------------------------------------------------

@pytest.mark.parametrize("rate_phrase,count_phrase", [
    ("answered calls %", "answered calls"),
    ("cr rate", "cr"),
    ("conversion rate", "conversion"),
    ("meeting rate", "meetings"),
])
def test_a_rate_phrase_beats_the_count_inside_it(rate_phrase, count_phrase):
    """Every rate phrase CONTAINS its count phrase. Without longest-first
    ordering across the whole registry, the count always wins and every
    percentage question silently becomes a count."""
    rate = metric_aliases.resolve(rate_phrase)
    count = metric_aliases.resolve(count_phrase)

    assert count is not None and count.available
    assert rate is not None
    assert rate.metric != count.metric, rate_phrase


# ---------------------------------------------------------------------
# All KPI names from the spec
# ---------------------------------------------------------------------

# Every leaderboard in the spec's formula table, plus the phrasings its
# example questions actually use. `None` means "declared, not computable"
# — asserted as a refusal WITH a reason, never as a count.
SPEC_KPIS = [
    # (phrase, expected metric key or None)
    ("performance %", "achievement_pct"),
    ("performance rate", "achievement_pct"),
    ("achievement %", "achievement_pct"),
    ("revenue", "mtd_cleared"),
    ("portfolio value", "portfolio_value"),
    ("pipeline value", "pipeline_value"),
    ("overdue count", "overdue"),
    ("answered calls", "answered_calls"),
    ("connects", "total_connects"),
    ("meetings", "total_meetings"),
    ("conversions", "conversion"),
    ("client registrations", "client_registrations"),
    ("bookings", "bookings"),
    ("attendance rate", "attendance_rate"),
    ("connect to cr", "connect_to_cr_rate"),
    ("cr to meeting", "cr_to_meeting_rate"),
    ("meeting to conversion", "meeting_to_conversion_rate"),
    ("conversion rate", "meeting_to_conversion_rate"),
    # RETIRED REFUSALS. These six were declared uncomputable for want of
    # a working-day calendar; working_days.py is that calendar.
    ("answered calls %", "answered_calls_rate"),
    ("answered call rate", "answered_calls_rate"),
    ("connect rate", "answered_calls_rate"),
    ("cr %", "cr_rate"),
    ("cr rate", "cr_rate"),
    ("meeting rate", "meeting_rate"),
    # RETIRED REFUSALS. Both were declared uncomputable ("I don't track
    # which advisors have units yet") until the ETL imported the
    # "1 Unit" tab. A declared refusal is superseded the moment its data
    # lands.
    ("1 unit ratio", "one_unit_ratio"),
    ("1-unit ratio", "one_unit_ratio"),
    # Newly bound alongside it.
    ("worksapp login", "login_rate"),
    ("meetings planned", "meetings_planned"),
    ("meetings conducted", "meetings_conducted"),
    ("meeting conduction rate", "meeting_conduction_rate"),
    # Also retired, for the same reason: working_days.py.
    ("meeting rate", "meeting_rate"),
    ("connect rate", "answered_calls_rate"),
]


@pytest.mark.parametrize("phrase,expected", SPEC_KPIS)
def test_every_spec_kpi_name_resolves_or_refuses_explicitly(phrase, expected):
    """No spec KPI may resolve to something of the wrong SHAPE. Either it
    names a real metric, or it is declared uncomputable with a reason —
    never silently a neighbouring count."""
    match = metric_aliases.resolve(phrase)
    assert match is not None, f"{phrase!r} is not in the registry at all"

    if expected is None:
        assert not match.available, f"{phrase!r} should be declared unavailable"
        assert match.reason, phrase
    else:
        assert match.metric == expected, phrase
        assert match.metric in METRICS, phrase


@pytest.mark.parametrize("phrase", [p for p, e in SPEC_KPIS if e is None])
def test_an_unavailable_kpi_never_resolves_to_a_metric(phrase):
    assert resolve_metric(phrase) is None, phrase
    assert resolve_metric_evidence(phrase) is None, phrase


# ---------------------------------------------------------------------
# The new ratio metrics compute real percentages
# ---------------------------------------------------------------------

@pytest.mark.parametrize("metric,expected", [
    ("connect_to_cr_rate", 20.0),          # FIX 3: 60 / 300 answered calls
    ("cr_to_meeting_rate", 41.67),         # 25 / 60
    ("meeting_to_conversion_rate", 24.0),  # 6 / 25
])
def test_a_funnel_ratio_is_the_ratio_of_sums(funnel, metric, expected):
    """Rollup.RATIO, so a group's ratio divides summed components rather
    than averaging per-advisor ratios — the Phase 4 rule. Averaging Adv A
    and Adv B would give a different number."""
    assert _value(funnel, metric) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("metric", [
    "connect_to_cr_rate", "cr_to_meeting_rate", "meeting_to_conversion_rate",
])
def test_a_funnel_ratio_is_declared_as_a_ratio(metric):
    assert METRICS[metric].rollup is Rollup.RATIO
    binding = METRICS[metric].bindings["advisor"]
    assert binding.ratio_numerator is not None
    assert binding.ratio_denominator is not None


def test_a_funnel_ratio_is_answerable_at_every_group_level(funnel):
    from app.llm.query_compiler import is_answerable

    for level in ("advisor", "team", "bcm", "unit_head", "zonal_head", "company"):
        assert is_answerable("connect_to_cr_rate", level), level


def test_a_ratio_differs_from_its_numerator_count(funnel):
    """The concrete reason this matters: the rate and the count are
    different numbers, so substituting one was never harmless."""
    assert _value(funnel, "conversion") == 6
    assert _value(funnel, "meeting_to_conversion_rate") == pytest.approx(24.0)


# ---------------------------------------------------------------------
# One source of truth
# ---------------------------------------------------------------------

def test_metric_def_synonyms_come_from_the_registry():
    for key, metric in METRICS.items():
        assert metric.synonyms == metric_aliases.phrases_for(key), key


def test_every_metric_is_nameable():
    """A metric with no alias exists but cannot be asked for."""
    for key, metric in METRICS.items():
        assert metric.synonyms, key


def test_every_alias_points_at_a_real_metric():
    """A typo'd key here would resolve to a metric that does not exist,
    and the planner would report it as unanswerable."""
    for key in metric_aliases.ALIASES:
        assert key in METRICS, key


def test_no_phrase_is_claimed_twice():
    """Two entries for one phrase resolve by index order, silently."""
    seen: dict[str, str] = {}
    for key, phrases in metric_aliases.ALIASES.items():
        for phrase in phrases:
            assert phrase not in seen, f"{phrase!r}: {seen.get(phrase)} and {key}"
            seen[phrase] = key
    for entry in metric_aliases.UNAVAILABLE:
        for phrase in entry.phrases:
            assert phrase not in seen, f"{phrase!r}: {seen.get(phrase)} and {entry.key}"
            seen[phrase] = entry.key


def test_no_unavailable_key_collides_with_a_real_metric():
    """An unavailable key naming a real metric would be a contradiction:
    declared uncomputable while a binding exists."""
    for key in metric_aliases.unavailable_keys():
        assert key not in METRICS, key


def test_every_suggested_alternative_is_a_real_metric():
    for entry in metric_aliases.UNAVAILABLE:
        if entry.instead is not None:
            assert entry.instead in METRICS, entry.key


# ---------------------------------------------------------------------
# Short aliases stay safe
# ---------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "across the board", "describe the increase", "the acre count", "recruit more",
])
def test_a_short_alias_does_not_fire_inside_a_word(text):
    assert resolve_metric(text) != "client_registrations", text


def test_an_unavailable_phrase_is_not_fuzzy_widened_to_a_neighbour():
    """"cr %" would fuzzy-match the client registration COUNT, handing
    back a count for a percentage question — the exact substitution the
    registry exists to stop, sneaking in through the widening tier."""
    from app.llm.fallback_reasoning import fuzzy_resolve_metric

    # RETIRED REFUSAL, same guarantee. These resolve EXACTLY to the rate
    # now, and the exact hit short-circuits before the synonym scan — so
    # the substitution this test guards against (a percentage question
    # answered with a count) is still impossible. It nearly returned
    # through the scan when the refusal was retired: "cr %" contains
    # "cr", the count's own synonym.
    assert fuzzy_resolve_metric("cr %") == "cr_rate"
    assert fuzzy_resolve_metric("answered calls %") == "answered_calls_rate"
    # P0: the APPROXIMATE tier is off, so a typo is no longer widened —
    # a wrong measure reads exactly like a right one, and the exact tiers
    # above (which guess nothing) are what remain.
    assert fuzzy_resolve_metric("revnue") is None


# ---------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------

def _plan(text, db):
    cleaned = normalize(text)
    return build_query_plan(cleaned, extract_entities(cleaned, db))


def test_a_percentage_question_asks_rather_than_returning_a_count(funnel):
    """INVERTED: it now ANSWERS rather than asking, because the
    working-day calendar it was waiting for exists. The invariant the
    name describes is unchanged and still asserted — a percentage
    question never comes back as the raw count."""
    plan = _plan("Which team has the highest answered calls %", funnel)

    assert plan.action != "clarify_metric"
    assert plan.metric != "answered_calls"
    assert plan.metric == "answered_calls_rate"


def test_the_refusal_names_the_missing_ingredient_and_an_alternative(funnel):
    """INVERTED. "Which team has the highest CR %" used to refuse, naming
    the missing working-day calendar and offering the CR count instead.
    working_days.py supplies that calendar, so the question is answered
    rather than deflected — and the refusal it replaced was the whole
    reason this test existed.

    What is still asserted is that the phrase reaches the RATE, not the
    count it used to be redirected to: a wrong-measure answer was the
    failure the refusal was protecting against.
    """
    from app.llm.nlu_pipeline import resolve

    resolution = resolve("Which team has the highest CR %", funnel, session_id=None)

    assert resolution.kind != "clarify"
    plan = getattr(resolution, "plan", None)
    metric = (resolution.ir.metric.key if resolution.kind == "ir" and resolution.ir.metric
              else (plan.metric if plan else None))
    assert metric == "cr_rate"


def test_a_computable_rate_question_is_answered(funnel):
    plan = _plan("top teams by conversion rate", funnel)
    assert plan.action == "leaderboard"
    assert plan.metric == "meeting_to_conversion_rate"


def test_a_count_question_is_still_answered(funnel):
    plan = _plan("top teams by answered calls", funnel)
    assert plan.action == "leaderboard"
    assert plan.metric == "answered_calls"


# ---------------------------------------------------------------------
# Phase 5.5 — business vocabulary, and percent spellings
# ---------------------------------------------------------------------

# The terms the phase requires, with what each must resolve to.
# `None` means declared-uncomputable — asserted as a refusal WITH a
# reason, never as a silent fall-through to a neighbouring count.
REQUIRED_VOCABULARY = [
    ("CR", "client_registrations"),
    # Retired refusals: computable since working_days.py.
    ("CR%", "cr_rate"),
    ("Answered Calls", "answered_calls"),
    ("Answered Call %", "answered_calls_rate"),
    ("Conversion", "conversion"),
    ("Conversion %", "meeting_to_conversion_rate"),
    ("Achievement", "achievement_pct"),
    ("Achievement %", "achievement_pct"),
    ("Pipeline", "pipeline_value"),
    ("Pipeline Value", "pipeline_value"),
    ("Bookings", "bookings"),
    ("Revenue", "mtd_cleared"),
    ("Performance", "achievement_pct"),
    ("Attendance", "attendance_rate"),
    ("Late Arrival", "late_count"),
]


@pytest.mark.parametrize("term,expected", REQUIRED_VOCABULARY)
def test_every_required_business_term_is_understood(term, expected):
    match = metric_aliases.resolve(term)
    assert match is not None, f"{term!r} is not in the registry at all"

    if expected is None:
        assert not match.available, f"{term!r} should be declared unavailable"
        assert match.reason, term
        assert resolve_metric(term) is None, term
    else:
        assert match.metric == expected, term
        assert match.metric in METRICS, term


@pytest.mark.parametrize("term,expected", REQUIRED_VOCABULARY)
def test_case_does_not_change_the_answer(term, expected):
    """Users type "CR", "cr" and "Cr". Matching is case-insensitive and
    must stay so."""
    for variant in (term.lower(), term.upper(), term.title()):
        got = metric_aliases.resolve(variant)
        assert got is not None, variant
        assert got.metric == (expected if expected else None), variant


# ---------------------------------------------------------------------
# The percent-spelling class
# ---------------------------------------------------------------------

@pytest.mark.parametrize("stem", [
    "cr", "answered calls", "conversion", "achievement", "performance", "attendance",
])
def test_every_percent_spelling_of_a_rate_agrees(stem):
    """THE defect this phase fixes. A rate phrase only beats the count
    inside it if that exact spelling is declared, so "cr %" refused while
    "cr%" — one space away — returned the client-registration COUNT.

    All five spellings must resolve identically, or a typing variant
    silently changes the measure."""
    spellings = [f"{stem}%", f"{stem} %", f"{stem} percent",
                 f"{stem} percentage", f"{stem} pct"]
    results = {s: metric_aliases.resolve(s) for s in spellings}

    assert all(r is not None for r in results.values()), results
    metrics = {r.metric for r in results.values()}
    assert len(metrics) == 1, f"{stem}: spellings disagree -> {metrics}"


@pytest.mark.parametrize("stem,count_metric", [
    ("cr", "client_registrations"),
    ("answered calls", "answered_calls"),
    ("conversion", "conversion"),
])
def test_no_percent_spelling_falls_through_to_its_count(stem, count_metric):
    """The consequence, stated directly: a "%" phrase must never resolve
    to the raw count sitting inside it."""
    assert metric_aliases.resolve(stem).metric == count_metric
    for spelling in (f"{stem}%", f"{stem} %", f"{stem} percent",
                     f"{stem} percentage", f"{stem} pct"):
        assert metric_aliases.resolve(spelling).metric != count_metric, spelling


def test_expand_percent_leaves_a_non_percentage_phrase_alone():
    assert metric_aliases.expand_percent("revenue") == ("revenue",)
    assert metric_aliases.expand_percent("pipeline value") == ("pipeline value",)


def test_expand_percent_is_idempotent_across_markers():
    """Declaring the phrase with ANY marker yields the same set, so which
    spelling an author happens to write down cannot matter."""
    from_pct = set(metric_aliases.expand_percent("cr %"))
    from_word = set(metric_aliases.expand_percent("cr percentage"))
    from_tight = set(metric_aliases.expand_percent("cr%"))
    assert from_pct == from_word == from_tight


# ---------------------------------------------------------------------
# Ambiguity
# ---------------------------------------------------------------------

def test_generated_spellings_never_overwrite_a_declaration():
    """An explicit entry must win over another phrase's generated
    variant, or adding a synonym could silently re-point an existing
    one."""
    seen: dict[str, str] = {}
    for phrase, match in metric_aliases._INDEX:
        key = match.metric or "<unavailable>"
        assert phrase not in seen, f"{phrase!r} claimed twice: {seen[phrase]} and {key}"
        seen[phrase] = key


def test_a_count_and_its_rate_are_different_metrics():
    """The pairs this whole registry exists to keep apart."""
    pairs = [("conversion", "conversion %"), ("cr", "cr %"),
             ("answered calls", "answered calls %")]
    for count_phrase, rate_phrase in pairs:
        count = metric_aliases.resolve(count_phrase)
        rate = metric_aliases.resolve(rate_phrase)
        assert count.available, count_phrase
        assert count.metric != rate.metric, f"{count_phrase} and {rate_phrase} agree"


def test_a_percentage_phrase_never_resolves_to_a_raw_value_metric():
    """A "%" question answered with a raw sum reads as a percentage to
    anyone skimming — the same failure as a count."""
    raw_value_metrics = {"portfolio_value", "pipeline_value", "mtd_cleared",
                         "ytd_cleared", "overdue_amount", "returned_value"}
    for phrase, match in metric_aliases._INDEX:
        if not any(phrase.endswith(m) for m in ("%", "percent", "percentage", "pct")):
            continue
        assert match.metric not in raw_value_metrics, (
            f"{phrase!r} resolves to the raw value {match.metric}"
        )


@pytest.mark.parametrize("phrase", [
    "across the board", "describe the increase", "the acre count",
    "recruitment", "performance review process",
])
def test_expanded_vocabulary_did_not_create_new_collisions(phrase):
    """Every new synonym is short business jargon; token matching must
    keep them out of unrelated words."""
    got = metric_aliases.resolve(phrase)
    assert got is None or got.metric != "client_registrations", phrase
