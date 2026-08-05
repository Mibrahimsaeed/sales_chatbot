"""Step 4 — a default metric only when the user named none (finding F6).

`DEFAULT_RANKING_METRIC` fired whenever a strong ranking word was present
and `resolve_metric` came back empty. That test conflates two situations:

    "top 5 advisors"                 named no measure       -> default is fine
    "which BCM has the highest CR%"  named one, unresolved  -> default is a lie

Both looked identical, so the second became a ranking by MTD Revenue
Cleared — correctly formatted, confidently presented, and an answer to a
question nobody asked.

The distinction now has a name: metric_intent.MetricIntent, with
`may_default` and `unresolved` as separate properties. `unresolved`
produces a clarification that quotes the user's words back.

Widening still runs first. Exact -> residue -> fuzzy -> embedding, all
inside detect(), so a typo is resolved rather than refused. Refusing a
typo would just trade one wrong outcome for another.
"""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import entity_extractor, metric_intent
from app.llm.entity_extractor import extract_entities
from app.llm.intent_catalog import DEFAULT_RANKING_METRIC
from app.llm.preprocessing import normalize
from app.llm.query_planner import build_query_plan


@pytest.fixture()
def org(db_session):
    for wid, name, team in ((1, "Adv One", "Blue Area"), (2, "Adv Two", "Blue Area")):
        db_session.add(Advisor(wid=wid, name=name, team=team, company="Graana",
                               rm="UH Ali", portfolio_lead="ZH Sara",
                               management_lead="BCM Omar", in_master_sheet=True))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=50 * wid, pct=50 * wid))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=10 * wid,
                                   mtd_followup_connect=0, mtd_cr=5 * wid))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    return db_session


def _plan(text, db):
    cleaned = normalize(text)
    return build_query_plan(cleaned, extract_entities(cleaned, db))


def _intent(text, db):
    cleaned = normalize(text)
    return metric_intent.detect(cleaned, extract_entities(cleaned, db))


# ---------------------------------------------------------------------
# The three required cases
# ---------------------------------------------------------------------

def test_top_5_advisors_uses_the_default_metric(org):
    """Rule 1. No measure named, so choosing one fills a gap the user
    left open rather than overriding something they said."""
    intent = _intent("Top 5 advisors", org)
    assert intent.may_default
    assert not intent.unresolved

    plan = _plan("Top 5 advisors", org)
    assert plan.action == "leaderboard"
    assert plan.metric == DEFAULT_RANKING_METRIC
    assert plan.limit == 5
    assert f"default_metric:{DEFAULT_RANKING_METRIC}" in plan.intent_evidence


def test_top_advisors_by_cr_resolves_the_cr_metric(org):
    """CR is Client Registration — a real funnel stage with a real
    column that simply had no metric, which is why it used to fall
    through to the revenue default.

    RETIRED ASSERTION on the "%" form. Step 4 pinned
    `"Top advisors by CR%" -> client_registrations`, and its report
    flagged the compromise explicitly: the spec's CR RATE needs a
    working-day calendar, so the answer was the COUNT under a percentage
    question. Phase 5.5 generates every percent spelling, so "CR%" now
    reaches the same declared-unavailable entry "CR %" always did — a
    refusal that names the missing ingredient beats a count wearing a
    percent sign. The bare count is unchanged."""
    intent = _intent("Top advisors by CR", org)
    assert intent.resolved
    assert intent.key == "client_registrations"
    assert _plan("Top advisors by CR", org).metric == "client_registrations"

    # The percentage form reaches the RATE. It refused for want of a
    # working-day calendar until working_days.py supplied one; the
    # guarantee this line has always carried is unchanged — a percentage
    # question is never answered with the count inside it.
    assert _plan("Top advisors by CR%", org).metric == "cr_rate"


def test_top_advisors_by_an_unknown_metric_asks_instead_of_answering(org):
    """Rule 2. The user named a measure; we cannot answer it; we must not
    answer a different one."""
    intent = _intent("Top advisors by widget velocity", org)
    assert intent.unresolved
    assert intent.named_text == "widget velocity"

    plan = _plan("Top advisors by widget velocity", org)
    assert plan.action == "clarify_metric"
    assert plan.metric != DEFAULT_RANKING_METRIC
    assert plan.reason == "widget velocity"


# ---------------------------------------------------------------------
# The two audit examples
# ---------------------------------------------------------------------

def test_highest_cr_percent_is_no_longer_highest_cleared(org):
    """The audit case. It must not be revenue — and, since Phase 5.5,
    must not be the CR count either."""
    plan = _plan("Which BCM has the highest CR%", org)
    assert plan.metric != DEFAULT_RANKING_METRIC
    assert plan.metric != "client_registrations"
    # It clarified, naming the missing working-day calendar, until
    # working_days.py supplied one. Both exclusions above — not revenue,
    # not the count — are what the test is named for and both still hold.
    assert plan.metric == "cr_rate"


def test_best_answered_call_rate_is_no_longer_highest_cleared(org):
    """RETIRED ASSERTION. Step 4 pinned `metric == "answered_calls"` —
    fuzzy widening mapped the hyphenated rate phrasing onto the COUNT.
    That was better than the revenue default it replaced, and the Step 4
    report flagged it explicitly: "Neither is a true rate."

    The alias registry supersedes that compromise. "answered-call rate"
    is now a real metric: it was DECLARED uncomputable for want of a
    working-day calendar, and working_days.py is that calendar. The
    phrase reaches the RATE rather than the count it was widened onto."""
    plan = _plan("Which team has best answered-call rate", org)

    assert plan.metric == "answered_calls_rate"
    assert plan.metric != DEFAULT_RANKING_METRIC
    assert plan.metric != "answered_calls"


# ---------------------------------------------------------------------
# The default must still work everywhere it legitimately did
# ---------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Top 5 advisors",
    "top 10 advisors in Blue Area",
    "best advisors",
    "who is the worst advisor",
    "top 3 teams",
    "top 5 unit heads",
    "best advisors year to date",
    "who performed best this month",
    # NOT covered: "bottom 5 advisors". "bottom" is absent from
    # RANKING_STRONG, so it never reached the default and resolves to
    # `unresolved` — pre-existing, unrelated to F6, and adding it needs a
    # matching entry in ASCENDING_ABSOLUTE or "bottom 5" would rank
    # descending and show the TOP 5. Left alone rather than half-fixed.
])
def test_a_query_naming_no_measure_still_defaults(org, text):
    """The regression risk of this change is refusing queries that used
    to work. Every one of these names a ranking and no measure."""
    intent = _intent(text, org)
    assert intent.may_default, f"{text}: named_text={intent.named_text!r}"

    plan = _plan(text, org)
    assert plan.action == "leaderboard", text
    assert plan.metric == DEFAULT_RANKING_METRIC, text


@pytest.mark.parametrize("text,expected", [
    ("top advisors by revenue", "mtd_cleared"),
    ("top advisors by connects", "total_connects"),
    ("top teams by achievement", "achievement_pct"),
    ("which team has the most overdue", "overdue"),
    ("top advisors by pipeline", "pipeline_value"),
    ("top advisors by CR", "client_registrations"),
])
def test_a_named_and_resolvable_measure_is_used(org, text, expected):
    assert _intent(text, org).key == expected
    assert _plan(text, org).metric == expected


# ---------------------------------------------------------------------
# Widening runs before refusing
# ---------------------------------------------------------------------

@pytest.mark.parametrize("typo,expected", [
    ("top advisors by revnue", "mtd_cleared"),
    ("top advisors by achievment", "achievement_pct"),
    ("top advisors by pipline", "pipeline_value"),
])
def test_a_typo_is_widened_not_refused(org, typo, expected):
    """Refusing a typo would trade a wrong answer for a wrong refusal."""
    intent = _intent(typo, org)
    assert intent.key == expected, intent
    assert not intent.unresolved


def test_entities_are_not_mistaken_for_measures(org):
    """"in Blue Area" is a subject. If entity words leaked into the slot
    residue, every entity-scoped ranking would start refusing."""
    for text in ("top advisors in Blue Area", "top 5 advisors in Blue Area",
                 "best advisors under BCM Omar"):
        intent = _intent(text, org)
        assert not intent.unresolved, f"{text}: {intent.named_text!r}"


def test_level_words_are_not_mistaken_for_measures(org):
    """"advisors"/"teams"/"unit heads" fill the slot after a ranking word
    without naming a measure."""
    for text in ("top advisors", "top teams", "top unit heads", "top bcms",
                 "top zonal heads", "top companies", "top regions"):
        assert not _intent(text, org).unresolved, text


def test_period_words_are_not_mistaken_for_measures(org):
    for text in ("top advisors this month", "top advisors year to date",
                 "best teams this quarter", "top advisors today"):
        assert not _intent(text, org).unresolved, text


# ---------------------------------------------------------------------
# Scope: this replaces the RANKING reading, not every intent
# ---------------------------------------------------------------------

def test_a_roster_still_wins_over_the_metric_clarification(org):
    """"all advisors in X by widget velocity" is still a request to list
    people. The clarification is scored where the leaderboard would have
    been, so a genuinely different intent still outranks it."""
    plan = _plan("all advisors in Blue Area by widget velocity", org)
    assert plan.action == "roster"


def test_a_reverse_lookup_is_unaffected(org):
    plan = _plan("who is Adv One's unit head", org)
    assert plan.action == "reverse_hierarchy"


# ---------------------------------------------------------------------
# The clarification itself
# ---------------------------------------------------------------------

def test_the_clarification_quotes_the_user_and_offers_alternatives():
    message = metric_intent.clarification("widget velocity")
    assert "widget velocity" in message
    # It must name what IS available, or the next message is another guess.
    assert "revenue" in message.lower() or "cleared" in message.lower()


def test_the_pipeline_returns_a_clarification_not_an_answer(org):
    from app.llm.nlu_pipeline import resolve

    resolution = resolve("Top advisors by widget velocity", org, session_id=None)

    assert resolution.kind == "clarify"
    assert "widget velocity" in resolution.clarify_message
    assert resolution.clarify_options


def test_the_pipeline_still_answers_when_no_measure_was_named(org):
    from app.llm.nlu_pipeline import resolve

    resolution = resolve("Top 5 advisors", org, session_id=None)
    assert resolution.kind != "clarify"


# ---------------------------------------------------------------------
# Short synonyms: exact yes, fuzzy no
# ---------------------------------------------------------------------

def test_a_short_synonym_matches_exactly_but_never_fuzzily(org):
    """Adding "cr" made fuzzy matching resolve "mars" -> client
    registrations at 0.58, so "Advisors in Mars Region" was answered with
    a metric leaderboard. Edit distance is meaningless at three
    characters."""
    from app.llm.fallback_reasoning import fuzzy_resolve_metric

    assert fuzzy_resolve_metric("top advisors by cr") == "client_registrations"
    assert fuzzy_resolve_metric("advisors in Mars Region") is None
    assert fuzzy_resolve_metric("advisors in North Region") is None
    # Typos still widen — the floor sits between coincidence and typo.
    assert fuzzy_resolve_metric("revnue") == "mtd_cleared"


def test_cr_does_not_fire_inside_longer_words():
    from app.llm.metric_ontology import resolve_metric

    for text in ("across the board", "describe the increase", "the acre count"):
        assert resolve_metric(text) != "client_registrations", text
