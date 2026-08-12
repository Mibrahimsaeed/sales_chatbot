"""Phase 31 — a stated role points at a person; their hierarchy sets the scope.

    "BCM Haseeb Arslan connects"  ->  0
    "Unit Head Haseeb Arslan connects"  ->  12,004

Both sentences name the same man. He is a Unit Head over 75 advisors,
and he grounds at `bcm` only because the people directly beneath him
also name him in `management_lead` — a scope of one. Answering 0 is a
true statement about that scope and a false statement about him.

THE ROLE WORD IS A POINTER, NOT A CLAIM. It says which Haseeb Arslan,
not which of his jobs the question is about; there is only one of him.
So the stated role selects the person and the person's own hierarchy
decides the scope, which is the senior-most role they hold.

TWO PLACES READ THE LEVEL, and both had to agree or the fix would be
half-applied — the scope came out right while the answer was shaped as a
list of BCMs inside it. `_pin_stated_level` resolves the subject;
`query_planner` re-reads the level word from the raw text. The pin now
records a level it OVERRULED, and only then, so every other query still
reaches detect_level untouched.

THE RANKING IS NOT NEW. `_highest_role` and `_ROLE_LEVELS` are Phase
28's, derived from `hierarchy.CHAIN` — so stating the role and omitting
it cannot reach different scopes for the same person, and there is no
second hierarchy definition to keep in sync.

The fixture's three people are the spec's Person A/B/C, and Adeel Raza
is the sharp one: his BCM scope (34) is LARGER than his Zonal Head scope
(28). Asserting 28 fails both if the fix does nothing and if it picks by
size rather than by rank.
"""

import pytest

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import (
    advisor_resolver, conversation_memory, entity_extractor, hierarchy,
    narrative, nlu_pipeline, semantic_parser,
)
from app.llm.nlu_pipeline import _authoritative_role
from app.services.chat_service import handle_chat_message

# Person A — Tahir Malik   : Advisor + BCM + Zonal Head + Unit Head
# Person B — Adeel Raza    : Advisor + BCM + Zonal Head
# Person C — Hina Sethi    : Advisor + BCM
# Rabia Noor               : Advisor only — the single-role control
#
# The CR and meetings figures are scaled so that no junior-scope total
# can coincide with a digit of a rendered percentage — an absence
# assertion over a small integer proves nothing.
#
# wid, name,        rm (unit head), portfolio_lead (zonal), management_lead (bcm), connects, cr, meetings
PEOPLE = [
    (1,  "Tahir Malik", "Tahir Malik", "Tahir Malik", "Tahir Malik",  7,  100, 200),
    (2,  "Sana Riaz",   "Tahir Malik", "Tahir Malik", "Tahir Malik", 11,  200, 300),
    (3,  "Bilal Khan",  "Tahir Malik", "Tahir Malik", "Other Bcm",   13,  400, 500),
    (4,  "Nida Aslam",  "Tahir Malik", "Other Zh",    "Other Bcm",   17,  800, 700),
    (5,  "Adeel Raza",  "Other Uh",    "Adeel Raza",  "Adeel Raza",   5,  100, 100),
    (6,  "Kiran Shah",  "Other Uh",    "Adeel Raza",  "Other Bcm",   23, 1600, 1100),
    (7,  "Omar Faruq",  "Other Uh",    "Other Zh",    "Adeel Raza",  29, 3200, 1300),
    (8,  "Hina Sethi",  "Other Uh",    "Other Zh",    "Hina Sethi",   3,  100, 100),
    (9,  "Zaid Anwar",  "Other Uh",    "Other Zh",    "Hina Sethi",  19, 6400, 1700),
    (10, "Rabia Noor",  "Other Uh",    "Other Zh",    "Other Bcm",   31, 12800, 1900),
]

TAHIR_UNIT = 7 + 11 + 13 + 17      # 48  <- authoritative
TAHIR_ZONE = 7 + 11 + 13           # 31
TAHIR_CENTRE = 7 + 11              # 18
TAHIR_OWN = 7

ADEEL_ZONE = 5 + 23                # 28  <- authoritative (the SMALLER one)
ADEEL_CENTRE = 5 + 29              # 34
ADEEL_OWN = 5

HINA_CENTRE = 3 + 19               # 22  <- authoritative, nothing above it
HINA_OWN = 3

TAHIR_UNIT_CR = 100 + 200 + 400 + 800     # 1,500
TAHIR_CENTRE_CR = 100 + 200               # 300
TAHIR_UNIT_MEETINGS = 200 + 300 + 500 + 700   # 1,700
TAHIR_CENTRE_MEETINGS = 200 + 300             # 500


@pytest.fixture()
def db(db_session, monkeypatch):
    for wid, name, rm, pl, ml, connects, cr, meetings in PEOPLE:
        db_session.add(Advisor(wid=wid, name=name, team="Alpha", company="Graana",
                               rm=rm, portfolio_lead=pl, management_lead=ml,
                               in_master_sheet=True))
        db_session.add(Calls(wid=wid, connects_mtd=connects, connects_daily=0,
                             answered_calls_mtd=connects, answered_calls_daily=0))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=connects,
                                   mtd_followup_connect=0, mtd_cr=cr,
                                   mtd_new_meeting=meetings, mtd_followup_meeting=0))
        db_session.add(Pipeline(wid=wid, pipeline=connects, overdue=0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=connects))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


def _ask(db, text):
    conversation_memory._store.clear()
    return handle_chat_message(db, text, session_id=None)


def _numbers(text):
    import re
    return {int(t.replace(",", "")) for t in re.findall(r"\d[\d,]*", text)}


def _headline(response):
    """The answer sentence alone.

    Phase 27 appends the member roster and Phase 29 the metric bundle, so
    the full reply legitimately contains each subordinate's OWN figure —
    and a junior scope total is by definition a sum of some of those. An
    absence assertion over the whole prose therefore proves nothing;
    over the headline it proves exactly what it claims.
    """
    return str(response["reply"]).split("\n")[0]


def _scope(db, text):
    """(subject_level, filters) as the pipeline resolved them — the
    guarantee itself, rather than its rendering."""
    resolution = nlu_pipeline.resolve(text, db)
    assert resolution.ir is not None, text
    return resolution.ir.subject_level, [(f.field, f.value) for f in resolution.ir.filters]


ALL_ROLES = ["unit_head", "zonal_head", "bcm", "advisor"]


# ---------------------------------------------------------------------
# The rule, in isolation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("stated,levels,expected", [
    # Person A — Advisor + BCM + Zonal Head + Unit Head
    ("bcm",        ALL_ROLES, "unit_head"),
    ("zonal_head", ALL_ROLES, "unit_head"),
    ("unit_head",  ALL_ROLES, "unit_head"),
    # Person B — Advisor + BCM + Zonal Head
    ("bcm",        ["zonal_head", "bcm", "advisor"], "zonal_head"),
    ("zonal_head", ["zonal_head", "bcm", "advisor"], "zonal_head"),
    # Person C — Advisor + BCM: nothing above it, so BCM stands
    ("bcm",        ["bcm", "advisor"], "bcm"),
])
def test_the_authoritative_role_is_the_senior_one_they_hold(stated, levels, expected):
    assert _authoritative_role(stated, levels) == expected


def test_a_stated_role_is_never_demoted():
    """Promotion moves UP the chain only. Naming a role someone does not
    hold cannot push them below the one they do."""
    assert _authoritative_role("unit_head", ["zonal_head", "bcm", "advisor"]) == "unit_head"


def test_advisor_is_never_promoted():
    """"Advisor X" asks for the person as a leaf — their own figure.
    Promoting it would make a manager's own record unreachable, which is
    the defect Phase 22 exists to prevent."""
    assert _authoritative_role("advisor", ALL_ROLES) == "advisor"


def test_the_ranking_is_phase_28s_not_a_second_one():
    """One hierarchy definition. If CHAIN is rebound, both the stated-role
    promotion and the no-role default follow it together."""
    assert list(nlu_pipeline._ROLE_LEVELS) == [
        lvl for lvl in hierarchy.CHAIN if lvl != "team"]
    assert nlu_pipeline._highest_role(ALL_ROLES) == _authoritative_role("bcm", ALL_ROLES)


# ---------------------------------------------------------------------
# Person A — every role word reaches the Unit Head scope
# ---------------------------------------------------------------------


@pytest.mark.parametrize("role", ["BCM", "Zonal Head", "Unit Head"])
def test_person_a_answers_at_their_unit_head_scope_whatever_role_is_named(db, role):
    response = _ask(db, f"{role} Tahir Malik connects")
    assert TAHIR_UNIT in _numbers(_headline(response))


@pytest.mark.parametrize("role", ["BCM", "Zonal Head"])
def test_person_a_does_not_answer_at_the_junior_scope(db, role):
    """The failure this phase fixes: a scope of two reported as the man's
    figure."""
    got = _numbers(_headline(_ask(db, f"{role} Tahir Malik connects")))
    assert TAHIR_CENTRE not in got


def test_naming_the_role_and_omitting_it_reach_the_same_scope(db):
    """One person, one authoritative scope — reachable by either
    phrasing, because both read the same ranking."""
    stated = _numbers(_headline(_ask(db, "BCM Tahir Malik connects")))
    unstated = _numbers(_headline(_ask(db, "connects of Tahir Malik's team")))
    assert TAHIR_UNIT in stated and TAHIR_UNIT in unstated


def test_the_answer_is_one_value_not_a_list_of_juniors(db):
    """Scope and shape must be settled together. Fixing only the filter
    left the answer scoped to the unit but rendered as a ranking of the
    BCMs inside it — the same question resolved two ways."""
    response = _ask(db, "BCM Tahir Malik connects")
    assert response["type"] != "leaderboard"
    assert response["reply"].startswith("Tahir Malik")


# ---------------------------------------------------------------------
# Person B — rank decides, not size
# ---------------------------------------------------------------------


@pytest.mark.parametrize("role", ["BCM", "Zonal Head"])
def test_person_b_answers_at_their_zonal_head_scope(db, role):
    got = _numbers(_headline(_ask(db, f"{role} Adeel Raza connects")))
    assert ADEEL_ZONE in got


def test_the_senior_role_wins_even_though_the_junior_one_is_bigger(db):
    """Adeel Raza leads 28 as Zonal Head and 34 as BCM. 34 is what the
    old behaviour returned for "BCM Adeel Raza"."""
    got = _numbers(_headline(_ask(db, "BCM Adeel Raza connects")))
    assert ADEEL_ZONE in got
    assert ADEEL_CENTRE not in got


def test_omar_faruq_is_outside_person_bs_authoritative_scope(db):
    """He reports to Adeel as BCM but not as Zonal Head — the whole
    34-vs-28 difference, stated as people rather than numbers."""
    assert "Omar Faruq" not in _ask(db, "BCM Adeel Raza connects")["reply"]


# ---------------------------------------------------------------------
# Person C — a role with nothing above it is left alone
# ---------------------------------------------------------------------


def test_person_c_answers_at_their_bcm_scope(db):
    got = _numbers(_headline(_ask(db, "BCM Hina Sethi connects")))
    assert HINA_CENTRE in got


def test_person_c_is_not_promoted_to_a_role_they_do_not_hold(db):
    """Hina Sethi is a BCM and nothing more. Promotion must not invent a
    Zonal Head or Unit Head scope for her."""
    response = _ask(db, "BCM Hina Sethi connects")
    assert response["type"] != "clarification"
    assert HINA_CENTRE in _numbers(_headline(response))


def test_a_single_role_person_behaves_exactly_as_before(db):
    """Rabia Noor grounds at `advisor` alone — no ambiguity, so nothing
    on this path is even consulted for her."""
    assert 31 in _numbers(_headline(_ask(db, "connects of Rabia Noor")))


# ---------------------------------------------------------------------
# Every metric family, not just connects
# ---------------------------------------------------------------------


@pytest.mark.parametrize("phrase,authoritative,junior", [
    ("connects",      TAHIR_UNIT,          TAHIR_CENTRE),
    ("CR",            TAHIR_UNIT_CR,       TAHIR_CENTRE_CR),
    ("answered calls", TAHIR_UNIT,         TAHIR_CENTRE),
    ("meetings",      TAHIR_UNIT_MEETINGS, TAHIR_CENTRE_MEETINGS),
    ("pipeline",      TAHIR_UNIT,          TAHIR_CENTRE),
])
def test_every_metric_family_uses_the_authoritative_role(db, phrase, authoritative, junior):
    """The rule belongs to subject resolution, so it must hold for every
    measure rather than for the one it was found with."""
    response = _ask(db, f"BCM Tahir Malik {phrase}")
    assert authoritative in _numbers(_headline(response)), phrase
    assert junior not in _numbers(_headline(response)), phrase
    assert _scope(db, f"BCM Tahir Malik {phrase}") == (
        "unit_head", [("unit_head", "Tahir Malik")]), phrase


@pytest.mark.parametrize("phrase", ["CR %", "answered calls rate", "meeting rate"])
def test_rate_metrics_resolve_at_the_authoritative_role(db, phrase):
    """A rate has no fixture-computable constant, so this pins the SCOPE
    the two phrasings agree on rather than a number."""
    stated = _ask(db, f"BCM Tahir Malik {phrase}")
    unstated = _ask(db, f"Unit Head Tahir Malik {phrase}")
    assert stated["reply"] == unstated["reply"]


# ---------------------------------------------------------------------
# Regressions: the three readings stay distinct
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name,own,group", [
    ("Tahir Malik", TAHIR_OWN, TAHIR_UNIT),
    ("Adeel Raza", ADEEL_OWN, ADEEL_ZONE),
    ("Hina Sethi", HINA_OWN, HINA_CENTRE),
])
def test_a_bare_person_query_still_returns_their_own_figure(db, name, own, group):
    """Phase 22's RULE 1, untouched: the role promotion applies to a
    STATED role, and a bare question states none."""
    got = _numbers(_headline(_ask(db, f"connects of {name}")))
    assert own in got
    assert group not in got


@pytest.mark.parametrize("name,group", [
    ("Tahir Malik", TAHIR_UNIT), ("Adeel Raza", ADEEL_ZONE), ("Hina Sethi", HINA_CENTRE),
])
def test_a_persons_team_query_still_resolves_the_highest_role(db, name, group):
    """Phase 28, unchanged — and now reaching the same scope the stated
    role does."""
    response = _ask(db, f"connects of {name}'s team")
    assert response["type"] != "clarification"
    assert group in _numbers(_headline(response))


def test_the_member_breakdown_still_follows(db):
    """Phase 27: the roster behind the total, and it still sums to it."""
    response = _ask(db, "connects of Tahir Malik's team")
    assert sum(m["value"] or 0 for m in response["members"]) == TAHIR_UNIT


def test_a_unit_head_roster_query_is_unaffected(db):
    """Phase 30: the roster path reads `plan.level`, not the level word,
    so the promotion never touches it."""
    response = _ask(db, "show all advisors under Unit Head Tahir Malik")
    assert response["type"] == "roster"
    assert {a["wid"] for a in response["data"]["advisors"]} == {1, 2, 3, 4}


def test_a_metric_over_a_unit_heads_advisors_keeps_advisor_level(db):
    """The roster noun still names the OUTPUT (Phase 30) — a promotion
    must not turn this back into a single unit-head figure."""
    resolution = nlu_pipeline.resolve("connects of advisors under Unit Head Tahir Malik", db)
    assert resolution.ir is not None
    assert resolution.ir.subject_level == "advisor"
    assert ("unit_head", "Tahir Malik") in [(f.field, f.value) for f in resolution.ir.filters]


def test_a_query_with_no_promotion_leaves_the_level_word_alone(db):
    """The breadcrumb is written only when the text was OVERRULED, so an
    ordinary query carries nothing and the planner reads it as always."""
    from app.llm.entity_extractor import extract_entities

    entities = extract_entities("connects of unit head tahir malik", db)
    assert nlu_pipeline.PINNED_LEVEL_KEY not in nlu_pipeline._pin_stated_level(
        "connects of unit head tahir malik", entities)

    promoted = nlu_pipeline._pin_stated_level(
        "connects of bcm tahir malik", extract_entities("connects of bcm tahir malik", db))
    assert promoted[nlu_pipeline.PINNED_LEVEL_KEY] == "unit_head"
