"""Phase 30 — "advisors" says what to RETURN, not who the subject is.

    "show all advisors under Unit Head Kaleem Satti"

answered with Kaleem Satti's own advisor profile. Not a wrong roster —
NO ROSTER AT ALL, for a question naming 137 people.

THE ROSTER LAYER WAS NEVER REACHED. `hierarchy_service.get_level_roster`
returns exactly the right people at every Unit Head, and
`hierarchy.scope_filter` reads the authoritative `advisors.rm` column,
which is 100% populated. Both were correct throughout and neither was
called.

The break was one word. That sentence names TWO levels — `advisors`
(what to return) and `Unit Head` (who Kaleem is) — and
`intent_catalog.detect_level` returns the first entry in LEVEL_KEYWORDS
order, which is `advisor`. So the output noun outranked the qualifier
purely by table position; `_pin_stated_level` believed the user had said
"the ADVISOR named Kaleem Satti", deleted his unit_head grounding, and
with no group entity left the roster candidate scored nothing and
`lookup` won.

Swapping one noun proved it: "show all STAFF under Unit Head Kaleem
Satti" — identical sentence otherwise — returned all 137, because
"staff" is a roster word that is not also a level keyword.

That is why the noun tests below are the heart of this file: four words
for the same people must reach the same roster, and none of them may
decide the subject's level.
"""

import pytest

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import (
    advisor_resolver, conversation_memory, entity_extractor, intent_catalog,
    narrative, nlu_pipeline, semantic_parser,
)
from app.services import hierarchy_service
from app.services.chat_service import handle_chat_message

# Adnan Sheikh is a SPANNING Zonal Head — the production shape found in
# Phase 26. His advisors sit under two different Unit Heads, so a roster
# built by walking Unit Head -> Zonal Head -> Advisor would list people
# twice; one built from each advisor's own `rm` cannot.
#
# wid, name,            rm (unit head),  portfolio_lead,  management_lead, connects
PEOPLE = [
    (1, "Nabeel Qadir",  "Nabeel Qadir",  "Nabeel Qadir",  "Nabeel Qadir",  11),
    (2, "Sana Yousaf",   "Nabeel Qadir",  "Adnan Sheikh",  "Adnan Sheikh",  22),
    (3, "Kamran Riaz",   "Nabeel Qadir",  "Adnan Sheikh",  "Faryal Iqbal",  33),
    (4, "Rukhsana Bibi", "Rukhsana Bibi", "Rukhsana Bibi", "Rukhsana Bibi", 44),
    (5, "Adnan Sheikh",  "Rukhsana Bibi", "Adnan Sheikh",  "Adnan Sheikh",  55),
    (6, "Mehwish Anwar", "Rukhsana Bibi", "Adnan Sheikh",  "Adnan Sheikh",  66),
    (7, "Tahir Zaman",   "Rukhsana Bibi", "Other Zh",      "Other Bcm",     77),
]

NABEEL = {1, 2, 3}
RUKHSANA = {4, 5, 6, 7}
NABEEL_CONNECTS = 11 + 22 + 33         # 66
NABEEL_OWN_CONNECTS = 11

ROSTER_NOUNS = ["advisors", "staff", "people", "agents"]


@pytest.fixture()
def db(db_session, monkeypatch):
    for wid, name, rm, pl, ml, connects in PEOPLE:
        db_session.add(Advisor(wid=wid, name=name, team="Alpha", company="Graana",
                               rm=rm, portfolio_lead=pl, management_lead=ml,
                               in_master_sheet=True))
        db_session.add(Calls(wid=wid, connects_mtd=connects, connects_daily=0,
                             answered_calls_mtd=connects, answered_calls_daily=0))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=connects,
                                   mtd_followup_connect=0, mtd_cr=1))
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


def _wids(response):
    data = response.get("data")
    if isinstance(data, dict) and "advisors" in data:
        return {a["wid"] for a in data["advisors"]}
    if isinstance(data, list):
        return {r.get("wid") for r in data if isinstance(r, dict)}
    return set()


def _service_wids(db, name):
    return {a["wid"] for a in
            hierarchy_service.get_level_roster(db, "unit_head", name)["advisors"]}


def _rm_wids(db, name):
    return {a.wid for a in db.query(Advisor).filter(
        Advisor.rm == name, Advisor.in_master_sheet.is_(True))}


# ---------------------------------------------------------------------
# 1-2. The roster is reached, and it is the service's roster
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Nabeel Qadir", "Rukhsana Bibi"])
def test_a_unit_head_roster_query_returns_a_roster_not_a_profile(db, name):
    response = _ask(db, f"show all advisors under Unit Head {name}")
    assert response["type"] == "roster", response["reply"][:120]


@pytest.mark.parametrize("name,expected", [
    ("Nabeel Qadir", NABEEL), ("Rukhsana Bibi", RUKHSANA),
])
def test_the_roster_wids_equal_the_hierarchy_services(db, name, expected):
    """The chatbot must not have its own idea of who is in a group. Same
    WIDs, not merely the same count — a count can match while the people
    differ."""
    response = _ask(db, f"show all advisors under Unit Head {name}")
    assert _wids(response) == _service_wids(db, name) == expected


# ---------------------------------------------------------------------
# 3. Every word for the same people reaches the same roster
# ---------------------------------------------------------------------


@pytest.mark.parametrize("noun", ROSTER_NOUNS)
def test_every_roster_noun_reaches_the_same_roster(db, noun):
    """THE regression. Only `advisors`/`agents` are also level keywords,
    so before the fix these four sentences — identical but for one word —
    gave two different answers."""
    response = _ask(db, f"show all {noun} under Unit Head Nabeel Qadir")
    assert response["type"] == "roster"
    assert _wids(response) == NABEEL


def test_the_noun_cannot_change_the_scope(db):
    """Stated as one assertion so a future change cannot fix one noun and
    quietly leave another behind."""
    results = {noun: _wids(_ask(db, f"show all {noun} under Unit Head Nabeel Qadir"))
               for noun in ROSTER_NOUNS}
    assert len(set(map(frozenset, results.values()))) == 1, results


# ---------------------------------------------------------------------
# 4-5. The roster is the authoritative column, and it does not double-count
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Nabeel Qadir", "Rukhsana Bibi"])
def test_the_roster_is_the_authoritative_rm_column(db, name):
    """`advisors.rm` IS the Unit Head relationship (Phase 25 pointed it at
    MasterSheet.Regional). The answer must be that column and nothing
    else — no traversal, no second definition."""
    assert _wids(_ask(db, f"show all advisors under Unit Head {name}")) == _rm_wids(db, name)


def test_a_spanning_zonal_head_does_not_put_anyone_under_two_unit_heads(db):
    """Adnan Sheikh leads advisors beneath BOTH Unit Heads. Walking
    Unit Head -> Zonal Head -> Advisor would list wids 2, 3, 5 and 6
    twice; each advisor's own `rm` names exactly one Unit Head, so the
    rosters stay disjoint."""
    nabeel = _wids(_ask(db, "show all advisors under Unit Head Nabeel Qadir"))
    rukhsana = _wids(_ask(db, "show all advisors under Unit Head Rukhsana Bibi"))

    assert nabeel & rukhsana == set()
    assert nabeel | rukhsana == {w for w, *_ in PEOPLE}
    assert len(nabeel) + len(rukhsana) == len(PEOPLE)


def test_the_spanning_zonal_heads_own_advisors_split_by_their_own_rm(db):
    """The people under Adnan Sheikh, read individually: 2 and 3 report to
    Nabeel, 5 and 6 to Rukhsana. That split is the data, and the roster
    must reproduce it rather than smoothing it over."""
    nabeel = _wids(_ask(db, "show all advisors under Unit Head Nabeel Qadir"))
    assert {2, 3} <= nabeel
    assert {5, 6} & nabeel == set()


# ---------------------------------------------------------------------
# 6-8. Metric queries are untouched
# ---------------------------------------------------------------------


def test_a_stated_level_metric_query_still_answers_that_level(db):
    """Phase 22's behaviour, unchanged by this fix: naming the level in
    the sentence settles WHICH Nabeel Qadir is meant, and the Unit Head
    reading answers for the Unit Head's scope. The guard added here
    cannot fire — there is no roster phrasing in this sentence."""
    response = _ask(db, "Unit Head Nabeel Qadir connects")
    assert response["type"] != "roster"
    assert f"{NABEEL_CONNECTS:,}" in response["reply"]


def test_a_metric_over_a_unit_heads_advisors_keeps_the_unit_head_scope(db):
    """"connects of advisors under Unit Head X" contains the same two
    level words as the roster query, so it goes through the same guard —
    and must still come out scoped to the Unit Head."""
    resolution = nlu_pipeline.resolve("connects of advisors under Unit Head Nabeel Qadir", db)
    assert resolution.ir is not None
    assert ("unit_head", "Nabeel Qadir") in [(f.field, f.value) for f in resolution.ir.filters]


def test_a_bare_person_query_still_returns_their_own_connects(db):
    """Phase 22's RULE 1. Nabeel Qadir grounds at four levels and a
    question about HIM still answers with his own 11, not his unit's 66."""
    reply = _ask(db, "connects of Nabeel Qadir")["reply"]
    assert str(NABEEL_OWN_CONNECTS) in reply
    assert f"{NABEEL_CONNECTS:,}" not in reply


def test_a_persons_team_query_still_resolves_the_highest_role(db):
    """Phase 28, unaffected: no clarification, and the Unit Head scope."""
    response = _ask(db, "connects of Nabeel Qadir's team")
    assert response["type"] != "clarification"
    assert f"{NABEEL_CONNECTS:,}" in response["reply"]


def test_a_named_team_roster_still_works(db):
    """The path that always worked — the noun is the same, but the value
    grounds at one level only, so the guard never engages."""
    response = _ask(db, "all advisors in Alpha")
    assert response["type"] == "roster"
    assert _wids(response) == {w for w, *_ in PEOPLE}


# ---------------------------------------------------------------------
# 9. The ordering dependency itself
# ---------------------------------------------------------------------


def test_detect_level_still_prefers_advisor_by_table_position():
    """The upstream behaviour this fix works around, pinned so the reason
    the guard exists stays visible. `advisor` comes first in
    LEVEL_KEYWORDS, so it wins even when a more specific level word is
    present in the same sentence."""
    assert list(intent_catalog.LEVEL_KEYWORDS).index("advisor") < \
           list(intent_catalog.LEVEL_KEYWORDS).index("unit_head")
    assert intent_catalog.detect_level("show all advisors under unit head kaleem satti") == "advisor"


@pytest.mark.parametrize("text,expected", [
    ("show all advisors under unit head kaleem satti", "unit_head"),
    ("show all agents under unit head kaleem satti", "unit_head"),
    ("all advisors under zonal head kaleem satti", "zonal_head"),
    ("all advisors under bcm kaleem satti", "bcm"),
])
def test_the_subject_level_is_the_qualifier_not_the_output_noun(text, expected):
    """The fix itself, in one call: when a roster names both a noun for
    the people and a level for the subject, the SUBJECT's level wins."""
    levels = ["unit_head", "zonal_head", "bcm", "advisor"]
    assert nlu_pipeline._subject_level_word(text, levels) == expected


def test_a_sentence_with_no_roster_noun_is_read_exactly_as_before():
    """The guard is gated on roster phrasing, so every other query keeps
    detect_level's answer verbatim."""
    for text in ("connects of zonal head faisal naqvi", "unit head kaleem satti connects"):
        assert nlu_pipeline._subject_level_word(text, ["unit_head", "zonal_head", "bcm", "advisor"]) \
            == intent_catalog.detect_level(text)


def test_an_ambiguous_qualifier_falls_through_to_the_question():
    """Two level words for the subject settle nothing, so the guard
    declines rather than picking one — the clarification is still the
    honest answer there."""
    assert nlu_pipeline._subject_level_word(
        "all advisors under unit head and zonal head kaleem satti",
        ["unit_head", "zonal_head", "bcm", "advisor"]) is None


def test_a_qualifier_the_value_does_not_hold_is_not_used():
    """"advisors under Unit Head X" where X is not a Unit Head must not
    be forced into a Unit Head reading."""
    assert nlu_pipeline._subject_level_word(
        "show all advisors under unit head kaleem satti", ["bcm", "advisor"]) is None
