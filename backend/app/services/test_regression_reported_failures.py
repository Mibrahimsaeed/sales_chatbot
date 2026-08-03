"""
REGRESSION SUITE — one test per reported failure.

Every test here corresponds to a symptom reported from production and
reproduced during the pipeline audit. They run through the real
handle_chat_message() path so a regression is caught at the level the
user actually experiences, not just in a unit.

The fixture deliberately mirrors the PRODUCTION SHAPES that caused each
failure, because the shapes are the bug:

  - "Adeel Dogar" is a unit head who is NOT an advisor, while "Adeel
    Mubarik Dogar" IS a different advisor. Whole-sentence fuzzy matching
    scored the lookalike at 0.62 and returned his personal revenue.
  - "Yasir Ali" is several real people (8 in production). The old lookup
    did `ORDER BY wid LIMIT 1` and the other 7 were unreachable.
  - "Asif Ali" is a different person whose name scores 0.82 against
    "Yasir Ali" — over the old 0.80 floor.
  - "Ali Murtaza" is BOTH an advisor and a unit head with reports, so
    "his team" and "his BM" are different questions about the same name.

If any assertion here fails, a specific user-visible wrong answer has
come back.
"""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import advisor_resolver, conversation_memory, entity_extractor, narrative, semantic_parser
from app.services.chat_service import handle_chat_message


@pytest.fixture()
def prod_shapes_db(db_session, monkeypatch):
    def advisor(wid, name, team, company="Graana", **kw):
        db_session.add(Advisor(wid=wid, name=name, team=team, company=company, **kw))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=10, mtd_followup_connect=0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD, target=1000, cleared=500))

    # --- "Adeel Dogar": a unit head with reports, NOT himself an advisor
    advisor(1, "Reportee One", "Blue Area", bm="Adeel Dogar", rm="Adeel Dogar")
    advisor(2, "Reportee Two", "Downtown", bm="Adeel Dogar", rm="Adeel Dogar")
    # --- the lookalike that used to be returned instead
    advisor(3, "Adeel Mubarik Dogar", "Gamma")

    # --- "Yasir Ali": several real people sharing one name
    advisor(10, "Yasir Ali", "North/KPK", "Agency21", bm="Aimal Khan", zm="Zarak Khan", portfolio_lead="Zarak Khan", rm="Atif Irfan")
    advisor(11, "Yasir Ali", "Downtown", "IMARAT", bm="Fraz Khalid", zm="Salman Arshad", portfolio_lead="Salman Arshad", rm="Rashid Majeed")
    advisor(12, "Yasir Ali", "Blue Area", "Graana")

    # --- "Asif Ali": a DIFFERENT person, name 0.82 similar to "Yasir Ali"
    advisor(20, "Asif Ali", "Alpha", "Graana", bm="Aimal Khan", rm="Aimal Khan")

    # --- "Ali Murtaza": both an advisor AND a unit head over others
    advisor(30, "Ali Murtaza", "North/KPK", "Graana", bm="Musab Sial", rm="Musab Sial", zm="Arsalan Jaraal", portfolio_lead="Arsalan Jaraal")
    advisor(31, "Murtaza Reportee A", "North/KPK", bm="Ali Murtaza", rm="Ali Murtaza")
    advisor(32, "Murtaza Reportee B", "North/KPK", bm="Ali Murtaza", rm="Ali Murtaza")

    # --- an unambiguous advisor, for the plain-profile control case
    advisor(40, "Kainat Khalid", "Blue Area", bm="Aimal Khan", zm="Zarak Khan", portfolio_lead="Zarak Khan", rm="Atif Irfan")
    db_session.commit()

    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()


def _ask(db, text, session_id=None):
    return handle_chat_message(db, text, session_id=session_id)


# =====================================================================
# 1. "Show Adeel Dogar's team" -> returns the Unit Head's team
#    WAS: "Adeel Mubarik Dogar has 99 MTD connects…" (a different human)
# =====================================================================

def test_show_adeel_dogars_team_returns_the_unit_head_team(prod_shapes_db):
    response = _ask(prod_shapes_db, "Show Adeel Dogar's team")

    assert response["type"] == "breakdown"
    assert response["data"]["level"] == "unit_head"
    assert response["data"]["value"] == "Adeel Dogar"
    assert response["data"]["advisors"] == 2
    # the lookalike must not appear anywhere in the answer
    assert "Adeel Mubarik Dogar" not in response["reply"]


# =====================================================================
# 2. "Who reports to Adeel Dogar?" -> returns his direct reports
#    WAS: correct only by luck — 0.57 vs the 0.6 gate, while the
#    possessive phrasing scored 0.62 and returned the wrong person
# =====================================================================

def test_who_reports_to_adeel_dogar_returns_direct_reports(prod_shapes_db):
    response = _ask(prod_shapes_db, "Who reports to Adeel Dogar?")

    assert response["type"] == "breakdown"
    assert response["data"]["value"] == "Adeel Dogar"
    names = {a["name"] for t in response["data"]["teams"] for a in t["advisors"]}
    assert names == {"Reportee One", "Reportee Two"}


def test_both_phrasings_of_the_same_question_agree(prod_shapes_db):
    """The reported INCONSISTENCY: a 0.05 scoring difference silently
    flipped between the right answer and a different person's profile."""
    a = _ask(prod_shapes_db, "Who reports to Adeel Dogar?")
    b = _ask(prod_shapes_db, "Show Adeel Dogar's team")
    c = _ask(prod_shapes_db, "Who is in Adeel Dogar's team?")

    assert a["type"] == b["type"] == c["type"] == "breakdown"
    assert a["data"]["value"] == b["data"]["value"] == c["data"]["value"] == "Adeel Dogar"


# =====================================================================
# 3. "Tell me about Yasir Ali" -> asks which one
#    WAS: silently returned whichever had the lowest wid
# =====================================================================

def test_tell_me_about_yasir_ali_requests_clarification(prod_shapes_db):
    response = _ask(prod_shapes_db, "Tell me about Yasir Ali")

    assert response["type"] == "clarification"
    assert len(response["options"]) == 3
    # every candidate must be distinguishable, or the question is
    # unanswerable — the name is exactly what failed to tell them apart
    assert "North/KPK" in response["reply"]
    assert "Downtown" in response["reply"]
    assert "Blue Area" in response["reply"]


def test_unambiguous_name_still_answers_directly(prod_shapes_db):
    """Control: disambiguation must not regress the common case."""
    response = _ask(prod_shapes_db, "Tell me about Kainat Khalid")
    assert response["type"] == "advisor"
    assert response["data"]["wid"] == 40


# =====================================================================
# 4. "Who is BM of Ali Murtaza?" -> returns his BM
# =====================================================================

@pytest.mark.parametrize("phrasing,level", [
    # PHASE 3: "BM" and "unit head" are no longer the same level. The
    # verified chain binds Unit Head to Advisor.rm; Advisor.bm keeps its
    # own reverse-only level so "who is X's BM" still answers from the
    # column it actually reads.
    ("Who is BM of Ali Murtaza?", "bm"),
    ("Who is Ali Murtaza's BM?", "bm"),
    ("Who is Ali Murtaza's unit head?", "unit_head"),
    ("Who does Ali Murtaza report to?", "unit_head"),
])
def test_who_is_bm_of_ali_murtaza(prod_shapes_db, phrasing, level):
    """A name that is BOTH an advisor and a unit head must not derail
    this: asking for someone's manager is asking about them as a person."""
    response = _ask(prod_shapes_db, phrasing)

    assert response["type"] == "manager"
    assert response["data"]["level"] == level
    assert response["data"]["advisor"] == "Ali Murtaza"


def test_zm_and_rm_reverse_lookups(prod_shapes_db):
    zm = _ask(prod_shapes_db, "Who is Kainat Khalid's ZM?")
    assert zm["data"]["level"] == "zm" and zm["data"]["manager"] == "Zarak Khan"

    rm = _ask(prod_shapes_db, "Who is Kainat Khalid's RM?")
    assert rm["data"]["level"] == "unit_head" and rm["data"]["manager"] == "Atif Irfan"


# =====================================================================
# 5. "Show Ali Murtaza's team" -> the team, NOT his advisor profile
# =====================================================================

def test_show_ali_murtazas_team_returns_team_not_profile(prod_shapes_db):
    response = _ask(prod_shapes_db, "Show Ali Murtaza's team")

    assert response["type"] == "breakdown"
    assert response["data"]["level"] == "unit_head"
    assert response["data"]["value"] == "Ali Murtaza"
    names = {a["name"] for t in response["data"]["teams"] for a in t["advisors"]}
    assert names == {"Murtaza Reportee A", "Murtaza Reportee B"}


def test_the_same_name_answers_two_different_questions_correctly(prod_shapes_db):
    """"Ali Murtaza's team" (his reports) and "Ali Murtaza's BM" (his
    manager) are different questions about one name — previously both
    collapsed to the identical clarification prompt."""
    team = _ask(prod_shapes_db, "Show Ali Murtaza's team")
    boss = _ask(prod_shapes_db, "Who is Ali Murtaza's BM?")

    assert team["type"] == "breakdown"
    assert boss["type"] == "manager"
    assert boss["data"]["manager"] == "Musab Sial"


# =====================================================================
# 6. Similar names ("Yasir Ali" vs "Asif Ali") never resolve incorrectly
#    WAS: "tell me about Yasir Ali" asked about ASIF Ali — a person the
#    user never mentioned — because 0.82 cleared the 0.80 floor
# =====================================================================

def test_asking_about_yasir_ali_never_surfaces_asif_ali(prod_shapes_db):
    response = _ask(prod_shapes_db, "Tell me about Yasir Ali")
    assert "Asif Ali" not in response["reply"]
    for option in response["options"]:
        assert "Asif Ali" not in option


def test_asking_about_asif_ali_never_surfaces_yasir_ali(prod_shapes_db):
    response = _ask(prod_shapes_db, "Tell me about Asif Ali")
    assert response["type"] == "advisor"
    assert response["data"]["wid"] == 20
    assert "Yasir Ali" not in response["reply"]


def test_similar_names_resolve_to_their_own_wid(prod_shapes_db):
    """Direct resolver check — the two names must never cross-resolve."""
    asif = advisor_resolver.resolve_advisor("Asif Ali", prod_shapes_db)
    assert asif.is_resolved and asif.wid == 20

    yasir = advisor_resolver.resolve_advisor("Yasir Ali", prod_shapes_db)
    assert yasir.is_ambiguous
    assert 20 not in {c.wid for c in yasir.candidates}


def test_similar_names_do_not_fabricate_hierarchy_entities(prod_shapes_db):
    """The reported symptom: a query about Yasir Ali produced
    unit_head='Asif Ali' AND zonal_head='Asif Ali', then asked a
    clarifying question about the wrong person entirely."""
    entities = entity_extractor.extract_entities("tell me about yasir ali", prod_shapes_db)
    assert entities.get("unit_head") != "Asif Ali"
    assert entities.get("zonal_head") != "Asif Ali"


# =====================================================================
# 7. Duplicate advisor names -> never silently choose one
# =====================================================================

def test_duplicate_names_never_silently_choose(prod_shapes_db):
    response = _ask(prod_shapes_db, "Tell me about Yasir Ali")
    assert response["type"] == "clarification"
    assert response["data"] is None          # no advisor record was returned


def test_resolver_exposes_no_wid_for_a_duplicated_name(prod_shapes_db):
    """A caller that forgets to check `.status` must fail loudly rather
    than silently use candidate zero."""
    resolution = advisor_resolver.resolve_advisor("Yasir Ali", prod_shapes_db)
    assert resolution.is_ambiguous
    assert resolution.wid is None
    assert resolution.confidence == 0.0
    assert {c.wid for c in resolution.candidates} == {10, 11, 12}


def test_partial_name_does_not_pick_the_lowest_wid(prod_shapes_db):
    """"Ali" used to return 1 of 90 matching rows with no signal that 89
    others existed."""
    assert advisor_resolver.resolve_advisor("Ali", prod_shapes_db).status == advisor_resolver.NOT_FOUND


def test_substring_name_does_not_match_a_longer_name(prod_shapes_db):
    """"Adeel Dogar" must not resolve to "Adeel Mubarik Dogar"."""
    assert advisor_resolver.resolve_advisor("Adeel Dogar", prod_shapes_db).status == advisor_resolver.NOT_FOUND


def test_choosing_a_duplicate_carries_that_wid_forward(prod_shapes_db):
    """Once answered, the choice sticks for the rest of the session —
    re-asking discards what the user already told us."""
    _ask(prod_shapes_db, "Tell me about Yasir Ali", session_id="dup")
    chosen = _ask(prod_shapes_db, "11", session_id="dup")
    assert chosen["type"] == "advisor"
    assert chosen["data"]["wid"] == 11

    followup = _ask(prod_shapes_db, "who is his BM?", session_id="dup")
    assert followup["type"] == "manager"
    assert followup["data"]["manager"] == "Fraz Khalid"   # wid 11's BM, not wid 10's


# =====================================================================
# Controls — the working behavior must not regress
# =====================================================================

def test_leaderboards_still_work(prod_shapes_db):
    response = _ask(prod_shapes_db, "top 3 advisors by connects")
    assert response["type"] == "leaderboard"
    assert len(response["data"]) == 3


def test_team_summary_still_works(prod_shapes_db):
    response = _ask(prod_shapes_db, "how is Blue Area doing")
    assert response["type"] == "team"


def test_greeting_still_works(prod_shapes_db):
    assert _ask(prod_shapes_db, "hi")["type"] == "text"
