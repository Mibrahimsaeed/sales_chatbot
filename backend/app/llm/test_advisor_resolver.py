"""Identity resolution (Phase 1 refactor). These tests encode the rule the
whole refactor exists to enforce: a NAME IS NOT AN IDENTIFIER. Production
has 238 duplicate-name groups — 8 real people are named "Yasir Ali" — so
any path that turns a name into exactly one person without checking must
be treated as a bug."""

import pytest

from app.database.models import Advisor
from app.llm import advisor_resolver
from app.llm.advisor_resolver import (
    AMBIGUOUS, NOT_FOUND, RESOLVED,
    resolve_by_name, resolve_choice, resolve_from_text,
)


@pytest.fixture(autouse=True)
def _reset():
    advisor_resolver._reset_for_tests()
    yield
    advisor_resolver._reset_for_tests()


@pytest.fixture()
def identity_db(db_session):
    db_session.add_all([
        Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana"),
        # three real people sharing one name — the production shape
        Advisor(wid=2, name="Yasir Ali", team="North/KPK", company="Agency21"),
        Advisor(wid=3, name="Yasir Ali", team="Downtown", company="IMARAT"),
        Advisor(wid=4, name="Yasir Ali", team="Blue Area", company="Graana"),
        Advisor(wid=5, name="Adeel Mubarik Dogar", team="Downtown", company="IMARAT"),
        # a raw-data-only ghost must never be resolvable
        Advisor(wid=6, name="Ghost Advisor", team="Nowhere", in_master_sheet=False),
    ])
    db_session.commit()
    return db_session


# ---- unique name ----

def test_unique_name_resolves_to_a_single_wid(identity_db):
    result = resolve_by_name("Waqar Haider", identity_db)
    assert result.status == RESOLVED
    assert result.wid == 1
    assert result.identity.name == "Waqar Haider"


def test_name_match_is_case_insensitive(identity_db):
    assert resolve_by_name("waqar haider", identity_db).wid == 1


# ---- duplicate names: the core bug ----

def test_duplicate_name_is_ambiguous_not_a_silent_pick(identity_db):
    """The old advisor_service did `ORDER BY wid LIMIT 1` and returned
    wid=2 here, making the other two people unreachable and the ambiguity
    invisible to the caller."""
    result = resolve_by_name("Yasir Ali", identity_db)
    assert result.status == AMBIGUOUS
    assert {c.wid for c in result.candidates} == {2, 3, 4}


def test_ambiguous_resolution_exposes_no_wid(identity_db):
    """`.wid` must be None on an ambiguous result — a caller that forgets
    to check `.status` has to fail loudly rather than silently use
    candidate zero, which is precisely the replaced bug."""
    result = resolve_by_name("Yasir Ali", identity_db)
    assert result.wid is None
    assert result.identity is None


def test_candidates_carry_context_that_makes_the_question_answerable(identity_db):
    result = resolve_by_name("Yasir Ali", identity_db)
    labels = {c.label() for c in result.candidates}
    assert "Yasir Ali (North/KPK · Agency21)" in labels
    assert "Yasir Ali (Downtown · IMARAT)" in labels


# ---- not found / ghosts ----

def test_unknown_name_is_not_found(identity_db):
    assert resolve_by_name("Nobody Real At All", identity_db).status == NOT_FOUND


def test_non_master_sheet_advisor_is_never_resolvable(identity_db):
    assert resolve_by_name("Ghost Advisor", identity_db).status == NOT_FOUND


# ---- resolution from free text ----

def test_resolves_a_name_embedded_in_a_sentence(identity_db):
    result = resolve_from_text("tell me about Waqar Haider please", identity_db)
    assert result.status == RESOLVED
    assert result.wid == 1


def test_whole_sentence_no_longer_drags_in_an_unrelated_person(identity_db):
    """The audit's headline failure: "show adeel dogar's team" fuzzy-
    matched the WHOLE SENTENCE against advisor names, hit "Adeel Mubarik
    Dogar" at 0.62, and returned that unrelated person's profile. Token-
    window matching at the person floor must not resolve anyone here —
    "Adeel Dogar" is a unit head, not an advisor in this fixture."""
    result = resolve_from_text("show adeel dogar's team", identity_db)
    assert result.status == NOT_FOUND


def test_gibberish_resolves_to_nothing(identity_db):
    assert resolve_from_text("xyzzy plugh quux", identity_db).status == NOT_FOUND


def test_duplicate_name_from_text_is_ambiguous(identity_db):
    result = resolve_from_text("how is Yasir Ali doing", identity_db)
    assert result.status == AMBIGUOUS
    assert len(result.candidates) == 3


# ---- answering the disambiguation question ----

def test_choice_by_wid(identity_db):
    candidates = resolve_by_name("Yasir Ali", identity_db).candidates
    assert resolve_choice("3", candidates).wid == 3


def test_choice_by_distinguishing_team(identity_db):
    candidates = resolve_by_name("Yasir Ali", identity_db).candidates
    assert resolve_choice("the one in Downtown", candidates).wid == 3


def test_ambiguous_choice_returns_none_rather_than_guessing(identity_db):
    candidates = resolve_by_name("Yasir Ali", identity_db).candidates
    # "Yasir Ali" doesn't distinguish any of them — must not pick one
    assert resolve_choice("Yasir Ali", candidates) is None


# ---- Phase 3: span extraction ----

def test_span_extraction_strips_filler_and_possessives():
    from app.llm.advisor_resolver import extract_name_spans
    assert extract_name_spans("show adeel dogar's team") == ["adeel dogar"]
    assert extract_name_spans("who reports to Waqar Haider") == ["waqar haider"]
    assert extract_name_spans("tell me about Yasir Ali") == ["yasir ali"]


def test_span_extraction_strips_hierarchy_keywords():
    """A level keyword is never part of a person's name — leaving "unit
    head" in the span would drag the score below the 0.90 floor and lose
    an otherwise-perfect match."""
    from app.llm.advisor_resolver import extract_name_spans
    assert extract_name_spans("tell me about unit head Fraz Khalid") == ["fraz khalid"]
    assert extract_name_spans("who is the BM of Sana Khan") == ["sana khan"]


def test_span_extraction_returns_longest_span_first():
    """A longer span is more specific: "adeel mubarik dogar" must be tried
    before any shorter fragment, or a different person could match first."""
    from app.llm.advisor_resolver import extract_name_spans
    spans = extract_name_spans("about Adeel Mubarik Dogar and Sana Khan")
    assert spans[0] == "adeel mubarik dogar"


def test_text_with_no_name_resolves_to_nobody(identity_db):
    """Stripping leaves at most stray non-name tokens ("well"), which is
    fine — what matters is that none of them resolve to a person. The
    stopword list is deliberately conservative, since dropping a token
    that WAS part of a name breaks resolution outright, whereas an extra
    token merely fails the 0.90 floor."""
    assert resolve_from_text("who is doing well", identity_db).status == NOT_FOUND


def test_resolution_records_the_matched_span_for_traceability(identity_db):
    """Which words were treated as the name is the question that was
    unanswerable when the whole sentence was fed to the matcher."""
    result = resolve_from_text("tell me about Waqar Haider please", identity_db)
    assert result.matched_text == "waqar haider"


def test_filler_words_cannot_contribute_to_a_person_match(identity_db):
    """The original failure mode: filler scored as if it were part of the
    name. "performance of Waqar" must not resolve — "waqar" alone is a
    partial name, and the filler around it can no longer prop it up."""
    assert resolve_from_text("performance of Waqar", identity_db).status == NOT_FOUND


# ---- Phase 3: decisive winner ----

def test_near_tie_between_different_names_asks_instead_of_picking(db_session):
    """Two DIFFERENT people, both clearing the 0.90 floor and within
    AMBIGUITY_MARGIN of each other (0.966 vs 0.933). A plain "highest
    score wins" would silently pick the first — a 0.03 scoring accident
    deciding which human the user gets told about."""
    db_session.add_all([
        Advisor(wid=1, name="Ahmed Ali Khan", team="Alpha", company="Graana"),
        Advisor(wid=2, name="Ahmad Ali Khaan", team="Beta", company="IMARAT"),
    ])
    db_session.commit()
    advisor_resolver._reset_for_tests()

    result = advisor_resolver.resolve_advisor("Ahmed Ali Khaan", db_session)
    assert result.status == AMBIGUOUS
    assert result.wid is None
    assert {c.wid for c in result.candidates} == {1, 2}


def test_a_clear_winner_still_resolves_despite_a_distant_runner_up(db_session):
    db_session.add_all([
        Advisor(wid=1, name="Waqar Haider", team="Alpha", company="Graana"),
        Advisor(wid=2, name="Sana Khan", team="Beta", company="IMARAT"),
    ])
    db_session.commit()
    advisor_resolver._reset_for_tests()

    result = advisor_resolver.resolve_advisor("Waqar Haidar", db_session)
    assert result.status == RESOLVED
    assert result.wid == 1


def test_person_floor_is_ninety():
    assert advisor_resolver.PERSON_FLOOR == 0.90
