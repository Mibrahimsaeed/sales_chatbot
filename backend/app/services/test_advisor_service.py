"""advisor_service after the Phase 2 entity-resolution split.

The replaced implementation was a single query:
    SELECT * FROM advisor_profile WHERE name ILIKE '%q%' ORDER BY wid LIMIT 1
which returned a DIFFERENT PERSON in two distinct ways — substring
containment ("Ahmed Ali" -> "Ahmed Ali Pirzada") and silent lowest-wid
selection among duplicates. Both are locked out here.

`find_advisor_by_name` no longer exists: resolution (name -> which human)
belongs to advisor_resolver, and this module only fetches by wid.
"""

import pytest

from app.database.models import Advisor
from app.llm import advisor_resolver
from app.services.advisor_service import (
    find_advisor_by_wid, find_advisor_candidates, find_advisors_by_wids, resolve_advisor,
)


@pytest.fixture(autouse=True)
def _reset_resolver():
    advisor_resolver._reset_for_tests()
    yield
    advisor_resolver._reset_for_tests()


@pytest.fixture()
def lookup_db(db_session):
    db_session.add_all([
        Advisor(wid=100, name="Ahmed Ali", team="AMD", company="IMARAT"),
        Advisor(wid=101, name="Ahmed Ali Pirzada", team="Blue Area", company="Graana"),
        Advisor(wid=200, name="Yasir Ali", team="North/KPK", company="Agency21"),
        Advisor(wid=201, name="Yasir Ali", team="Downtown", company="IMARAT"),
        Advisor(wid=300, name="Waqar Haider", team="Blue Area", company="Graana"),
        Advisor(wid=400, name="Ghost", team="Nowhere", in_master_sheet=False),
    ])
    db_session.commit()
    return db_session


# ---- identity-first lookup ----

def test_find_by_wid_returns_exactly_that_person(lookup_db):
    assert find_advisor_by_wid(lookup_db, 101)["name"] == "Ahmed Ali Pirzada"
    assert find_advisor_by_wid(lookup_db, 100)["name"] == "Ahmed Ali"


def test_find_by_wid_returns_none_for_unknown_wid(lookup_db):
    assert find_advisor_by_wid(lookup_db, 99999) is None


def test_find_by_wid_respects_the_master_sheet_filter(lookup_db):
    """The view filters raw-data-only ghosts; a wid lookup must not be a
    back door around that."""
    assert find_advisor_by_wid(lookup_db, 400) is None


# ---- exact-before-substring ----

def test_exact_name_never_competes_with_a_longer_substring_match(lookup_db):
    """The audit's wrong-person case: "Ahmed Ali" exists exactly, so
    "Ahmed Ali Pirzada" must not be returned alongside (or instead of) it."""
    candidates = find_advisor_candidates(lookup_db, "Ahmed Ali")
    assert [c["wid"] for c in candidates] == [100]


def test_all_duplicates_are_returned_not_just_the_lowest_wid(lookup_db):
    candidates = find_advisor_candidates(lookup_db, "Yasir Ali")
    assert {c["wid"] for c in candidates} == {200, 201}


def test_substring_is_no_longer_a_resolution_tier(lookup_db):
    """Phase 2 removed `ILIKE '%q%'` from person resolution entirely. A
    fragment of a name is exactly the case that must ask rather than
    guess — "Pirzada" is not a request for whoever contains it."""
    assert find_advisor_candidates(lookup_db, "Pirzada") == []


def test_bare_partial_name_resolves_to_nothing_instead_of_1_of_90(lookup_db):
    """"Ali" used to return 1 of 90 rows with no signal that 89 others
    existed. It now matches nobody rather than inventing a winner."""
    assert find_advisor_candidates(lookup_db, "Ali") == []


def test_candidate_search_is_bounded(lookup_db):
    assert len(find_advisor_candidates(lookup_db, "Yasir Ali", limit=1)) == 1


def test_empty_query_returns_nothing_rather_than_the_whole_table(lookup_db):
    assert find_advisor_candidates(lookup_db, "") == []
    assert find_advisor_candidates(lookup_db, "   ") == []


def test_batch_fetch_by_wids(lookup_db):
    rows = find_advisors_by_wids(lookup_db, [100, 300])
    assert {r["wid"] for r in rows} == {100, 300}
    assert find_advisors_by_wids(lookup_db, []) == []


# ---- the ResolvedAdvisor contract ----

def test_resolve_returns_wid_name_confidence_for_a_unique_advisor(lookup_db):
    resolution = resolve_advisor(lookup_db, "Waqar Haider")
    assert resolution.is_resolved
    assert resolution.wid == 300
    assert resolution.name == "Waqar Haider"
    assert resolution.confidence == 1.0


def test_resolve_exposes_no_single_advisor_when_ambiguous(lookup_db):
    """Previously returned wid=200 with no indication a choice was made."""
    resolution = resolve_advisor(lookup_db, "Yasir Ali")
    assert resolution.is_ambiguous
    assert resolution.wid is None
    assert resolution.name is None
    assert resolution.confidence == 0.0
    assert len(resolution.candidates) == 2


def test_resolve_typo_via_high_confidence_fuzzy(lookup_db):
    resolution = resolve_advisor(lookup_db, "Waqar Haidar")   # one letter off
    assert resolution.is_resolved
    assert resolution.wid == 300


def test_resolve_swapped_word_order(lookup_db):
    resolution = resolve_advisor(lookup_db, "Haider Waqar")
    assert resolution.is_resolved
    assert resolution.wid == 300


def test_resolve_unknown_name(lookup_db):
    assert resolve_advisor(lookup_db, "Nobody Real At All").status == advisor_resolver.NOT_FOUND


def test_resolve_to_dict_shape(lookup_db):
    payload = resolve_advisor(lookup_db, "Yasir Ali").to_dict()
    assert payload["wid"] is None
    assert payload["status"] == advisor_resolver.AMBIGUOUS
    assert len(payload["candidates"]) == 2
    assert {"wid", "name", "team", "company", "score"} == set(payload["candidates"][0])
