"""Phase 32 — "all" means all.

    "connects of all BCMs"  ->  10 rows, total=10, has_more=False

181 BCMs exist. The answer showed ten of them and reported ten as the
count, so nothing on screen or in the payload said the list had been
cut — the same default of 10 that serves "top advisors by connects"
was applied to a question that said ALL.

TWO DEFAULTS HAD TO GO, and finding only the first is why the first
attempt changed nothing: query_planner set the limit, and then
`plan_to_ir` re-imposed it with `plan.limit or 10`, so a lifted cap
arrived at the compiler restored. The plan's own default is 10, so
`None` now means exactly what it says.

RANKINGS KEEP THE CAP. "top advisors by connects" and "who has most
connects" ask who is ahead, not for the roll; only an explicit
all/every/each lifts it, and a stated "top 5" beats both.

THE PAGING WAS NOT SAFE TO USE. Lifting the cap exposed a defect that
had been unreachable while every answer was one page: `ORDER BY value`
had no tiebreaker, and 136 advisors tie at 0 connects, so a LIMIT/OFFSET
walk returned 573 rows carrying 570 distinct people — three shown twice,
three never shown. A list that silently drops people is not the list
that was asked for, so the walk-to-completion tests below are the point
of this file, not a nicety.
"""

import pytest

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import (
    advisor_resolver, conversation_memory, entity_extractor, narrative,
    nlu_pipeline, semantic_parser,
)
from app.services.chat_service import PAGE_SIZE, handle_chat_message, handle_show_more

# 40 advisors across 20 BCMs, 8 zonal heads and 3 unit heads — every
# level lands on both sides of the 15-row page so the See More boundary
# is exercised rather than assumed.
#
# MOST CONNECTS ARE ZERO on purpose. Ties are what made paging
# non-deterministic, and a fixture of distinct values would pass with no
# tiebreaker at all.
ADVISORS = 40
BCMS = 20
ZONAL_HEADS = 8
UNIT_HEADS = 3


def _connects(wid: int) -> int:
    """Three distinct values and a long tail of zeros — 34 of the 40 tie."""
    return {1: 500, 2: 400, 3: 300, 4: 200, 5: 100, 6: 50}.get(wid, 0)


@pytest.fixture()
def db(db_session, monkeypatch):
    for wid in range(1, ADVISORS + 1):
        db_session.add(Advisor(
            wid=wid, name=f"Advisor {wid:02d}", team=f"Team {wid % 4}",
            company="Graana",
            rm=f"Unit Head {wid % UNIT_HEADS}",
            portfolio_lead=f"Zonal Head {wid % ZONAL_HEADS}",
            management_lead=f"Bcm {wid % BCMS:02d}",
            in_master_sheet=True))
        value = _connects(wid)
        db_session.add(Calls(wid=wid, connects_mtd=value, connects_daily=0,
                             answered_calls_mtd=value, answered_calls_daily=0))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=value,
                                   mtd_followup_connect=0, mtd_cr=value,
                                   mtd_new_meeting=value, mtd_followup_meeting=0))
        db_session.add(Pipeline(wid=wid, pipeline=value, overdue=0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=value))
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


def _ask(db, text, session="p32"):
    conversation_memory._store.pop(session, None)
    return handle_chat_message(db, text, session_id=session)


def _identity(row):
    """What makes an output row a distinct subject — the advisor's wid, or
    the group's name above advisor level."""
    return row.get("wid"), row.get("name")


def _walk(db, text):
    """Every row the question can reach, page by page, exactly as the UI
    does: the first answer, then Show More until it says there is none."""
    session = f"walk-{text}"
    conversation_memory._store.pop(session, None)
    response = handle_chat_message(db, text, session_id=session)
    rows = [_identity(r) for r in response["data"]]
    pages = 1
    while response.get("has_more"):
        response = handle_show_more(db, session)
        rows += [_identity(r) for r in response["data"]]
        pages += 1
    return rows, pages, response


# ---------------------------------------------------------------------
# The cap comes off when the query says "all"
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query,expected", [
    ("connects of all BCMs", BCMS),
    ("connects of all zonal heads", ZONAL_HEADS),
    ("connects of all unit heads", UNIT_HEADS),
    ("connects of all advisors", ADVISORS),
])
def test_an_all_query_counts_every_member(db, query, expected):
    """`total_count` is the TRUE number of matches. Reporting 10 was the
    part that made the truncation invisible."""
    assert _ask(db, query)["total_count"] == expected


@pytest.mark.parametrize("query,expected", [
    ("connects of all BCMs", BCMS),
    ("connects of all advisors", ADVISORS),
])
def test_a_long_list_offers_see_more(db, query, expected):
    response = _ask(db, query)
    assert expected > PAGE_SIZE
    assert response["has_more"] is True
    assert len(response["data"]) == PAGE_SIZE


@pytest.mark.parametrize("query,expected", [
    ("connects of all unit heads", UNIT_HEADS),
    ("connects of all zonal heads", ZONAL_HEADS),
])
def test_a_short_list_is_shown_whole_with_no_see_more(db, query, expected):
    response = _ask(db, query)
    assert expected <= PAGE_SIZE
    assert response["has_more"] is False
    assert len(response["data"]) == expected


@pytest.mark.parametrize("word", ["all", "every", "each"])
def test_any_enumeration_word_lifts_the_cap(db, word):
    assert _ask(db, f"connects of {word} BCMs")["total_count"] == BCMS


@pytest.mark.parametrize("metric", ["connects", "CR", "meetings", "pipeline",
                                    "answered calls"])
def test_the_cap_comes_off_for_every_metric(db, metric):
    """The limit belongs to the question's shape, not to the measure."""
    assert _ask(db, f"{metric} of all BCMs")["total_count"] == BCMS


# ---------------------------------------------------------------------
# See More reaches every single one, exactly once
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query,expected", [
    ("connects of all BCMs", BCMS),
    ("connects of all zonal heads", ZONAL_HEADS),
    ("connects of all advisors", ADVISORS),
])
def test_paging_reaches_every_member_exactly_once(db, query, expected):
    """THE guarantee. 34 of these 40 advisors tie at 0 connects, so
    without a deterministic tiebreaker the walk repeats some rows and
    skips others — and the totals still look right, which is what made
    the defect survive."""
    rows, _pages, _last = _walk(db, query)
    assert len(rows) == expected
    assert len(set(rows)) == expected


def test_paging_stops_on_its_own(db):
    rows, pages, last = _walk(db, "connects of all advisors")
    assert last["has_more"] is False
    assert pages == -(-ADVISORS // PAGE_SIZE)
    assert len(rows) == ADVISORS


def test_a_tied_block_is_ordered_the_same_way_every_time(db):
    """The tiebreaker must be stable across QUERIES, not merely within
    one — two identical questions that disagree about page 2 are the same
    defect seen from a different angle."""
    first, _p, _l = _walk(db, "connects of all advisors")
    second, _p2, _l2 = _walk(db, "connects of all advisors")
    assert first == second


def test_the_ranking_order_is_unchanged_by_the_tiebreaker(db):
    """It orders EQUALS only. The advisors with real values must still
    lead, in value order."""
    rows, _pages, _last = _walk(db, "connects of all advisors")
    leaders = [wid for wid, _name in rows[:6]]
    assert leaders == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------
# Rankings keep the cap
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "top advisors by connects", "who has most connects",
    "lowest advisors by connects",
])
def test_a_ranking_still_shows_ten(db, query):
    """"Top"/"most" ask who is ahead, not for the roll. Uncapping these
    would answer a different question."""
    assert len(_ask(db, query)["data"]) == 10


def test_a_stated_number_beats_the_enumeration_word(db):
    """"top 5 of all BCMs" — a stated size is the most specific thing the
    user can say, so it wins over both defaults."""
    assert len(_ask(db, "top 5 of all BCMs by connects")["data"]) == 5


def test_a_plain_group_query_is_unchanged(db):
    """No "all", no ranking word: the default of 10 still applies, so
    nothing outside the enumeration case moved."""
    assert len(_ask(db, "connects by bcm")["data"]) <= 10


# ---------------------------------------------------------------------
# The limit rule itself
# ---------------------------------------------------------------------


def test_an_all_query_carries_no_limit_in_the_ir(db):
    """`None` is what chat_service reads as "page the true match count".
    Pinned at the IR so the rule is visible without running a query."""
    assert nlu_pipeline.resolve("connects of all BCMs", db).ir.limit is None


def test_a_ranking_carries_the_default_limit_in_the_ir(db):
    assert nlu_pipeline.resolve("top advisors by connects", db).ir.limit == 10


def test_a_stated_limit_survives_to_the_ir(db):
    assert nlu_pipeline.resolve("top 5 of all BCMs by connects", db).ir.limit == 5
