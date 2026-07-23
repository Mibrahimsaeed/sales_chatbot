"""Part 8: paginated responses. A >15-row result caps at PAGE_SIZE with a
"Show More" cursor; handle_show_more() (shared by the POST /chat/more
button endpoint and typed "show more") walks through the remaining pages
until exhausted; a <=15-row result never triggers pagination at all.

Core mechanics are tested by driving _dispatch_ir directly with a
hand-built QueryIR — the rule-based text pipeline can never itself
produce limit=None (query_planner always defaults to 10), so an
unbounded "true total" query can only happen via the LLM in real usage;
building the IR directly here tests the pagination logic without
depending on NLU parsing quirks. A couple of tests still go through the
full handle_chat_message() pipeline where that's the actual thing being
tested (explicit "top N" via the rule-based path, and typed "show more"
recognition in nlu_pipeline.py)."""

import pytest

from app.database.models import Advisor, SalesFunnel
from app.llm import conversation_memory, entity_extractor, narrative, nlu_pipeline, semantic_parser
from app.llm.query_ir import MetricRef, QueryIR, Sort
from app.services.chat_service import PAGE_SIZE, _dispatch_ir, handle_chat_message, handle_show_more


@pytest.fixture()
def pagination_db(db_session, monkeypatch):
    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    # no network calls in this test — narrative polish would otherwise
    # hit the real (local) Ollama server on every dispatch
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _seed(db, n):
    for i in range(n):
        db.add(Advisor(wid=i + 1, name=f"Advisor {i + 1}", team="Alpha", company="IMARAT"))
        db.add(SalesFunnel(wid=i + 1, mtd_new_connect=n - i, mtd_followup_connect=0))
    db.commit()


def _unbounded_ir() -> QueryIR:
    return QueryIR(
        intent="leaderboard",
        subject_level="advisor",
        metric=MetricRef(key="total_connects", confidence=0.95),
        sort=Sort(metric="total_connects", direction="desc"),
        limit=None,
        overall_confidence=0.95,
    )


def _resolution(ir: QueryIR) -> nlu_pipeline.Resolution:
    return nlu_pipeline.Resolution(kind="ir", ir=ir, entities={})


def test_large_result_set_caps_at_page_size(pagination_db):
    _seed(pagination_db, 40)
    response = _dispatch_ir(pagination_db, _resolution(_unbounded_ir()), session_id="p1")

    assert response["shown_count"] == PAGE_SIZE
    assert response["total_count"] == 40
    assert response["has_more"] is True
    assert len(response["data"]) == PAGE_SIZE
    assert "Showing 15 of 40" in response["reply"]
    assert conversation_memory.get_pagination("p1") is not None


def test_small_result_set_never_paginates(pagination_db):
    _seed(pagination_db, 5)
    response = _dispatch_ir(pagination_db, _resolution(_unbounded_ir()), session_id="p2")

    assert response["has_more"] is False
    assert response["total_count"] == 5
    assert response["shown_count"] == 5
    assert "Showing" not in response["reply"]
    assert conversation_memory.get_pagination("p2") is None


def test_show_more_returns_next_batch_without_repeating(pagination_db):
    _seed(pagination_db, 40)
    first = _dispatch_ir(pagination_db, _resolution(_unbounded_ir()), session_id="p3")
    first_names = {r["name"] for r in first["data"]}

    second = handle_show_more(pagination_db, "p3")
    second_names = {r["name"] for r in second["data"]}

    assert len(second["data"]) == PAGE_SIZE
    assert not first_names & second_names  # no overlap between pages
    assert second["shown_count"] == 30
    assert second["has_more"] is True
    # numbering continues from where page 1 left off (16-30), doesn't
    # restart at 1 AND doesn't skip ahead — regression test for a bug
    # where reading state.offset AFTER advance_pagination() mutated it
    # in place made the header/numbering report the wrong page
    assert "Showing 30 of 40" in second["reply"]
    assert "\n16. " in second["reply"]
    assert "\n31. " not in second["reply"]


def test_show_more_walks_to_exhaustion(pagination_db):
    _seed(pagination_db, 40)
    _dispatch_ir(pagination_db, _resolution(_unbounded_ir()), session_id="p4")

    handle_show_more(pagination_db, "p4")  # rows 16-30
    third = handle_show_more(pagination_db, "p4")  # rows 31-40

    assert len(third["data"]) == 10
    assert third["shown_count"] == 40
    assert third["has_more"] is False
    assert conversation_memory.get_pagination("p4") is None


def test_show_more_with_no_active_cursor_is_graceful(pagination_db):
    response = handle_show_more(pagination_db, "nonexistent-session")
    assert response["data"] is None
    assert "nothing more" in response["reply"].lower()


def test_a_new_query_clears_stale_pagination_cursor(pagination_db):
    _seed(pagination_db, 40)
    _dispatch_ir(pagination_db, _resolution(_unbounded_ir()), session_id="p5")
    assert conversation_memory.get_pagination("p5") is not None

    # a small follow-up result has no more pages, so the old 40-row
    # cursor must not still be sitting there
    small_ir = _unbounded_ir().model_copy(update={"limit": 5})
    _dispatch_ir(pagination_db, _resolution(small_ir), session_id="p5")
    assert conversation_memory.get_pagination("p5") is None


# ---- through the full text pipeline: real rule-based parsing + typed "show more" ----

def test_explicit_top_n_through_real_pipeline_caps_at_n(pagination_db):
    _seed(pagination_db, 40)
    response = handle_chat_message(pagination_db, "top 20 advisors by connects", session_id="p6")

    assert response["total_count"] == 20
    assert response["has_more"] is True
    assert response["shown_count"] == PAGE_SIZE

    second = handle_show_more(pagination_db, "p6")
    assert second["shown_count"] == 20
    assert second["has_more"] is False


def test_typed_show_more_recognized_by_nlu_pipeline(pagination_db):
    _seed(pagination_db, 40)
    _dispatch_ir(pagination_db, _resolution(_unbounded_ir()), session_id="p7")

    response = handle_chat_message(pagination_db, "show more", session_id="p7")

    assert len(response["data"]) == PAGE_SIZE
    assert response["shown_count"] == 30


def test_typed_show_more_with_no_cursor_falls_through_normally(pagination_db):
    # no active pagination for this session — "show more" must not be
    # swallowed as a pagination request, it should resolve normally
    # (in this case as an unrecognized/clarify-style query)
    response = handle_chat_message(pagination_db, "show more", session_id="fresh-session")
    assert response.get("data") is None or "nothing more" not in response.get("reply", "").lower()
