"""Follow-up understanding belongs to the model, not to a patcher.

    conversation context + current query -> LLM -> complete SemanticModel

BEFORE THIS PHASE a bare follow-up never reached the model's answer. An
ellipsis detector classified the turn and `conversation_context.merge()`
rebuilt it field by field from the previous IR — copying the metric, the
period, the subject level, the filters and the limit. The model was
already being handed the conversation in its prompt; it simply had no
say.

WHAT THESE TESTS PIN, and the second is the one that matters:

  1. the conversation IS supplied — the previous resolved query and the
     recent turns are in the prompt, because a model that cannot see the
     context cannot resolve a reference to it.
  2. the model's answer STANDS. Deterministic code must not re-add a
     field the model left out, however obviously "wrong" that looks. A
     merge that silently restores the previous subject is indistinguishable,
     from the outside, from a model that understood the follow-up — which
     is exactly why the patcher hid this problem for so long.

The fake provider here returns scripted IRs so the pipeline's behaviour
is what is under test, not the model's. Whether the real model resolves
these correctly is a separate question, answered by the `live` test at
the bottom.
"""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod
from app.llm import (
    conversation_memory, entity_extractor, ir_patcher, nlu_pipeline, semantic_parser,
)

SESSION = "follow-ups"


@pytest.fixture()
def org(db_session, monkeypatch):
    db_session.add_all([
        Advisor(wid=1, name="Ahmed Raza", team="Blue Area", company="Graana"),
        Advisor(wid=2, name="Sara Iqbal", team="Blue Area", company="Graana"),
        Advisor(wid=3, name="Omar Farooq", team="Downtown", company="IMARAT"),
    ])
    db_session.add_all([
        Performance(wid=1, period=PerformancePeriod.MTD, cleared=900),
        Performance(wid=2, period=PerformancePeriod.MTD, cleared=100),
        Performance(wid=3, period=PerformancePeriod.MTD, cleared=500),
    ])
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "llm_first")
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()


def _ir(metric="total_connects", period="MTD", team="Blue Area",
        operation="group_metric", limit=None, sort=None):
    """One scripted model answer."""
    return {
        "intent": "filtered_list",
        "operation": operation,
        "subject_level": "team" if team else "advisor",
        "subjects": ([{"type": "team", "value": team, "match_confidence": 1.0}]
                     if team else []),
        "metric": {"key": metric, "confidence": 0.95} if metric else None,
        "metrics": [],
        "filters": [],
        "filter_tree": None,
        "time_range": {"mode": "snapshot", "period": period,
                       "compare_to": None, "confidence": 0.9},
        "sort": {"metric": sort or metric, "direction": "desc"},
        "limit": limit,
        "group_by": None,
        "target_level": None,
        "subject_of": None,
        "relation": "subtree",
        "overall_confidence": 0.95,
        "intent_confidence": 0.95,
    }


@pytest.fixture()
def script(monkeypatch):
    """Serves one scripted answer per turn and records every prompt."""
    state = {"answers": [], "prompts": []}

    def fake(prompt, schema, schema_name=None):
        state["prompts"].append(prompt)
        return state["answers"].pop(0) if state["answers"] else None

    monkeypatch.setattr(semantic_parser, "call_llm_structured", fake)
    return state


def _ask(db, text, script, answer):
    script["answers"].append(answer)
    conversation_memory.record_turn(SESSION, "user", text)
    return nlu_pipeline.resolve(text, db, session_id=SESSION)


# ---------------------------------------------------------------------
# 1. The conversation reaches the model
# ---------------------------------------------------------------------

def test_the_previous_query_and_turns_are_put_in_the_prompt(org, script):
    _ask(org, "show connects for blue area", script, _ir())
    _ask(org, "what about year to date", script, _ir(period="YTD"))

    follow_up_prompt = script["prompts"][1]
    assert "Previous turn's resolved query" in follow_up_prompt
    assert "total_connects" in follow_up_prompt, "the prior IR itself, not just a label"
    assert "Blue Area" in follow_up_prompt
    assert "show connects for blue area" in follow_up_prompt, "the recent turn"


def test_the_first_turn_has_no_prior_query_to_offer(org, script):
    _ask(org, "show connects for blue area", script, _ir())

    assert "Previous turn's resolved query" not in script["prompts"][0]


# ---------------------------------------------------------------------
# 2. The model's answer stands — no field-by-field repair
# ---------------------------------------------------------------------

def test_a_time_follow_up_takes_the_models_period(org, script):
    _ask(org, "show connects for blue area", script, _ir(period="MTD"))
    result = _ask(org, "what about year to date", script, _ir(period="YTD"))

    assert result.kind == "ir"
    assert result.ir.time_range.period == "YTD"
    assert result.ir.metric.key == "total_connects", "carried by the MODEL"
    assert [s.value for s in result.ir.subjects] == ["Blue Area"]


def test_a_dropped_field_is_not_silently_restored(org, script):
    """THE TEST THIS PHASE EXISTS FOR.

    The model answers the follow-up without a subject. That may well be a
    worse reading — but deterministic code re-adding "Blue Area" from the
    previous turn is the behaviour being removed, and it is invisible from
    the outside: the reply looks like a model that understood.
    """
    _ask(org, "show connects for blue area", script, _ir(team="Blue Area"))
    result = _ask(org, "what about year to date", script,
                  _ir(team=None, period="YTD"))

    assert result.ir.subjects == [], "the merge would have put Blue Area back"
    assert result.ir.time_range.period == "YTD"


def test_the_patcher_is_not_consulted_when_the_model_answers(org, script, monkeypatch):
    calls = []
    monkeypatch.setattr(ir_patcher, "try_patch",
                        lambda *a, **k: calls.append(a) or None)

    _ask(org, "show connects for blue area", script, _ir())
    _ask(org, "top 5", script, _ir(operation="leaderboard", limit=5))

    assert not calls, "follow-up understanding is the model's job now"


# ---------------------------------------------------------------------
# 3. Metric, subject and repeated follow-ups
# ---------------------------------------------------------------------

def test_a_metric_change_follow_up(org, script):
    _ask(org, "show connects for blue area", script, _ir(metric="total_connects"))
    result = _ask(org, "what about revenue", script, _ir(metric="mtd_cleared"))

    assert result.ir.metric.key == "mtd_cleared"
    assert [s.value for s in result.ir.subjects] == ["Blue Area"]


def test_a_subject_change_follow_up(org, script):
    _ask(org, "show connects for blue area", script, _ir(team="Blue Area"))
    result = _ask(org, "and downtown", script, _ir(team="Downtown"))

    assert [s.value for s in result.ir.subjects] == ["Downtown"]
    assert result.ir.metric.key == "total_connects"


def test_several_follow_ups_in_a_row(org, script):
    """Each turn's answer becomes the next turn's context, so the chain
    accumulates in memory rather than in a patch."""
    _ask(org, "show connects for blue area", script, _ir())
    _ask(org, "what about year to date", script, _ir(period="YTD"))
    _ask(org, "and revenue instead", script, _ir(metric="mtd_cleared", period="YTD"))
    result = _ask(org, "downtown please", script,
                  _ir(metric="mtd_cleared", period="YTD", team="Downtown"))

    assert result.ir.metric.key == "mtd_cleared"
    assert result.ir.time_range.period == "YTD"
    assert [s.value for s in result.ir.subjects] == ["Downtown"]

    # every turn saw the one before it
    assert "total_connects" in script["prompts"][1]
    assert "YTD" in script["prompts"][2]
    assert "mtd_cleared" in script["prompts"][3]


def test_each_turn_is_stored_as_the_next_turns_context(org, script):
    _ask(org, "show connects for blue area", script, _ir())
    assert conversation_memory.get(SESSION).metric.key == "total_connects"

    _ask(org, "what about revenue", script, _ir(metric="mtd_cleared"))
    stored = conversation_memory.get(SESSION)
    assert stored.metric.key == "mtd_cleared", "the model's answer is what is stored"


# ---------------------------------------------------------------------
# 4. The degrade path still remembers
# ---------------------------------------------------------------------

def test_the_deterministic_path_still_handles_follow_ups_when_the_model_is_down(org, script):
    """conversation_context and ir_patcher are kept for exactly this.
    An outage must not become a conversation that forgets."""
    _ask(org, "show top advisors by revenue", script,
         _ir(metric="mtd_cleared", operation="leaderboard", team=None, limit=10))

    # provider returns nothing for the follow-up
    script["answers"].append(None)
    conversation_memory.record_turn(SESSION, "user", "top 3")
    result = nlu_pipeline.resolve("top 3", org, session_id=SESSION)

    assert result.kind == "ir"
    assert result.ir.limit == 3, "the patcher still supplies the previous turn"
    assert result.ir.metric.key == "mtd_cleared"


# ---------------------------------------------------------------------
# 5. Live — whether the real model actually resolves follow-ups
# ---------------------------------------------------------------------

@pytest.mark.live
def test_the_real_model_carries_context_across_a_follow_up():
    """`pytest -m live`. The hermetic tests above prove the pipeline
    defers to the model; this shows the model actually uses the context.

    MEASURED, turn 2 of "show connects for blue area" / "what about year
    to date": the model returned subject=Blue Area and period=YTD. Both
    came from the conversation — the second message names neither — so
    the context genuinely reached it and its answer genuinely stands.
    """
    from app.database.session import SessionLocal

    db = SessionLocal()
    conversation_memory._store.clear()
    try:
        first = nlu_pipeline.resolve("show connects for blue area", db,
                                     session_id="live-followup")
        assert first.kind == "ir"
        conversation_memory.record_turn("live-followup", "user",
                                        "show connects for blue area")
        second = nlu_pipeline.resolve("what about year to date", db,
                                      session_id="live-followup")
    finally:
        conversation_memory._store.clear()
        db.close()

    assert second.kind == "ir"
    assert second.ir.time_range.period == "YTD", "the window the follow-up named"
    assert [s.value for s in second.ir.subjects] == ["Blue Area"], \
        "the subject carried over, and only the conversation could supply it"


@pytest.mark.live
@pytest.mark.xfail(
    reason="MEASURED DEFECT, not a flaky test. Asked 'what about year to "
           "date' after a connects question, gpt-4o-mini returns "
           "metric=ytd_cleared — it folds the window into the metric KEY "
           "and lands on a different measure entirely (revenue cleared "
           "instead of connects). The subject and the period are correct, "
           "so this is not a context-plumbing failure; it is the model "
           "choosing the wrong measure. Recorded here rather than removed "
           "so the fix has a test waiting for it.",
    strict=False,
)
def test_the_real_model_keeps_the_measure_across_a_time_follow_up():
    """A follow-up that names only a WINDOW must not change the MEASURE.

    "connects" then "what about year to date" is one question about
    connects over a different period. Returning revenue is a confident,
    well-formed answer to a question nobody asked.
    """
    from app.database.session import SessionLocal

    db = SessionLocal()
    conversation_memory._store.clear()
    try:
        nlu_pipeline.resolve("show connects for blue area", db,
                             session_id="live-measure")
        conversation_memory.record_turn("live-measure", "user",
                                        "show connects for blue area")
        second = nlu_pipeline.resolve("what about year to date", db,
                                      session_id="live-measure")
    finally:
        conversation_memory._store.clear()
        db.close()

    resolved = second.ir.primary_metric() or ""
    assert "connect" in resolved, (
        f"the follow-up named a window, not a measure, but the metric "
        f"became {resolved!r}")
