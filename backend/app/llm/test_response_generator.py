"""The final answer is written by the model and grounded in the rows.

    verified result set -> final LLM generator -> final answer

The generator may explain, summarise, compare and converse. What it may
NOT do is state a figure, a name or a claim that is not in the result set
— and that is enforced, not requested. "Do not invent numbers" in a
prompt is a wish; a reply that reports 26,811 as 26,800, or mentions a
team that was never queried, is well-formed and confident and nothing
downstream can tell it is wrong.

Every rejection test below drives a REAL model response through the same
guard the production path uses, and asserts the deterministic explanation
is served instead. The fallback is the behaviour that existed before this
module, so a bad generation costs phrasing and never a number.
"""

import pytest

from app.llm import response_generator
from app.llm.response_generator import generate, generate_or

ROWS = [
    {"wid": 1, "name": "Ahmed Raza", "team": "Blue Area", "company": "Graana", "value": 900.0},
    {"wid": 2, "name": "Sara Iqbal", "team": "Blue Area", "company": "Graana", "value": 100.0},
]
FACTS = {"top": "Ahmed Raza", "top_value": 900.0, "rows": 2}
EXPLANATION = "Ahmed Raza leads with 900, ahead of Sara Iqbal on 100."
QUERY = "top advisors in Blue Area by revenue"
METADATA = {"metric": "mtd_cleared", "level": "advisor", "period": "MTD",
            "shown_count": 2, "total_count": 2, "has_more": False}


@pytest.fixture(autouse=True)
def narrative_on(monkeypatch):
    monkeypatch.setattr(response_generator.settings, "nlu_narrative", True)


@pytest.fixture()
def answers(monkeypatch):
    """Scripts the model's reply and records the prompt it was given."""
    state = {"reply": None, "prompts": []}

    def fake(prompt):
        state["prompts"].append(prompt)
        return {"answer": state["reply"]} if state["reply"] is not None else None

    monkeypatch.setattr(response_generator, "call_llm_json", fake)
    return state


def _run(answers, reply, **kw):
    answers["reply"] = reply
    return generate(QUERY, ROWS, FACTS, EXPLANATION, metadata=METADATA, **kw)


# ---------------------------------------------------------------------
# What it is allowed to do
# ---------------------------------------------------------------------

def test_a_grounded_answer_is_accepted(answers):
    result = _run(answers, "Ahmed Raza leads Blue Area with 900, well clear of "
                           "Sara Iqbal on 100.")
    assert result.accepted
    assert result.violations == []


def test_it_may_explain_and_compare_without_new_figures(answers):
    """Explaining is the point — a sentence with no numbers at all is
    still an answer, and must not be rejected for saying something the
    template did not."""
    result = _run(answers, "Ahmed Raza is well ahead of Sara Iqbal here, so the "
                           "team's total leans heavily on one person.")
    assert result.accepted


def test_it_may_state_the_size_of_the_result(answers):
    """Counting the rows in front of it is not a claim about data."""
    result = _run(answers, "Both advisors are shown; Ahmed Raza leads.")
    assert result.accepted


# ---------------------------------------------------------------------
# What it must not do
# ---------------------------------------------------------------------

def test_an_invented_number_is_rejected(answers):
    result = _run(answers, "Ahmed Raza leads with 950.")
    assert not result.accepted
    assert any("numbers not in the result set" in v for v in result.violations)


def test_a_recalculated_value_is_rejected(answers):
    """The sum is arithmetically correct and still forbidden: it is a
    figure the database never returned, and the moment one derived number
    is allowed there is no line to hold."""
    result = _run(answers, "Blue Area totals 1000 across the two advisors.")
    assert not result.accepted
    assert any("1000" in v for v in result.violations)


def test_a_rounded_value_is_rejected(answers):
    result = _run(answers, "Ahmed Raza is on about 890.")
    assert not result.accepted


def test_an_unsupported_entity_is_rejected(answers):
    """Downtown was never in these results."""
    result = _run(answers, "Ahmed Raza leads with 900, ahead of Downtown.")
    assert not result.accepted
    assert any("entities not in the result set" in v for v in result.violations)


def test_an_invented_person_is_rejected(answers):
    result = _run(answers, "Ahmed Raza leads with 900, followed by Bilal Khan.")
    assert not result.accepted
    assert any("Bilal Khan" in v for v in result.violations)


def test_a_rambling_answer_is_rejected(answers):
    result = _run(answers, "Ahmed Raza leads. " * 80)
    assert not result.accepted
    assert any("too long" in v for v in result.violations)


def test_an_empty_answer_is_rejected(answers):
    assert not _run(answers, "   ").accepted


def test_entities_named_in_the_question_are_allowed(answers):
    """The user's own words are context, not an invention — "Blue Area"
    appears in the question and in the rows."""
    result = _run(answers, "Across Blue Area, Ahmed Raza leads on 900.")
    assert result.accepted


def test_ordinary_capitalised_prose_is_not_treated_as_an_entity(answers):
    """The entity check must not fire on sentence openers, or every
    answer would be rejected."""
    result = _run(answers, "Overall, Ahmed Raza leads. However, Sara Iqbal "
                           "trails on 100.")
    assert result.accepted, result.violations


# ---------------------------------------------------------------------
# Inputs the generator is given
# ---------------------------------------------------------------------

def test_the_prompt_carries_the_query_rows_metadata_and_context(answers):
    _run(answers, "Ahmed Raza leads with 900.",
         recent_turns=[("user", "show me Blue Area"), ("assistant", "here you go")])
    prompt = answers["prompts"][0]

    assert QUERY in prompt
    assert "Ahmed Raza" in prompt and "900" in prompt, "the verified rows"
    assert "mtd_cleared" in prompt, "the result metadata"
    assert "show me Blue Area" in prompt, "the conversation context"
    assert "ONLY SOURCE OF TRUTH" in prompt


def test_an_empty_result_set_is_not_narrated(answers):
    """Nothing to ground against, and a model asked to be conversational
    about an empty set is exactly where inventions come from."""
    answers["reply"] = "There were no results, but performance looks healthy."
    result = generate(QUERY, [], FACTS, EXPLANATION, metadata=METADATA)

    assert not result.accepted
    assert answers["prompts"] == [], "the model is not even called"


def test_nothing_is_generated_when_the_feature_is_off(answers, monkeypatch):
    monkeypatch.setattr(response_generator.settings, "nlu_narrative", False)
    answers["reply"] = "Ahmed Raza leads with 900."

    assert not generate(QUERY, ROWS, FACTS, EXPLANATION).accepted
    assert answers["prompts"] == []


# ---------------------------------------------------------------------
# Fail-soft: the reply is never blanked or corrupted
# ---------------------------------------------------------------------

def test_generate_or_serves_the_generated_answer_when_it_is_grounded(answers):
    answers["reply"] = "Ahmed Raza leads Blue Area with 900."
    assert generate_or(EXPLANATION, QUERY, ROWS, FACTS,
                       metadata=METADATA) == "Ahmed Raza leads Blue Area with 900."


@pytest.mark.parametrize("bad_reply", [
    "Ahmed Raza leads with 950.",                  # invented number
    "Blue Area totals 1000.",                      # recalculated
    "Ahmed Raza beat Downtown.",                   # unsupported entity
    None,                                          # provider returned nothing
])
def test_generate_or_falls_back_to_the_deterministic_explanation(bad_reply, answers):
    """The worst case is exactly the behaviour that existed before this
    module — never a blank reply, never a wrong number."""
    answers["reply"] = bad_reply
    assert generate_or(EXPLANATION, QUERY, ROWS, FACTS,
                       metadata=METADATA) == EXPLANATION


def test_a_provider_failure_never_raises(monkeypatch):
    """A response layer that can throw turns a cosmetic feature into an
    outage: the caller is already midway through building a reply."""
    def boom(prompt):
        raise RuntimeError("provider down")

    monkeypatch.setattr(response_generator, "call_llm_json", boom)

    assert not generate(QUERY, ROWS, FACTS, EXPLANATION).accepted
    assert generate_or(EXPLANATION, QUERY, ROWS, FACTS) == EXPLANATION


# ---------------------------------------------------------------------
# End to end, through the reply the API actually returns
# ---------------------------------------------------------------------

def _chat_fixture(db_session, monkeypatch, reply):
    from app.database.models import Advisor, SalesFunnel
    from app.llm import conversation_memory, entity_extractor, semantic_parser

    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(response_generator.settings, "nlu_narrative", True)
    monkeypatch.setattr(response_generator, "call_llm_json",
                        lambda prompt: {"answer": reply} if reply else None)

    db_session.add_all([
        Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana"),
        Advisor(wid=2, name="Ali Raza", team="Blue Area", company="Graana"),
    ])
    db_session.add_all([
        SalesFunnel(wid=1, mtd_new_connect=10),
        SalesFunnel(wid=2, mtd_new_connect=4),
    ])
    db_session.commit()
    return db_session


def test_a_grounded_answer_reaches_the_api_reply(db_session, monkeypatch):
    """The generated sentence is what the user reads, and the response
    envelope is unchanged."""
    from app.services.chat_service import handle_chat_message

    db = _chat_fixture(db_session, monkeypatch,
                       "Waqar Haider is ahead on 10, with Ali Raza on 4.")
    result = handle_chat_message(db, "top advisors by connects", session_id="gen-1")

    assert "Waqar Haider is ahead on 10" in result["reply"]
    # the envelope the API already returned, unchanged: the generated
    # text goes in `reply` ahead of the rendered table, and no field of
    # the response was added, renamed or removed for it
    for key in ("type", "reply", "data", "insights"):
        assert key in result, key
    assert result["type"] == "leaderboard"
    assert len(result["data"]) == 2, "the rows are still returned as data"


def test_an_ungrounded_answer_never_reaches_the_user(db_session, monkeypatch):
    """The model claims a figure the database never produced. The user
    must see the deterministic explanation instead — and must never see
    the number 9999."""
    from app.services.chat_service import handle_chat_message

    db = _chat_fixture(db_session, monkeypatch,
                       "Waqar Haider dominated with 9999 connects.")
    result = handle_chat_message(db, "top advisors by connects", session_id="gen-2")

    assert "9999" not in result["reply"]
    assert result["reply"].strip(), "and the reply is not blanked"
