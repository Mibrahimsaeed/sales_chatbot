"""Prompt decomposition (app/llm/prompt_inspector.py).

The properties that matter:

1. LOSSLESS. The sections are a partition of the prompt, so re-joining
   them reproduces it byte-for-byte. Everything the audit log claims
   rests on this; a breakdown that drops or duplicates text would be
   worse than no breakdown, because it would look authoritative.
2. FAITHFUL TO THE REAL BUILDERS. The anchors are checked against
   prompts produced by build_ir_prompt() / build_planner_prompt()
   themselves, not against hand-written samples that could agree with a
   stale anchor set while production disagrees.
3. HONEST ABOUT ROLES. The app sends one user message; the breakdown
   must not imply a system-role message was transmitted.
"""

import pytest

from app.llm import prompt_inspector as pi
from app.llm.planner_prompt import build_planner_prompt
from app.llm.prompt_builder import build_ir_prompt


@pytest.fixture()
def ir_prompt():
    # Names Kaleem Ullah explicitly: since TASK 3 a person gazetteer is
    # only sent when the message could actually be referring to someone on
    # it (prompt_builder._person_gazetteer), and this fixture exists to
    # check that the gazetteer is SEGMENTED as retrieved context — which
    # needs a prompt that has one.
    return build_ir_prompt(
        "top 5 advisors under Kaleem Ullah in Graana who were late",
        known_teams=["Blue Area", "Downtown"],
        known_companies=["Graana", "Agency21"],
        grounded_entities={"company": "Graana", "attendance_status": "Late"},
        prior_ir_json='{"intent": "leaderboard", "limit": 10}',
        known_unit_heads=["Kaleem Ullah"],
        known_zonal_heads=["Adeel Dogar"],
        known_bcms=["Gulberg"],
    )


def test_segmentation_is_lossless_for_the_real_parser_prompt(ir_prompt):
    breakdown = pi.segment(ir_prompt)
    assert breakdown.is_lossless
    assert breakdown.reconstruct() == ir_prompt


def test_segmentation_is_lossless_for_the_real_planner_prompt():
    prompt = build_planner_prompt(
        "who works under Kaleem",
        known_teams=["Blue Area"],
        known_companies=["Graana"],
        prior_plan_json='{"intent": "roster"}',
    )
    breakdown = pi.segment(prompt)
    assert breakdown.is_lossless


def test_all_five_views_are_populated_for_the_parser_prompt(ir_prompt):
    breakdown = pi.segment(ir_prompt, roles_sent=["user"])
    assert breakdown.present(pi.SYSTEM)
    assert breakdown.present(pi.DEVELOPER)
    assert breakdown.present(pi.CONTEXT)
    assert breakdown.present(pi.HISTORY)
    assert breakdown.present(pi.USER)


def test_retrieved_context_holds_the_gazetteer_and_grounded_entities(ir_prompt):
    context = "\n".join(s.text for s in pi.segment(ir_prompt).by_category(pi.CONTEXT))
    assert "Blue Area" in context           # teams gazetteer
    assert "Graana" in context              # companies gazetteer
    assert "Kaleem Ullah" in context        # unit heads
    assert "mtd_cleared" in context         # metric ontology
    assert "attendance_status" in context   # per-request grounded entities


def test_conversation_history_is_the_prior_turn_ir(ir_prompt):
    history = "\n".join(s.text for s in pi.segment(ir_prompt).by_category(pi.HISTORY))
    assert "Previous turn's resolved query" in history
    assert '"intent": "leaderboard"' in history


def test_history_is_absent_on_a_first_turn():
    """No prior IR means no history section — reported as absent rather
    than as an empty one, so "the model had no context" is visible."""
    prompt = build_ir_prompt("top 5 by revenue", ["Blue Area"], ["Graana"], {})
    assert not pi.segment(prompt).present(pi.HISTORY)


def test_user_message_section_carries_the_actual_query(ir_prompt):
    user = "\n".join(s.text for s in pi.segment(ir_prompt).by_category(pi.USER))
    assert "top 5 advisors under Kaleem Ullah in Graana who were late" in user


def test_nothing_is_left_unclassified_in_a_real_prompt(ir_prompt):
    """Unclassified text is the drift alarm: it means the builders emit
    something the inspector doesn't recognise."""
    assert not pi.segment(ir_prompt).present(pi.UNCLASSIFIED)


def test_unrecognised_prompt_survives_as_unclassified():
    breakdown = pi.segment("some prompt from a builder that doesn't exist yet")
    assert breakdown.is_lossless
    assert breakdown.present(pi.UNCLASSIFIED)


def test_empty_and_blank_prompts_do_not_crash():
    for text in ("", "\n", "\n\n   \n"):
        assert pi.segment(text).is_lossless


def test_roles_report_what_was_actually_transmitted(ir_prompt):
    """The role instruction is inlined text, NOT a system-role message."""
    breakdown = pi.segment(ir_prompt, roles_sent=["user"])
    assert breakdown.roles_sent == ["user"]
    assert breakdown.present(pi.SYSTEM)  # as a text section only
