"""Chat audit log (app/core/audit.py).

Two properties matter and nothing else does:

1. With the DEBUG flag OFF, the module is inert — no capture, no file,
   no directory, no console output. It ships enabled-by-nobody.
2. With it ON, one user query produces one block that contains the query,
   a timestamp, the COMPLETE prompt text sent to the LLM, and the final
   response — because a prompt recorded partially is worthless for
   explaining why two runs of the same question disagreed.
"""

import pytest

from app.core import audit


@pytest.fixture()
def on(monkeypatch, tmp_path):
    """Audit enabled, writing into a tmp dir, console echo off."""
    monkeypatch.setattr(audit.settings, "chat_audit_debug", True)
    monkeypatch.setattr(audit.settings, "chat_audit_console", False)
    monkeypatch.setattr(audit.settings, "chat_audit_dir", str(tmp_path))
    monkeypatch.setattr(audit, "_file_logger", None)
    return tmp_path


@pytest.fixture()
def blocks(monkeypatch):
    """Captures formatted blocks instead of reading the file back."""
    seen = []
    monkeypatch.setattr(audit, "_emit", lambda entry: seen.append(audit._format(entry)))
    return seen


def test_disabled_by_default_records_nothing(monkeypatch, blocks, tmp_path):
    monkeypatch.setattr(audit.settings, "chat_audit_debug", False)
    monkeypatch.setattr(audit.settings, "chat_audit_dir", str(tmp_path))

    with audit.audit_query("top 5 by revenue"):
        audit.record_prompt("SECRET PROMPT", purpose="json")
        audit.record_response({"type": "leaderboard", "reply": "hi"})

    assert blocks == []
    assert not list(tmp_path.iterdir())


def test_block_carries_query_timestamp_prompt_and_response(on, blocks):
    prompt = "You are a query planner.\nOntology: revenue, connects\nUser: top 5 by revenue"

    with audit.audit_query("top 5 by revenue", session_id="s-1"):
        audit.record_prompt(prompt, purpose="structured:query_ir", model="gpt-4.1-mini")
        audit.record_llm_response('{"intent": "leaderboard"}')
        audit.record_response({
            "type": "leaderboard",
            "reply": "Top 5 advisors by revenue: ...",
            "data": [{"wid": 1}, {"wid": 2}],
        })

    assert len(blocks) == 1
    block = blocks[0]
    assert "CHAT AUDIT" in block
    assert "User Query:  top 5 by revenue" in block
    assert "Timestamp:" in block
    assert "Final Response:" in block
    assert "Top 5 advisors by revenue: ..." in block
    # The whole prompt, not a preview — the point of the module.
    assert prompt in block
    assert '{"intent": "leaderboard"}' in block
    assert "2 row(s)" in block


def test_prompt_is_recorded_even_when_inference_fails(on, blocks):
    """A timed-out or refused call degrades to the rule-based path — the
    single most likely cause of an inconsistent answer, so its prompt has
    to survive."""
    with audit.audit_query("who was late today"):
        audit.record_prompt("PROMPT-A", purpose="json")  # no response ever recorded
        audit.record_response({"type": "attendance", "reply": "3 people were late.", "data": []})

    assert "PROMPT-A" in blocks[0]
    assert "LLM Raw Output: None" in blocks[0]


def test_block_is_emitted_even_when_the_request_raises(on, blocks):
    with pytest.raises(ValueError):
        with audit.audit_query("boom"):
            audit.record_prompt("PROMPT-B", purpose="json")
            raise ValueError("kaboom")

    assert len(blocks) == 1
    assert "PROMPT-B" in blocks[0]


def test_nested_audit_query_does_not_start_a_second_block(on, blocks):
    """"show more" reached through _dispatch nests inside the outer
    message; one user message must stay one block."""
    with audit.audit_query("show more"):
        with audit.audit_query("[show more]"):
            audit.record_response({"type": "leaderboard", "reply": "next page", "data": []})

    assert len(blocks) == 1
    assert "User Query:  show more" in blocks[0]


def test_capture_outside_a_query_is_a_no_op(on, blocks):
    """ETL and startup probes call the LLM with no request in flight."""
    audit.record_prompt("STARTUP PROBE", purpose="json")
    audit.record_response({"type": "text", "reply": "x"})
    assert blocks == []


def test_writes_a_readable_file_at_the_documented_path(on):
    with audit.audit_query("how is Blue Area doing"):
        audit.record_prompt("PROMPT-C", purpose="json", model="gpt-4.1-mini")
        audit.record_response({"type": "team", "reply": "Blue Area: 40 connects.", "data": {}})

    path = audit.log_path()
    assert path == on / "chat_audit.log"
    text = path.read_text()
    assert "CHAT AUDIT" in text
    assert "how is Blue Area doing" in text
    assert "PROMPT-C" in text
    assert "Blue Area: 40 connects." in text


def test_breakdown_reports_the_five_views_and_the_real_roles(on, blocks):
    """The five requested views, plus the distinction that the role
    instruction is INLINE TEXT rather than a system-role message — a
    reader who assumed otherwise would draw the wrong conclusion about
    where to change behaviour."""
    from app.llm.prompt_builder import build_ir_prompt

    prompt = build_ir_prompt(
        "top 5 in Graana",
        known_teams=["Blue Area"],
        known_companies=["Graana"],
        grounded_entities={"company": "Graana"},
        prior_ir_json='{"intent": "leaderboard"}',
    )

    with audit.audit_query("top 5 in Graana"):
        audit.record_prompt(
            prompt,
            purpose="structured:query_ir",
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        audit.record_response({"type": "leaderboard", "reply": "…", "data": []})

    block = blocks[0]
    for heading in ("System Prompt", "Developer Prompt", "Retrieved Context",
                    "Conversation History", "User Message",
                    "Final Prompt sent to the LLM"):
        assert f"### {heading}" in block

    assert "roles transmitted: ['user']" in block
    assert "no system/developer role message" in block
    # The partition claim the whole breakdown rests on.
    assert "sections above reconstruct it exactly: yes" in block


def test_absent_sections_are_stated_not_omitted(on, blocks):
    """A first turn has no conversation history. Printing "(none)" makes
    "the model was given no prior context" visible; omitting the heading
    would leave it looking like it simply wasn't logged."""
    from app.llm.prompt_builder import build_ir_prompt

    prompt = build_ir_prompt("top 5 by revenue", ["Blue Area"], ["Graana"], {})
    with audit.audit_query("top 5 by revenue"):
        audit.record_prompt(prompt, purpose="structured:query_ir",
                            messages=[{"role": "user", "content": prompt}])
        audit.record_response({"type": "leaderboard", "reply": "…", "data": []})

    assert "### Conversation History: (none)" in blocks[0]


def test_audit_failure_never_breaks_the_request(on, monkeypatch):
    """Fail-soft is the whole contract: diagnostics must not be able to
    take the chat endpoint down."""
    monkeypatch.setattr(audit, "_format", lambda entry: 1 / 0)

    with audit.audit_query("still fine"):
        audit.record_response({"type": "text", "reply": "ok"})
