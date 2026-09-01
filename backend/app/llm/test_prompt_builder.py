from app.llm.prompt_builder import build_ir_prompt


def _prompt(prior_ir_json=None):
    return build_ir_prompt(
        "top 5 advisors by revenue",
        known_teams=["Blue Area", "Downtown"],
        known_companies=["Graana", "IMARAT"],
        grounded_entities={"limit": 5},
        prior_ir_json=prior_ir_json,
    )


def test_prompt_contains_all_grounding_sections():
    prompt = _prompt()
    assert "Known teams: Blue Area, Downtown" in prompt
    assert "Known companies: Graana, IMARAT" in prompt
    assert "Metric catalog" in prompt
    # full synonym lists, not the old first-3 truncation — "highest closer"
    # is synonym #7 of mtd_cleared
    assert "highest closer" in prompt
    assert "COMMON BUSINESS PHRASES" in prompt
    assert "Examples (follow these exactly" in prompt
    assert "Return ONLY a JSON object" in prompt
    assert '"top 5 advisors by revenue"' in prompt


def test_prompt_includes_prior_ir_only_when_present():
    assert "treat the new message as a semantic patch" not in _prompt()
    with_prior = _prompt(prior_ir_json='{"intent": "leaderboard"}')
    assert "treat the new message as a semantic patch" in with_prior
    assert '{"intent": "leaderboard"}' in with_prior
