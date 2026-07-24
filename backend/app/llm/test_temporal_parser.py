from app.llm import temporal_parser
from app.llm.temporal_parser import parse_period


def test_this_month_is_equivalent_to_mtd():
    m = parse_period("revenue this month")
    assert m.kind == "equivalent"
    assert m.period == "MTD"


def test_ytd_keyword():
    m = parse_period("top advisors ytd")
    assert m.kind == "equivalent"
    assert m.period == "YTD"


def test_quarter_maps_to_3m():
    m = parse_period("revenue this quarter")
    assert m.kind == "equivalent"
    assert m.period == "3M"


def test_last_month_is_unsupported_not_silently_mtd():
    m = parse_period("revenue last month")
    assert m.kind == "unsupported"
    assert m.period is None


def test_yesterday_is_unsupported():
    m = parse_period("who was top performer yesterday")
    assert m.kind == "unsupported"


def test_past_n_days_is_unsupported():
    m = parse_period("revenue in the past 30 days")
    assert m.kind == "unsupported"


def test_this_week_is_unsupported():
    m = parse_period("top advisors this week")
    assert m.kind == "unsupported"


def test_no_temporal_expression_returns_none():
    assert parse_period("top advisors by revenue") is None


def test_bare_month_word_no_longer_silently_maps_to_mtd():
    # a lone "month" without "this"/"last" is not confidently MTD or a
    # rejectable expression either — no match at all is the honest answer
    assert parse_period("monthly leaderboard") is None


# ---- Part 12: semantic retrieval fallback (only when both pattern lists miss) ----

def test_semantic_fallback_disabled_by_default_in_tests():
    # conftest's autouse fixture forces entity_linking_enabled=False
    assert parse_period("since the start of the month") is None


def test_semantic_fallback_classifies_an_unusual_mtd_phrasing(monkeypatch):
    monkeypatch.setattr(temporal_parser.entity_linker.settings, "entity_linking_enabled", True)
    monkeypatch.setattr(
        temporal_parser.entity_linker, "semantic_classify",
        lambda text, exemplar_type, **kw: [{"value": "MTD", "score": 0.9}] if exemplar_type == "temporal" else [],
    )
    m = parse_period("since the start of the month")
    assert m.kind == "equivalent"
    assert m.period == "MTD"
    assert m.confidence == 0.9


def test_semantic_fallback_never_overrides_unsupported_verdict(monkeypatch):
    monkeypatch.setattr(temporal_parser.entity_linker.settings, "entity_linking_enabled", True)
    monkeypatch.setattr(
        temporal_parser.entity_linker, "semantic_classify",
        lambda text, exemplar_type, **kw: [{"value": "3M", "score": 0.99}],
    )
    # "last month" already hits the deterministic unsupported pattern — the
    # mocked semantic step must never even be consulted
    m = parse_period("revenue last month")
    assert m.kind == "unsupported"


def test_semantic_fallback_never_overrides_equivalent_verdict(monkeypatch):
    monkeypatch.setattr(temporal_parser.entity_linker.settings, "entity_linking_enabled", True)
    monkeypatch.setattr(
        temporal_parser.entity_linker, "semantic_classify",
        lambda text, exemplar_type, **kw: [{"value": "YTD", "score": 0.99}],
    )
    m = parse_period("revenue this month")
    assert m.kind == "equivalent"
    assert m.period == "MTD"


def test_semantic_fallback_skipped_without_a_temporal_hint(monkeypatch):
    monkeypatch.setattr(temporal_parser.entity_linker.settings, "entity_linking_enabled", True)
    calls = []
    monkeypatch.setattr(
        temporal_parser.entity_linker, "semantic_classify",
        lambda text, exemplar_type, **kw: calls.append(text) or [],
    )
    # no time-adjacent vocabulary at all — the hint regex must skip the
    # embedding call entirely
    assert parse_period("top 5 advisors by revenue") is None
    assert calls == []
