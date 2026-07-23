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
