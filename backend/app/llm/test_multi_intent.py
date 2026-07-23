from app.llm.multi_intent import split_subqueries


def test_semicolon_splits_into_two_segments():
    segments = split_subqueries("show top advisors by revenue; list attendance issues today")
    assert segments == ["show top advisors by revenue", "list attendance issues today"]


def test_also_keyword_splits():
    segments = split_subqueries("show top advisors by revenue, also list attendance issues today")
    assert segments is not None
    assert len(segments) == 2
    assert "attendance issues today" in segments[1]


def test_numbered_list_splits():
    segments = split_subqueries("1) top advisors by revenue 2) attendance issues this month")
    assert segments == ["top advisors by revenue", "attendance issues this month"]


def test_two_question_marks_split():
    segments = split_subqueries("who leads on revenue? who has the most connects?")
    assert segments is not None
    assert len(segments) == 2


def test_newline_splits():
    segments = split_subqueries("top advisors by revenue\nattendance issues this month")
    assert segments == ["top advisors by revenue", "attendance issues this month"]


# ---- must NOT trigger on ordinary compound-filter queries (these stay
# ONE multi-filter query today via semantic_parser's _COMPOUND_HINTS) ----

def test_bare_and_does_not_split():
    assert split_subqueries(
        "Show Graana advisors with attendance at least 90 and achievement above 80, sorted by meetings"
    ) is None


def test_bare_but_does_not_split():
    assert split_subqueries(
        "sort advisors by ytd revenue but only those with mtd revenue under 400"
    ) is None


def test_plain_single_query_does_not_split():
    assert split_subqueries("top 5 advisors by revenue") is None


def test_single_question_mark_does_not_split():
    assert split_subqueries("who leads on revenue?") is None


def test_trivial_fragments_are_rejected():
    # each side must have at least 3 words to count as a real sub-query
    assert split_subqueries("revenue; ok") is None


def test_sentence_ending_number_period_does_not_look_like_a_list_marker():
    # "achievement below 60." must not be mistaken for a "1." style list
    # marker just because it's a mid-sentence digit followed by a period
    assert split_subqueries("advisors with achievement below 60. sorted by overdue") is None
