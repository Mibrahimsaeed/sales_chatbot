"""Step 1 — keyword matching is token-aware (audit findings F1 and F9).

Every keyword table in the NLU layer was tested with `keyword in text`,
plain substring containment. That matches inside unrelated words, and
because the two worst collisions land on high-weight signals, the query
was not merely mis-scored — it was routed somewhere else entirely and
answered confidently.

  F1  "late" is inside calcuLATEd, reLATEd, transLATE, escaLATE.
      attendance_filter carries W_SPECIFIC_CONSTRAINT (score 0.98) and
      is in _RULE_BASED_ACTIONS, so it wins AND returns before the
      semantic parser ever runs. "How is the answered calls %
      calculated?" was answered with a list of late advisors.

  F9  "least" is inside "at least", "most" is inside "almost". Both are
      ranking/direction words, so a threshold phrase silently reversed
      the sort order and manufactured ranking evidence that wasn't there.

The rule now is one shared matcher (app/llm/token_match.py) used by every
table, so a new keyword cannot reintroduce the class.
"""

import pytest

from app.database.models import Advisor
from app.llm import entity_extractor, intent_catalog as cat, token_match
from app.llm.entity_extractor import ATTENDANCE_STATUS_KEYWORDS, extract_entities
from app.llm.preprocessing import normalize
from app.llm.query_planner import _sort_signal, build_query_plan, score_intents


# ---------------------------------------------------------------------
# The matcher itself
# ---------------------------------------------------------------------

@pytest.mark.parametrize("phrase,text", [
    ("late", "who was late today"),
    ("late", "late?"),
    ("late", "(late)"),
    ("late", "LATE"),
    ("not marked", "who is not marked"),
    ("not marked", "anyone  not   marked today"),   # collapsed whitespace
    ("at least", "at least 80%"),
    ("unit-head", "the unit-head of Blue Area"),
])
def test_a_whole_token_still_matches(phrase, text):
    assert token_match.contains(text, phrase)


@pytest.mark.parametrize("phrase,text", [
    ("late", "how is this calculated"),
    ("late", "related teams"),
    ("late", "please translate this"),
    ("late", "escalate to the manager"),
    ("late", "a belated response"),
    ("present", "the representative"),
    ("present", "a presentation"),
    ("most", "almost 80%"),
    ("most", "mostly on time"),
    ("least", "leastwise"),
    ("top", "stopped"),
    ("top", "laptop"),
    ("top", "the topic"),
    ("best", "bestseller"),
])
def test_a_substring_inside_another_word_does_not_match(phrase, text):
    assert not token_match.contains(text, phrase)


def test_contains_any_and_first_match_agree():
    phrases = ["not marked", "late", "present"]
    assert token_match.contains_any("who was late", phrases)
    assert token_match.first_match("who was late", phrases) == "late"
    assert token_match.first_match("how is this calculated", phrases) is None


def test_first_match_preserves_caller_order():
    """ATTENDANCE_STATUS_KEYWORDS relies on iteration order: "not marked"
    is checked before "late" so "not marked late arrivals" classifies as
    Not Marked, exactly as the original `for ... break` loop did."""
    assert token_match.first_match("not marked", ["not marked", "late"]) == "not marked"
    assert token_match.first_match("not marked", ["late", "not marked"]) == "not marked"


# ---------------------------------------------------------------------
# F1 — the reported query
# ---------------------------------------------------------------------

def test_calculated_does_not_extract_an_attendance_status(db_session):
    """The audit's headline case. 'calculated' contains 'late'."""
    entities = extract_entities(normalize("How is answered calls percentage calculated?"), db_session)

    assert "attendance_status" not in entities
    assert entities.get("attendance_status") != "Late"


def test_calculated_no_longer_routes_to_the_attendance_filter(db_session):
    """The consequence, not just the entity: attendance_filter scored
    0.98 and short-circuited the whole pipeline before the semantic
    parser could see the query."""
    text = normalize("How is answered calls percentage calculated?")
    entities = extract_entities(text, db_session)
    plan = build_query_plan(text, entities)

    assert plan.action != "attendance_filter"
    _ctx, candidates = score_intents(text, entities)
    assert "attendance_filter" not in [c.intent for c in candidates]


@pytest.mark.parametrize("text", [
    "how is this calculated",
    "show me related teams",
    "escalate to the manager",
    "please translate this reply",
    "who is the representative",
])
def test_words_merely_containing_a_status_word_are_not_statuses(db_session, text):
    entities = extract_entities(normalize(text), db_session)
    assert "attendance_status" not in entities, text


@pytest.mark.parametrize("text,expected", [
    ("who was late today", "Late"),
    ("who is absent", "Absent"),
    ("who is present today", "Present"),
    ("who is not marked", "Not Marked"),
    ("late arrivals in Blue Area", "Late"),
])
def test_genuine_attendance_phrasings_still_work(db_session, text, expected):
    """Preserving supported phrases is half the requirement — a fix that
    also stops recognising 'who was late today' is not a fix."""
    entities = extract_entities(normalize(text), db_session)
    assert entities.get("attendance_status") == expected


def test_every_status_keyword_is_still_reachable(db_session):
    for keyword, status in ATTENDANCE_STATUS_KEYWORDS.items():
        entities = extract_entities(normalize(f"who was {keyword} today"), db_session)
        assert entities.get("attendance_status") == status, keyword


# ---------------------------------------------------------------------
# F9 — ranking words inside other words
# ---------------------------------------------------------------------

def test_at_least_keeps_its_threshold_and_does_not_reverse_the_sort(db_session):
    """"at least" contains "least", which is an ASCENDING_ABSOLUTE word.
    The threshold was extracted correctly and then the ranking was
    flipped, so the answer led with the LOWEST achievers above 80%."""
    text = normalize("Show advisors with at least 80% achievement")
    entities = extract_entities(text, db_session)

    assert entities["thresholds"] == [{"operator": ">=", "value": 80.0}]

    plan = build_query_plan(text, entities)
    assert plan.ascending is not True, "'at least' must not force an ascending sort"


def test_at_least_is_not_ranking_evidence(db_session):
    """It also manufactured `ranking_strong`, which is worth 0.25 plus a
    0.25 combo bonus — enough to move the query to a different intent."""
    text = normalize("Show advisors with at least 80% achievement")
    entities = extract_entities(text, db_session)
    ctx, _candidates = score_intents(text, entities)

    assert not ctx.has_ranking_strong


def test_most_is_detected_as_a_ranking_direction():
    """The signal that must keep working."""
    assert _sort_signal("which team has the most revenue", "mtd_cleared") is False


def test_most_is_detected_as_ranking_evidence(db_session):
    text = normalize("Which team has the most revenue?")
    entities = extract_entities(text, db_session)
    ctx, _candidates = score_intents(text, entities)

    assert ctx.has_ranking_strong
    assert build_query_plan(text, entities).ascending is False


def test_almost_is_not_detected_as_most(db_session):
    """"almost 80%" is a threshold-ish phrase, not a request for the
    maximum. It used to set descending AND claim ranking evidence."""
    text = normalize("Which team has almost 80% achievement?")
    entities = extract_entities(text, db_session)
    ctx, _candidates = score_intents(text, entities)

    assert not ctx.has_ranking_strong
    assert _sort_signal(text.lower(), "achievement_pct") is None


@pytest.mark.parametrize("text,metric,expected", [
    ("lowest revenue", "mtd_cleared", True),
    ("highest revenue", "mtd_cleared", False),
    ("least overdue", "overdue", True),
    ("most overdue", "overdue", False),
    # "worst" is a QUALITY word resolved against the metric's polarity.
    ("worst revenue", "mtd_cleared", True),
    ("worst overdue", "overdue", False),
    # No direction word at all — the metric's polarity decides downstream.
    ("revenue by team", "mtd_cleared", None),
    ("almost 80 percent", "achievement_pct", None),
    ("at least 80 percent", "achievement_pct", None),
])
def test_sort_direction_vocabulary_is_intact(text, metric, expected):
    assert _sort_signal(text, metric) is expected


@pytest.mark.parametrize("word", cat.RANKING_STRONG)
def test_every_strong_ranking_word_is_still_recognised(db_session, word):
    text = normalize(f"{word} advisors by revenue")
    _ctx, _c = score_intents(text, extract_entities(text, db_session))
    ctx, _ = score_intents(text, extract_entities(text, db_session))
    assert ctx.has_ranking_strong, word


# ---------------------------------------------------------------------
# Level keywords — same table, same class of bug
# ---------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("top 5 teams by revenue", "team"),
    ("top 5 advisors by revenue", "advisor"),
    ("top 5 companies by revenue", "company"),
    ("top 5 unit heads by revenue", "unit_head"),
    ("top 5 zonal heads by revenue", "zonal_head"),
    ("top 5 bcms by revenue", "bcm"),
    ("top 5 regions by revenue", "region"),
    ("top 5 business centers by revenue", "office"),
])
def test_level_detection_is_intact(text, expected):
    assert cat.detect_level(text) == expected


def test_level_detection_ignores_a_substring_inside_another_word():
    assert cat.detect_level("teamwork across the org") is None
    assert cat.detect_level("regionally consistent results") is None


# ---------------------------------------------------------------------
# The gazetteer — same class, on entity names
# ---------------------------------------------------------------------

@pytest.fixture()
def named(db_session):
    """A deliberately short team name and a punctuated region — the two
    shapes that pull in opposite directions. "GRO" must stop matching
    inside "grocery"; "North/KPK" must keep matching despite the slash,
    which a blanket \\b wrap would break."""
    db_session.add(Advisor(wid=1, name="Ali Raza", team="GRO", company="Graana",
                           region="North/KPK", in_master_sheet=True))
    db_session.add(Advisor(wid=2, name="Sara Khan", team="Blue Area", company="IMARAT",
                           in_master_sheet=True))
    db_session.commit()
    # The gazetteer cache is process-global with a TTL; expire it so this
    # fixture's rows are the ones matched against (same pattern as
    # test_entity_extractor.py).
    entity_extractor._cache["loaded_at"] = 0
    return db_session


def test_a_short_team_name_no_longer_matches_inside_a_word(named):
    assert extract_entities("how is GRO doing", named).get("teams") == ["GRO"]
    assert extract_entities("the grocery budget", named).get("teams") is None


def test_a_name_containing_punctuation_still_grounds(named):
    assert extract_entities("advisors in North/KPK", named).get("regions") == ["North/KPK"]


@pytest.mark.parametrize("text", ["revenue for Blue Area", "Blue Area's revenue",
                                  "Blue Area, please", "(Blue Area)"])
def test_a_name_still_grounds_next_to_punctuation(named, text):
    assert extract_entities(text, named).get("teams") == ["Blue Area"]


# ---------------------------------------------------------------------
# The guard: this must not come back for a keyword added later
# ---------------------------------------------------------------------

def _tables():
    """Every keyword table now routed through token_match, with the
    callable that reads it. Listed here so a table added later without a
    matching entry is a visible omission rather than a silent one."""
    return [
        ("ATTENDANCE_STATUS_KEYWORDS", list(ATTENDANCE_STATUS_KEYWORDS)),
        ("RANKING_STRONG", list(cat.RANKING_STRONG)),
        ("RANKING_WEAK", list(cat.RANKING_WEAK)),
        ("ASCENDING_ABSOLUTE", list(cat.ASCENDING_ABSOLUTE)),
        ("DESCENDING_ABSOLUTE", list(cat.DESCENDING_ABSOLUTE)),
        ("WORST_RELATIVE", list(cat.WORST_RELATIVE)),
        ("FLAT_KEYWORDS", list(cat.FLAT_KEYWORDS)),
        *[(f"LEVEL_KEYWORDS[{lvl}]", list(kws)) for lvl, kws in cat.LEVEL_KEYWORDS.items()],
    ]


@pytest.mark.parametrize("table,keywords", _tables())
def test_no_keyword_in_any_table_matches_inside_a_longer_word(table, keywords):
    """A property over the tables themselves, not a list of the
    collisions we happened to find. Bury each keyword inside a longer
    word; none may be detected. A keyword added tomorrow is covered the
    moment it is declared.
    """
    for keyword in keywords:
        buried = f"zq{keyword}zq"
        assert not token_match.contains(buried, keyword), f"{table}: {keyword!r} in {buried!r}"


@pytest.mark.parametrize("table,keywords", _tables())
def test_every_keyword_in_every_table_still_matches_itself(table, keywords):
    """The other half — the fix must not silence any supported phrase."""
    for keyword in keywords:
        assert token_match.contains(f"please show {keyword} now", keyword), f"{table}: {keyword!r}"


def test_comparator_phrases_shadow_the_direction_words_inside_them():
    """"at least"/"at most" are comparators that contain a direction
    word. The shadowing is DERIVED from the comparator registry, so a
    phrase declared later is covered without touching the planner."""
    from app.llm import comparators
    from app.llm.query_planner import _without_comparators

    assert "at least" in comparators.phrases()
    assert "at most" in comparators.phrases()

    for phrase in comparators.phrases():
        masked = _without_comparators(f"advisors with {phrase} 80 percent")
        assert not token_match.contains_any(masked, cat.ASCENDING_ABSOLUTE), phrase
        assert not token_match.contains_any(masked, cat.DESCENDING_ABSOLUTE), phrase


def test_masking_preserves_the_rest_of_the_query():
    from app.llm.query_planner import _without_comparators

    masked = _without_comparators("top 5 advisors with at least 80% revenue")
    assert token_match.contains(masked, "top")
    assert "80%" in masked and "revenue" in masked
