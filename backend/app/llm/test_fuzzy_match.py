from app.llm.fuzzy_match import best_match, find_in_text, STRONG_FLOOR
from app.llm.fallback_reasoning import fuzzy_resolve_metric


TEAMS = ["Blue Area", "Downtown", "DHA Phase 5", "Gulberg Greens"]
COMPANIES = ["IMARAT", "Graana", "Agency21"]
ADVISORS = ["Waqar Haider", "Ali Raza", "Sana Khan"]


def test_exact_match_short_circuits_to_full_confidence():
    assert best_match("graana", COMPANIES, kind="company") == ("Graana", 1.0)


def test_company_typo():
    result = best_match("Grana", COMPANIES, kind="company")
    assert result is not None
    assert result[0] == "Graana"
    assert result[1] >= STRONG_FLOOR


def test_advisor_swapped_word_order():
    result = best_match("Haider Waqar", ADVISORS, kind="advisor")
    assert result is not None
    assert result[0] == "Waqar Haider"
    assert result[1] >= 0.9


def test_advisor_first_name_only_matches_but_discounted():
    result = best_match("Waqar", ADVISORS, kind="advisor")
    assert result is not None
    assert result[0] == "Waqar Haider"
    assert result[1] < 1.0


def test_no_match_below_floor_returns_none():
    assert best_match("zzzzqqq", TEAMS, kind="team") is None


def test_find_in_text_locates_typod_company_in_sentence():
    hits = find_in_text("show only grana advisors", COMPANIES, kind="company")
    assert hits and hits[0][0] == "Graana"


def test_find_in_text_multiword_team_in_sentence():
    hits = find_in_text("compare blue area with downtown", TEAMS, kind="team")
    names = [h[0] for h in hits]
    assert "Blue Area" in names and "Downtown" in names


def test_find_in_text_empty_text():
    assert find_in_text("", TEAMS) == []


def test_fuzzy_resolve_metric_refuses_a_typo():
    """P0 SAFETY: the approximate tier is off.

    This widened a misspelling onto the intended measure by edit
    distance. The same scan is what let a measure be chosen because it
    RESEMBLED part of a sentence, and a guessed measure is
    indistinguishable from a correct one in the reply — so the deliberate
    trade is to refuse and let the LLM parser, which reads the whole
    sentence, recover the intent instead.
    """
    assert fuzzy_resolve_metric("atendance rate above 90") is None


def test_fuzzy_resolve_metric_substring_still_short_circuits():
    assert fuzzy_resolve_metric("who has the highest sales") == "mtd_cleared"


def test_fuzzy_resolve_metric_gibberish_returns_none():
    assert fuzzy_resolve_metric("xyzzy plugh") is None
