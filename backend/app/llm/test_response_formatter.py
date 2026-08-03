"""Bug fix: format_advisor_reply/format_team_reply/format_company_reply
used to only ever mention MTD figures — the underlying data (advisor_
profile view, team/company service rollups) already had YTD available,
it just never made it into the reply text, so asking about YTD
performance silently got an MTD-only answer."""

from app.llm.response_formatter import (
    format_advisor_reply, format_breakdown_reply, format_company_reply,
    format_flat_breakdown_reply, format_person_disambiguation_reply, format_team_reply,
)


def test_advisor_reply_includes_ytd_when_present():
    advisor = {
        "name": "Haseeb Arslan",
        "mtd_new_connect": 0, "mtd_followup_connect": 0,
        "mtd_cleared": 13_070_156, "mtd_target": 0,
        "ytd_cleared": 147_092_660, "ytd_target": 100_000_000,
        "overdue": 0,
    }
    reply = format_advisor_reply(advisor)
    assert "147,092,660" in reply
    assert "Year to date" in reply


def test_advisor_reply_omits_ytd_note_when_both_zero():
    advisor = {"name": "A", "mtd_cleared": 0, "mtd_target": 0, "ytd_cleared": 0, "ytd_target": 0}
    reply = format_advisor_reply(advisor)
    assert "Year to date" not in reply


def test_team_reply_includes_ytd_when_present():
    summary = {
        "team": "Alpha", "advisors": 5, "connects": 100,
        "target": 1000, "achieved": 500,
        "ytd_cleared": 50_000, "ytd_target": 40_000,
    }
    reply = format_team_reply(summary)
    assert "50,000" in reply
    assert "Year to date" in reply


def test_company_reply_includes_ytd_when_present():
    summary = {
        "company": "Graana", "advisors": 10, "connects": 200,
        "mtd_cleared": 1000, "mtd_target": 2000,
        "ytd_cleared": 900_000, "ytd_target": 800_000,
    }
    reply = format_company_reply(summary)
    assert "900,000" in reply
    assert "Year to date" in reply


def test_company_reply_omits_ytd_note_when_absent():
    summary = {"company": "Graana", "advisors": 1, "connects": 5, "mtd_cleared": 100, "mtd_target": 200}
    reply = format_company_reply(summary)
    assert "Year to date" not in reply


def _breakdown(**overrides) -> dict:
    base = dict(
        level="unit_head", level_label="Unit Head", value="Zeeshan Tariq",
        advisors=2, connects=30, overdue=0, pipeline=0,
        mtd_target=3000, mtd_cleared=1500, ytd_target=0, ytd_cleared=0,
        teams=[
            {"team": "Blue Area", "advisor_count": 1, "advisors": [
                {"wid": 1, "name": "Advisor One", "connects": 10, "mtd_cleared": 500, "mtd_target": 1000},
            ]},
            {"team": "Downtown", "advisor_count": 1, "advisors": [
                {"wid": 2, "name": "Advisor Two", "connects": 20, "mtd_cleared": 1000, "mtd_target": 2000},
            ]},
        ],
    )
    base.update(overrides)
    return base


def test_breakdown_reply_nests_advisors_under_their_team():
    reply = format_breakdown_reply(_breakdown())
    assert "Unit Head Zeeshan Tariq" in reply
    assert "2 advisors across 2 team(s)" in reply
    assert "Blue Area" in reply
    assert "Downtown" in reply
    assert "Advisor One" in reply
    assert "Advisor Two" in reply
    # nesting: an advisor line must come after its own team's header
    assert reply.index("Blue Area") < reply.index("Advisor One") < reply.index("Downtown")


def test_breakdown_reply_includes_ytd_when_present():
    reply = format_breakdown_reply(_breakdown(ytd_cleared=8000, ytd_target=12000))
    assert "Year to date" in reply
    assert "8,000" in reply


def test_breakdown_reply_omits_ytd_note_when_absent():
    reply = format_breakdown_reply(_breakdown())
    assert "Year to date" not in reply


# ---- Phase 2: flat opt-in ----

def _flat(**overrides) -> dict:
    base = dict(
        level="unit_head", level_label="Unit Head", value="Zeeshan Tariq",
        advisors=2, connects=30, overdue=0, pipeline=0,
        mtd_target=3000, mtd_cleared=1500, ytd_target=0, ytd_cleared=0,
        advisor_list=[
            {"wid": 1, "name": "Advisor One", "team": "Blue Area", "connects": 10, "mtd_cleared": 500, "mtd_target": 1000},
            {"wid": 2, "name": "Advisor Two", "team": "Downtown", "connects": 20, "mtd_cleared": 1000, "mtd_target": 2000},
        ],
    )
    base.update(overrides)
    return base


def test_flat_breakdown_reply_has_no_team_grouping():
    reply = format_flat_breakdown_reply(_flat())
    assert "Unit Head Zeeshan Tariq" in reply
    assert "2 advisors" in reply
    assert "across" not in reply   # no "across N team(s)" wording — that's the nested reply's phrasing
    assert "Advisor One" in reply
    assert "Advisor Two" in reply
    assert "(Blue Area)" in reply
    assert "(Downtown)" in reply


def test_flat_breakdown_reply_includes_ytd_when_present():
    reply = format_flat_breakdown_reply(_flat(ytd_cleared=8000, ytd_target=12000))
    assert "Year to date" in reply
    assert "8,000" in reply


# ---- Phase 2: person disambiguation ----

class _Cand:
    """Minimal stand-in for advisor_resolver.AdvisorIdentity."""
    def __init__(self, wid, name, team=None, company=None):
        self.wid, self.name, self.team, self.company = wid, name, team, company


def test_disambiguation_uses_the_numbered_format():
    reply = format_person_disambiguation_reply("Yasir Ali", [
        _Cand(1, "Yasir Ali", "North/KPK Region"),
        _Cand(2, "Yasir Ali", "Team ABC"),
    ])
    assert reply.startswith("I found multiple advisors named Yasir Ali.")
    assert "1. Yasir Ali — North/KPK Region" in reply
    assert "2. Yasir Ali — Team ABC" in reply
    assert reply.rstrip().endswith("Which one did you mean?")


def test_team_alone_is_used_when_it_already_distinguishes():
    reply = format_person_disambiguation_reply("Yasir Ali", [
        _Cand(1, "Yasir Ali", "North/KPK", "Agency21"),
        _Cand(2, "Yasir Ali", "Downtown", "IMARAT"),
    ])
    assert "ID 1" not in reply and "ID 2" not in reply
    assert "Agency21" not in reply   # company adds nothing here


def test_wid_is_added_only_for_candidates_a_team_cannot_separate():
    """Production has 8 people named "Yasir Ali", 6 of them in the same
    team — listing six identical lines asks a question the user cannot
    answer, which is no better than the silent guess this replaced."""
    reply = format_person_disambiguation_reply("Yasir Ali", [
        _Cand(10, "Yasir Ali", "North/KPK", "Agency21"),
        _Cand(11, "Yasir Ali", "North/KPK", "Agency21"),
        _Cand(12, "Yasir Ali", "Team Rashid Majeed"),
    ])
    assert "ID 10" in reply and "ID 11" in reply
    # the one with a unique team stays clean
    assert "3. Yasir Ali — Team Rashid Majeed" in reply


def test_different_names_are_not_described_as_the_same_name():
    """A near-tie between DIFFERENT names ("Ahmed Ali" vs "Ali Ahmed")
    must not claim they are all "people named Ahmed Ali"."""
    reply = format_person_disambiguation_reply("Ahmed Ali", [
        _Cand(1, "Ahmed Ali", "AMD"),
        _Cand(2, "Ali Ahmed", "Blue Area"),
    ])
    assert "matches more than one advisor" in reply
    assert "people named" not in reply


def test_missing_team_still_yields_a_usable_line():
    reply = format_person_disambiguation_reply("X Y", [
        _Cand(1, "X Y", None, "Graana"),
        _Cand(2, "X Y", None, "IMARAT"),
    ])
    assert "Graana" in reply and "IMARAT" in reply
