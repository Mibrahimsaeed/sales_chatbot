"""Bug fix: format_advisor_reply/format_team_reply/format_company_reply
used to only ever mention MTD figures — the underlying data (advisor_
profile view, team/company service rollups) already had YTD available,
it just never made it into the reply text, so asking about YTD
performance silently got an MTD-only answer."""

from app.llm.response_formatter import format_advisor_reply, format_company_reply, format_team_reply


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
