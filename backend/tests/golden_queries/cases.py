"""
The golden corpus: realistic business questions and the understanding
each must produce.

HOW TO READ A CASE. `expect` lists only the fields that case is about. A
field left out is not asserted, which keeps a case focused on the thing
it is protecting — a leaderboard case should fail when the metric changes,
not when an unrelated default does. Every case must pin `intent` plus at
least one more field; a test enforces that.

HOW TO ADD ONE. Append to the right category with the understanding you
believe is correct. If it fails, decide which side is wrong before
changing anything — that decision is the whole value of this file. Cases
recorded from observed output without that judgement would just enshrine
current bugs as the contract.

Where the current understanding is WRONG and the fix is out of scope, the
case is marked `known_gap` with an explanation. Those are asserted to
still behave as described, so the gap cannot quietly get worse, and they
are a to-do list rather than a silence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Case:
    query: str
    expect: dict[str, Any]
    # Set when the recorded understanding is NOT what it should be. The
    # case still runs — it pins the current behaviour so the gap cannot
    # widen unnoticed — but the text says what the right answer would be.
    known_gap: Optional[str] = None
    # The audit finding this case is the permanent regression test for
    # ("F1", "F4", ...). A structural test asserts every resolved finding
    # has at least one, so a fixed bug cannot lose its guard when the
    # file is reorganised.
    finding: Optional[str] = None

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.expect)


# =====================================================================
# 1. LEADERBOARDS
# =====================================================================

LEADERBOARDS = [
    Case("top 5 advisors by revenue",
         {"intent": "leaderboard", "metric": "mtd_cleared", "level": "advisor",
          "limit": 5, "ranking": "desc", "period": "MTD"}),
    Case("top 10 advisors by connects",
         {"intent": "leaderboard", "metric": "total_connects", "level": "advisor", "limit": 10}),
    Case("top 3 teams by achievement",
         {"intent": "leaderboard", "metric": "achievement_pct", "level": "team", "limit": 3}),
    Case("who are the best performers",
         {"intent": "leaderboard", "metric": "achievement_pct"}),
    Case("show me the top advisors by pipeline value",
         {"intent": "leaderboard", "metric": "pipeline_value", "level": "advisor"}),
    Case("top 5 unit heads by connects",
         {"intent": "leaderboard", "metric": "total_connects", "level": "unit_head", "limit": 5}),
    Case("top 3 zonal heads by revenue",
         {"intent": "leaderboard", "metric": "mtd_cleared", "level": "zonal_head", "limit": 3}),
    Case("top BCMs by revenue",
         {"intent": "leaderboard", "metric": "mtd_cleared", "level": "bcm"}),
    Case("top companies by revenue",
         {"intent": "leaderboard", "metric": "mtd_cleared", "level": "company"}),
    Case("top business centers by connects",
         {"intent": "leaderboard", "metric": "total_connects", "level": "office"}),
    Case("top regions by portfolio value",
         {"intent": "leaderboard", "metric": "portfolio_value", "level": "region"}),
    Case("top 5 advisors",
         {"intent": "leaderboard", "metric": "mtd_cleared", "level": "advisor", "limit": 5}),
    Case("who has the most connects",
         {"intent": "leaderboard", "metric": "total_connects", "ranking": "desc"}),
    Case("who has the most overdue",
         {"intent": "leaderboard", "metric": "overdue", "ranking": "desc"}),
    Case("who has the least overdue",
         {"intent": "leaderboard", "metric": "overdue", "ranking": "asc"}),
    Case("which advisor has the lowest achievement",
         {"intent": "leaderboard", "metric": "achievement_pct", "ranking": "asc"}),
    Case("who is the worst advisor",
         {"intent": "leaderboard", "metric": "mtd_cleared", "ranking": "asc"}),
    Case("worst advisors by overdue",
         {"intent": "leaderboard", "metric": "overdue", "ranking": "desc"}),
    Case("rank teams by meetings",
         {"intent": "leaderboard", "metric": "total_meetings", "level": "team"}),
    Case("top advisors by client registrations",
         {"intent": "leaderboard", "metric": "client_registrations"}),
    Case("top teams by conversion rate",
         {"intent": "leaderboard", "metric": "meeting_to_conversion_rate", "level": "team"}),
    Case("top advisors by connect to cr",
         {"intent": "leaderboard", "metric": "connect_to_cr_rate"}),
    Case("top advisors by answered calls",
         {"intent": "leaderboard", "metric": "answered_calls"}),
    Case("top advisors by bookings",
         {"intent": "leaderboard", "metric": "bookings"}),
    Case("top advisors by portfolio",
         {"intent": "leaderboard", "metric": "portfolio_value"}),
    Case("top 5 advisors in Blue Area by revenue",
         {"intent": "leaderboard", "metric": "mtd_cleared", "limit": 5,
          "filters": (("team", "=", "Blue Area"),)}),
    Case("top advisors in Graana by connects",
         {"intent": "leaderboard", "metric": "total_connects",
          "filters": (("company", "=", "Graana"),)}),
    Case("top 5 advisors by revnue",
         {"intent": "leaderboard", "metric": "mtd_cleared", "limit": 5}),
    Case("top advisors by widget velocity",
         {"intent": "clarify_metric", "metric": None}),
    # RETIRED REFUSAL — working_days.py made the rate computable.
    Case("which team has the highest CR %",
         {"intent": "leaderboard", "metric": "cr_rate"}),
]


# =====================================================================
# 2. COMPARISONS
# =====================================================================

COMPARISONS = [
    Case("compare Blue Area and Downtown",
         {"intent": "comparison", "level": "team", "entity": "Blue Area"}),
    Case("compare Graana and IMARAT",
         {"intent": "comparison", "level": "company", "entity": "Graana"}),
    Case("compare Blue Area and Downtown by revenue",
         {"intent": "comparison", "metric": "mtd_cleared", "level": "team"}),
    Case("Blue Area vs Downtown",
         {"intent": "comparison", "level": "team"}),
    Case("Blue Area versus Downtown on achievement",
         {"intent": "comparison", "metric": "achievement_pct", "level": "team"}),
    Case("what is the difference between Blue Area and Downtown",
         {"intent": "comparison", "level": "team"}),
    Case("how does Blue Area compare to Downtown",
         {"intent": "comparison", "level": "team"}),
    Case("is Blue Area better than Downtown",
         {"intent": "comparison", "level": "team"}),
    Case("compare Tariq Mehmood and Sadia Rehman",
         {"intent": "comparison", "level": "unit_head"}),
    Case("compare Usman Ghani and Rabia Anjum on connects",
         {"intent": "comparison", "metric": "total_connects", "level": "bcm"}),
    Case("compare Blue Area with Graana",
         {"intent": "comparison", "metric": None}),
    Case("compare Blue Area and Atlantis",
         {"intent": "comparison_incomplete", "entity": "Blue Area"}),
    Case("which company is doing better, Graana or IMARAT",
         {"intent": "comparison", "level": "company"}),
    Case("which team is doing better, Blue Area or Downtown",
         {"intent": "comparison", "level": "team"}),
    Case("compare Blue Area and Downtown on overdue",
         {"intent": "comparison", "metric": "overdue", "level": "team"}),

    # ---- Phase 5.4: every level compares, and "which X is better" is a
    # comparison rather than a one-sided summary at any of them.
    Case("which bcm is doing better, Usman Ghani or Rabia Anjum",
         {"intent": "comparison", "level": "bcm"}),
    Case("which unit head performed better, Tariq Mehmood or Sadia Rehman",
         {"intent": "comparison", "level": "unit_head"}),
    Case("which zonal head is doing better, Fawad Hafeez or Adeel Aslam",
         {"intent": "comparison", "level": "zonal_head"}),
    Case("compare North/KPK and Central",
         {"intent": "comparison", "level": "region"}),
    Case("which region is performing better, Central or South",
         {"intent": "comparison", "level": "region"}),
    Case("compare Beverly Center and Gold Crest",
         {"intent": "comparison", "level": "office"}),
    Case("which office is doing better, Emporium or Gold Crest",
         {"intent": "comparison", "level": "office"}),
    Case("compare Usman Ghani and Bilal Qadir on revenue",
         {"intent": "comparison", "metric": "mtd_cleared", "level": "bcm"}),
    Case("compare Blue Area, Downtown and Gulberg",
         {"intent": "comparison", "level": "team"}),
    Case("which of the two teams is better, Blue Area or Downtown",
         {"intent": "comparison", "level": "team"}),
    # A SUPERLATIVE is a ranking, not a two-sided comparison. The
    # "which <noun> ... better" widening must not swallow these.
    Case("which team has the highest overdue count",
         {"intent": "leaderboard", "metric": "overdue", "level": "team"}),
    Case("which advisor has the most meetings",
         {"intent": "leaderboard", "metric": "total_meetings"}),
]


# =====================================================================
# 3. KPI QUESTIONS (one measure, or one entity's numbers)
# =====================================================================

KPI_QUESTIONS = [
    Case("connects of Shehryar Abbasi",
         {"intent": "advisor_metric", "metric": "total_connects", "entity": "Shehryar Abbasi"}),
    Case("what is Hina Malik's pipeline value",
         {"intent": "advisor_metric", "metric": "pipeline_value", "entity": "Hina Malik"}),
    Case("how many meetings does Nadia Sheikh have",
         {"intent": "advisor_metric", "metric": "total_meetings", "entity": "Nadia Sheikh"}),
    Case("Faisal Iqbal overdue",
         {"intent": "advisor_metric", "metric": "overdue", "entity": "Faisal Iqbal"}),
    Case("what is Yasir Ali's portfolio value",
         {"intent": "advisor_metric", "metric": "portfolio_value", "entity": "Yasir Ali"}),
    Case("tell me about Yasir Ali",
         {"intent": "advisor_profile", "entity": "Yasir Ali", "metric": None}),
    Case("who is Waqar Haider",
         {"intent": "advisor_profile", "entity": "Waqar Haider"}),
    Case("show me Omar Farooq's profile",
         {"intent": "advisor_profile", "entity": "Omar Farooq"}),
    Case("what is the performance of Sana Tariq",
         {"intent": "advisor_profile", "entity": "Sana Tariq"}),
    Case("how is Blue Area doing",
         {"intent": "summary", "level": "team", "entity": "Blue Area"}),
    Case("how is Graana doing",
         {"intent": "summary", "level": "company", "entity": "Graana"}),
    Case("Downtown summary",
         {"intent": "summary", "level": "team", "entity": "Downtown"}),
    # A CORRECT refusal, not a gap: the answered-call rate needs a
    # working-day calendar. Pinned so it cannot silently become the
    # underlying count again.
    # RETIRED REFUSALS. These three were pinned as clarifications for
    # want of a working-day calendar; working_days.py is that calendar.
    # Still pinned, and for the original reason: a RATE must never
    # silently become the COUNT inside it.
    # `intent` here is the IR's, and a group_metric PLAN compiles to a
    # leaderboard IR scoped to the one group (Phase 7) — one row, which
    # the response planner then renders as a metric value.
    Case("what is the answered calls percentage for Blue Area",
         {"intent": "leaderboard", "metric": "answered_calls_rate"}),
    Case("what is the CR rate for Downtown",
         {"intent": "leaderboard", "metric": "cr_rate"}),
    Case("what is the meeting rate for Blue Area",
         {"intent": "leaderboard", "metric": "meeting_rate"}),
    # RETIRED REFUSAL — the ETL now imports the "1 Unit" tab, so this
    # answers instead of explaining why it cannot.
    Case("what is the 1 unit ratio",
         {"intent": "leaderboard", "metric": "one_unit_ratio", "level": "team"}),
]


# =====================================================================
# 4. HIERARCHY QUESTIONS
# =====================================================================

HIERARCHY = [
    Case("who is Yasir Ali's unit head",
         {"intent": "reverse_hierarchy", "level": "unit_head", "entity": "Yasir Ali"}),
    Case("who is Yasir Ali's zonal head",
         {"intent": "reverse_hierarchy", "level": "zonal_head", "entity": "Yasir Ali"}),
    Case("who is Yasir Ali's BCM",
         {"intent": "reverse_hierarchy", "level": "bcm", "entity": "Yasir Ali"}),
    Case("who does Shehryar Abbasi report to",
         {"intent": "reverse_hierarchy", "entity": "Shehryar Abbasi"}),
    Case("who is Hina Malik's manager",
         {"intent": "reverse_hierarchy", "entity": "Hina Malik"}),
    Case("which unit head manages Omar Farooq",
         {"intent": "reverse_hierarchy", "level": "unit_head", "entity": "Omar Farooq"}),
    Case("which zonal head oversees Zainab Noor",
         {"intent": "reverse_hierarchy", "level": "zonal_head", "entity": "Zainab Noor"}),
    Case("which BCM leads Faisal Iqbal",
         {"intent": "reverse_hierarchy", "level": "bcm", "entity": "Faisal Iqbal"}),
    Case("all advisors in Blue Area",
         {"intent": "roster", "level": "team", "entity": "Blue Area"}),
    Case("list the advisors under Tariq Mehmood",
         {"intent": "roster", "level": "unit_head", "entity": "Tariq Mehmood"}),
    Case("who works in Downtown",
         {"intent": "roster", "level": "team", "entity": "Downtown"}),
    Case("show me all employees in GCC",
         {"intent": "roster", "level": "team", "entity": "GCC"}),
    Case("advisors under Usman Ghani",
         {"intent": "roster", "level": "bcm", "entity": "Usman Ghani"}),
    Case("Tariq Mehmood's team",
         {"intent": "breakdown", "level": "unit_head", "entity": "Tariq Mehmood"}),
    Case("show me Fawad Hafeez's team",
         {"intent": "breakdown", "level": "zonal_head", "entity": "Fawad Hafeez"}),
    Case("give me a breakdown of Usman Ghani",
         {"intent": "breakdown", "level": "bcm", "entity": "Usman Ghani"}),
    Case("how is Sadia Rehman doing",
         {"intent": "breakdown", "level": "unit_head", "entity": "Sadia Rehman"}),
    Case("tell me about Beverly Center",
         {"intent": "breakdown", "level": "office", "entity": "Beverly Center"}),
    Case("list all advisors under Tariq Mehmood, not grouped",
         {"intent": "roster", "level": "unit_head", "entity": "Tariq Mehmood"}),
    Case("which zonal head oversees BCM Usman Ghani",
         {"intent": "reverse_hierarchy", "level": "zonal_head", "entity": "Usman Ghani"}),
    Case("which unit head is Fawad Hafeez under",
         {"intent": "reverse_hierarchy", "level": "unit_head", "entity": "Fawad Hafeez"}),

    # ---- Phase 5.4: a manager's manager, and whole-chain traversal.
    Case("who is Usman Ghani's zonal head",
         {"intent": "reverse_hierarchy", "level": "zonal_head", "entity": "Usman Ghani"}),
    Case("which unit head manages Fawad Hafeez",
         {"intent": "reverse_hierarchy", "level": "unit_head", "entity": "Fawad Hafeez"}),
    Case("which zonal head is Rabia Anjum under",
         {"intent": "reverse_hierarchy", "level": "zonal_head", "entity": "Rabia Anjum"}),
    # No role named: the chain's own parent decides, so the answer depends
    # on where the subject sits rather than on a fixed default level.
    Case("who is above Yasir Ali",
         {"intent": "ancestry", "level": "advisor", "entity": "Yasir Ali"}),
    Case("who is above Usman Ghani",
         {"intent": "ancestry", "level": "bcm", "entity": "Usman Ghani"}),
    Case("show me the full hierarchy above Shehryar Abbasi",
         {"intent": "ancestry", "level": "advisor", "entity": "Shehryar Abbasi"}),
    Case("what is the reporting line for Fawad Hafeez",
         {"intent": "ancestry", "level": "zonal_head", "entity": "Fawad Hafeez"}),
    Case("show me the chain of command above Rabia Anjum",
         {"intent": "ancestry", "level": "bcm", "entity": "Rabia Anjum"}),
    # Forward hierarchy must be unaffected by the reverse widening.
    Case("who reports to Tariq Mehmood",
         {"intent": "breakdown", "level": "unit_head", "entity": "Tariq Mehmood"}),
]


# =====================================================================
# 5. PERIOD QUESTIONS
# =====================================================================

PERIODS = [
    Case("top advisors by revenue this month",
         {"intent": "leaderboard", "metric": "mtd_cleared", "period": "MTD"}),
    Case("top advisors by revenue year to date",
         {"intent": "leaderboard", "period": "YTD"}, finding="F4"),
    Case("top advisors by revenue this year",
         {"intent": "leaderboard", "period": "YTD"}),
    Case("top advisors by ytd revenue",
         {"intent": "leaderboard", "metric": "ytd_cleared", "period": "YTD"}),
    Case("top advisors by revenue this quarter",
         {"intent": "leaderboard", "period": "3M"}, finding="F4"),
    Case("top advisors by revenue over the last 3 months",
         {"intent": "leaderboard", "period": "3M"}),
    Case("top advisors by revenue month to date",
         {"intent": "leaderboard", "period": "MTD"}),
    Case("top advisors by revenue currently",
         {"intent": "leaderboard", "period": "MTD"}),
    Case("top advisors by revenue today",
         {"intent": "leaderboard", "period": "DAILY"}, finding="F4"),
    Case("top advisors by revenue right now",
         {"intent": "leaderboard", "period": "DAILY"}),
    # Phase 12: connects gained a REAL daily source (calls.connects_daily),
    # so "today" now resolves the daily sibling instead of carrying DAILY
    # on the MTD key and being refused downstream. The period expectation
    # is unchanged — what changed is that there is now a binding for it.
    Case("top advisors by connects today",
         {"intent": "leaderboard", "metric": "daily_connects", "period": "DAILY"}),
    # ... while a measure with no daily source keeps carrying DAILY on its
    # own key, so the compiler can refuse honestly rather than answer MTD.
    Case("top advisors by cr today",
         {"intent": "leaderboard", "metric": "client_registrations", "period": "DAILY"}),
    Case("top advisors by revenue",
         {"intent": "leaderboard", "period": "MTD"}),
    Case("top advisors by revenue last month",
         {"intent": "clarify", "period": None}),
    Case("top advisors by revenue yesterday",
         {"intent": "clarify", "period": None}),
    Case("top advisors by revenue this week",
         {"intent": "clarify", "period": None}),
    Case("top advisors by revenue in the past 7 days",
         {"intent": "clarify", "period": None}),
    Case("top advisors by revenue last quarter",
         {"intent": "clarify", "period": None}),
    Case("how is Blue Area doing this year",
         {"intent": "summary", "level": "team", "period": "YTD"}),
    # The YTD tabs are imported, so these resolve to real ytd_* siblings
    # instead of refusing. Before the import they were all "no data".
    Case("top advisors by ytd connects",
         {"intent": "leaderboard", "metric": "ytd_connects", "period": "YTD"}),
    Case("top advisors by connects year to date",
         {"intent": "leaderboard", "metric": "ytd_connects", "period": "YTD"}),
    Case("top advisors by ytd meetings",
         {"intent": "leaderboard", "metric": "ytd_meetings", "period": "YTD"}),
    Case("top advisors by ytd conversions",
         {"intent": "leaderboard", "metric": "ytd_conversion", "period": "YTD"}),
    Case("top advisors by ytd pipeline",
         {"intent": "leaderboard", "metric": "ytd_pipeline_value", "period": "YTD"}),
    Case("top advisors by ytd overdue",
         {"intent": "leaderboard", "metric": "ytd_overdue", "period": "YTD"}),
    Case("top advisors by ytd client registrations",
         {"intent": "leaderboard", "metric": "ytd_client_registrations", "period": "YTD"}),
    Case("top advisors by ytd bookings",
         {"intent": "leaderboard", "metric": "ytd_bookings", "period": "YTD"}),
    Case("top advisors by new connects year to date",
         {"intent": "leaderboard", "metric": "ytd_new_connects", "period": "YTD"}),
    Case("top advisors by followup connects ytd",
         {"intent": "leaderboard", "metric": "ytd_followup_connects", "period": "YTD"}),
    Case("top advisors by meetings conducted",
         {"intent": "leaderboard", "metric": "meetings_conducted"}),
    # Phase 5.1 vocabulary. Each of these matched NOTHING before, so it
    # fell through to the MTD default and a question about today was
    # answered with month-to-date figures.
    Case("top advisors by revenue this morning",
         {"intent": "leaderboard", "period": "DAILY"}),
    Case("top advisors by revenue this afternoon",
         {"intent": "leaderboard", "period": "DAILY"}),
    Case("top advisors by revenue this evening",
         {"intent": "leaderboard", "period": "DAILY"}),
    Case("what is the current revenue for Blue Area",
         {"intent": "leaderboard", "period": "MTD"}),
    Case("top advisors by revenue in the current month",
         {"intent": "leaderboard", "period": "MTD"}),
    Case("top advisors by revenue in the current year",
         {"intent": "leaderboard", "period": "YTD"}),
    Case("top advisors by revenue in the current quarter",
         {"intent": "leaderboard", "period": "3M"}),
    Case("how is Blue Area doing today",
         {"intent": "summary", "level": "team", "period": "DAILY"}),
]


# =====================================================================
# 6. THRESHOLD FILTERS
# =====================================================================

THRESHOLDS = [
    Case("advisors above 80% achievement",
         {"intent": "leaderboard", "metric": "achievement_pct",
          "comparators": (">",), "filters": (("achievement_pct", ">", 80.0),)},
         finding="F8"),
    Case("advisors over 80 percent achievement",
         {"intent": "leaderboard", "comparators": (">",)}),
    Case("advisors with more than 80 percent achievement",
         {"intent": "leaderboard", "comparators": (">",)}),
    Case("advisors with at least 80 percent achievement",
         {"intent": "leaderboard", "comparators": (">=",),
          "filters": (("achievement_pct", ">=", 80.0),)}),
    Case("advisors below 50 percent achievement",
         {"intent": "leaderboard", "comparators": ("<",),
          "filters": (("achievement_pct", "<", 50.0),)}),
    Case("advisors under 50 percent achievement",
         {"intent": "leaderboard", "comparators": ("<",)}),
    Case("advisors with at most 50 percent achievement",
         {"intent": "leaderboard", "comparators": ("<=",),
          "filters": (("achievement_pct", "<=", 50.0),)}),
    Case("advisors with no more than 50 percent achievement",
         {"intent": "leaderboard", "comparators": ("<=",),
          "filters": (("achievement_pct", "<=", 50.0),)}, finding="F8"),
    Case("advisors with no less than 80 percent achievement",
         {"intent": "leaderboard", "comparators": (">=",),
          "filters": (("achievement_pct", ">=", 80.0),)}),
    Case("advisors not below 80 percent achievement",
         {"intent": "leaderboard", "comparators": (">=",)}),
    Case("advisors not above 50 percent achievement",
         {"intent": "leaderboard", "comparators": ("<=",)}),
    Case("advisors with 80 percent or higher achievement",
         {"intent": "leaderboard", "comparators": (">=",)}),
    Case("advisors with 50 percent or less achievement",
         {"intent": "leaderboard", "comparators": ("<=",)}),
    Case("teams between 60 and 80 achievement",
         {"intent": "leaderboard", "level": "team", "comparators": ("<=", ">="),
          "filters": (("achievement_pct", "<=", 80.0), ("achievement_pct", ">=", 60.0))},
         finding="F8"),
    Case("advisors with achievement between 50 and 70",
         {"intent": "leaderboard", "comparators": ("<=", ">=")}),
    Case("advisors in Blue Area with achievement above 80 percent",
         {"intent": "leaderboard", "comparators": (">",),
          "filters": (("achievement_pct", ">", 80.0), ("team", "=", "Blue Area"))}),
    Case("advisors with more than 50 connects",
         {"intent": "leaderboard", "metric": "total_connects", "comparators": (">",)}),
    Case("teams with overdue above 3",
         {"intent": "leaderboard", "metric": "overdue", "level": "team", "comparators": (">",)}),
    Case("advisors with pipeline over 3000",
         {"intent": "leaderboard", "metric": "pipeline_value", "comparators": (">",)}),
    Case("top 5 advisors with achievement above 60 percent",
         {"intent": "leaderboard", "limit": 5, "comparators": (">",)}),
]


# =====================================================================
# 7. ATTENDANCE QUERIES
# =====================================================================

ATTENDANCE = [
    Case("who was late today",
         {"intent": "attendance_filter", "period": "DAILY"}),
    Case("show attendance issues",
         {"intent": "shortcut:attendance_check", "metric": None}),
    Case("any attendance problems today",
         {"intent": "shortcut:attendance_check", "metric": None}),
    Case("late advisors in Blue Area",
         {"intent": "attendance_filter", "entity": "Blue Area"}),
    Case("who is absent in Gulberg",
         {"intent": "attendance_filter", "entity": "Gulberg"}),
    Case("show me advisors who are not marked",
         {"intent": "attendance_filter", "metric": None, "entity": None}),
    Case("who is present in GCC",
         {"intent": "attendance_filter", "entity": "GCC"}),
    Case("top advisors by attendance rate",
         {"intent": "leaderboard", "metric": "attendance_rate"}),
    Case("advisors with attendance below 60 percent",
         {"intent": "leaderboard", "metric": "attendance_rate", "comparators": ("<",)}),
    Case("who has the most late arrivals",
         {"intent": "leaderboard", "metric": "late_count", "ranking": "desc"}),
    Case("top teams by punctuality",
         {"intent": "leaderboard", "metric": "attendance_rate", "level": "team"}),
    # Not attendance questions at all. They live in this category because
    # "calculated"/"related"/"escalate" all contain "late", and each was
    # once answered with a list of late advisors (finding F1). The
    # standing property test asserts none of them can route to
    # attendance_filter again.
]




# =====================================================================
# 8. REVERSE HIERARCHY (split out — the chain traversed UPWARD)
# =====================================================================

REVERSE_HIERARCHY = [
    Case("who is Sana Tariq's zonal head",
         {"intent": "reverse_hierarchy", "level": "zonal_head", "entity": "Sana Tariq"}),
    Case("who is Hina Malik's unit head",
         {"intent": "reverse_hierarchy", "level": "unit_head", "entity": "Hina Malik"}),
    Case("who is Waqar Haider's manager",
         {"intent": "reverse_hierarchy", "entity": "Waqar Haider"}),
    Case("who is Omar Farooq's boss",
         {"intent": "reverse_hierarchy", "entity": "Omar Farooq"}),
    Case("who does Zainab Noor report to",
         {"intent": "reverse_hierarchy", "entity": "Zainab Noor"}),
    Case("who is Faisal Iqbal reporting to",
         {"intent": "reverse_hierarchy", "entity": "Faisal Iqbal"}),
    Case("who is Nadia Sheikh managed by",
         {"intent": "reverse_hierarchy", "entity": "Nadia Sheikh"}),
    Case("which unit head manages Shehryar Abbasi",
         {"intent": "reverse_hierarchy", "level": "unit_head", "entity": "Shehryar Abbasi"}),
    Case("which bcm leads Salman Arshad",
         {"intent": "reverse_hierarchy", "level": "bcm", "entity": "Salman Arshad"}),
    Case("which zonal head oversees Rabia Anjum",
         {"intent": "reverse_hierarchy", "level": "zonal_head", "entity": "Rabia Anjum"},
         finding="F10"),
    Case("which unit head is Adeel Aslam under",
         {"intent": "reverse_hierarchy", "level": "unit_head", "entity": "Adeel Aslam"},
         finding="F10"),
    Case("who is above Bilal Qadir",
         {"intent": "ancestry", "level": "bcm", "entity": "Bilal Qadir"},
         finding="F10"),
    Case("show me the reporting line for Sana Tariq",
         {"intent": "ancestry", "level": "advisor", "entity": "Sana Tariq"}),
    Case("show me the whole management chain above Nadia Sheikh",
         {"intent": "ancestry", "level": "advisor", "entity": "Nadia Sheikh"}),
    Case("who is Salman Arshad's zonal head",
         {"intent": "reverse_hierarchy", "level": "zonal_head", "entity": "Salman Arshad"}),
    Case("who is Zainab Noor's unit head",
         {"intent": "reverse_hierarchy", "level": "unit_head", "entity": "Zainab Noor"}),
    Case("who is above Fawad Hafeez",
         {"intent": "ancestry", "level": "zonal_head", "entity": "Fawad Hafeez"}),
    Case("which zonal head is Usman Ghani under",
         {"intent": "reverse_hierarchy", "level": "zonal_head", "entity": "Usman Ghani"},
         finding="F10"),
]


# =====================================================================
# 9. KPI TERMINOLOGY (business words -> the right measure)
# =====================================================================

KPI_TERMINOLOGY = [
    Case("top advisors by CR", {"intent": "leaderboard", "metric": "client_registrations"}),
    Case("top advisors by conversions", {"intent": "leaderboard", "metric": "conversion"}),
    Case("top teams by conversion percentage",
         {"intent": "leaderboard", "metric": "meeting_to_conversion_rate", "level": "team"}),
    Case("top advisors by achievement",
         {"intent": "leaderboard", "metric": "achievement_pct"}),
    Case("top advisors by achievement %",
         {"intent": "leaderboard", "metric": "achievement_pct"}),
    Case("top advisors by performance %",
         {"intent": "leaderboard", "metric": "achievement_pct"}),
    Case("top advisors by pipeline value",
         {"intent": "leaderboard", "metric": "pipeline_value"}),
    Case("top advisors by portfolio value",
         {"intent": "leaderboard", "metric": "portfolio_value"}),
    Case("top advisors by cr to meeting",
         {"intent": "leaderboard", "metric": "cr_to_meeting_rate"}),
    Case("top teams by overdue count",
         {"intent": "leaderboard", "metric": "overdue", "level": "team"}),
    Case("top advisors by target", {"intent": "leaderboard", "metric": "mtd_target"}),
    Case("top advisors by meetings", {"intent": "leaderboard", "metric": "total_meetings"}),
    # A RATE must never resolve to the COUNT inside it.
    # RETIRED REFUSALS — computable since working_days.py. F13's point
    # holds and is still what these assert: the rate, never the count.
    Case("top advisors by CR %",
         {"intent": "leaderboard", "metric": "cr_rate"}, finding="F13"),
    Case("top advisors by CR%",
         {"intent": "leaderboard", "metric": "cr_rate"}, finding="F13"),
    Case("top advisors by answered calls %",
         {"intent": "leaderboard", "metric": "answered_calls_rate"}, finding="F13"),
    Case("top advisors by answered call percent",
         {"intent": "leaderboard", "metric": "answered_calls_rate"}, finding="F13"),
    Case("top teams by meeting rate",
         {"intent": "leaderboard", "metric": "meeting_rate"}),
    Case("top teams by 1 unit ratio",
         {"intent": "leaderboard", "metric": "one_unit_ratio", "level": "team"}),
    Case("top advisors by worksapp login",
         {"intent": "leaderboard", "metric": "login_rate"}),
    Case("top advisors by meetings planned",
         {"intent": "leaderboard", "metric": "meetings_planned"}),
    Case("top teams by meeting conduction rate",
         {"intent": "leaderboard", "metric": "meeting_conduction_rate", "level": "team"}),
    Case("top advisors by portfolio %", {"intent": "clarify_metric", "metric": None}),
]


# =====================================================================
# 10. CLARIFICATIONS (the system asks rather than guessing)
# =====================================================================

CLARIFICATIONS = [
    Case("which team has the highest sparkle index",
         {"intent": "clarify_metric", "metric": None}, finding="F6"),
    Case("rank advisors by synergy score",
         {"intent": "clarify_metric", "metric": None}, finding="F6"),
    Case("top advisors by revenue in the past 30 days", {"intent": "clarify", "period": None}),
    Case("top advisors by revenue for the previous month", {"intent": "clarify", "period": None}),
    Case("rank teams by flux capacitance", {"intent": "clarify_metric", "metric": None},
         finding="F6"),
    Case("compare Downtown and Narnia",
         {"intent": "comparison_incomplete", "entity": "Downtown"}),
    Case("top advisors by moonbeam efficiency",
         {"intent": "clarify_metric", "metric": None}, finding="F6"),
    Case("compare Gulberg and Wakanda",
         {"intent": "comparison_incomplete", "entity": "Gulberg"}),
    Case("top advisors by revenue between jan 1 and mar 31 2024",
         {"intent": "clarify", "period": None}),
    Case("show me the sales figures for Atlantis",
         {"intent": "leaderboard", "metric": "mtd_cleared"},
         known_gap="An unknown entity is silently DROPPED rather than questioned: "
                   "'Atlantis' grounds to nothing, so the query becomes a global "
                   "revenue leaderboard. It should say it could not find Atlantis. "
                   "Distinct from comparison_incomplete, which does report the "
                   "missing side, because there is only one entity here to lose."),
]


# =====================================================================
# 11. AMBIGUOUS ENTITIES (a name that means more than one thing)
# =====================================================================

AMBIGUOUS = [
    # "Kamran Shah" is both a BCM and an advisor in the fixture.
    Case("how is Kamran Shah doing",
         {"intent": "clarify_ambiguous", "metric": None}),
    Case("tell me about Kamran Shah",
         {"intent": "clarify_ambiguous", "metric": None}),
    # Naming the level resolves it without asking.
    Case("how is BCM Kamran Shah doing",
         {"intent": "breakdown", "level": "bcm", "entity": "Kamran Shah"}),
    # A reverse question is about the PERSON, so "unit head" names the
    # role asked for rather than the subject.
    Case("who is Kamran Shah's unit head",
         {"intent": "reverse_hierarchy", "level": "unit_head", "entity": "Kamran Shah"}),
    # Two real people share "Ali Raza" — ask which, never pick.
    Case("tell me about Ali Raza",
         {"intent": "clarify_person", "entity": "Ali Raza"}),
    Case("what is Ali Raza's revenue",
         {"intent": "clarify_person", "entity": "Ali Raza"}),
    Case("who is Ali Raza's bcm",
         {"intent": "clarify_person", "entity": "Ali Raza"}),
    # Phase 22: a bare measure question about a name that is ALSO a person
    # answers about the PERSON. Every manager is an advisor with their own
    # figures, and being a BCM must not turn "what is X's revenue" into a
    # question about the people under X — there was no phrasing left that
    # reached the individual. The two cases above still ask, because they
    # name no measure and so have no person reading to prefer.
    Case("what is Kamran Shah's revenue",
         {"intent": "advisor_metric", "level": "advisor", "entity": "Kamran Shah",
          "metric": "mtd_cleared"}),
    Case("all advisors under Kamran Shah",
         {"intent": "advisor_profile", "level": "advisor", "entity": "Kamran Shah"},
         known_gap="Should be a roster of BCM Kamran Shah's advisors, or a "
                   "clarification. The relational phrase 'under X' makes "
                   "_score_clarify_ambiguous prune to the managerial reading, but the "
                   "advisor reading then outscores the roster, so the query returns "
                   "one person's profile and drops 'all advisors' entirely."),
    # Unambiguous names in the SAME shapes, so the clarifications above
    # are shown to be driven by the ambiguity rather than the phrasing.
    Case("what is Hina Malik's revenue",
         {"intent": "advisor_metric", "metric": "mtd_cleared", "entity": "Hina Malik"}),
    Case("tell me about Waqar Haider",
         {"intent": "advisor_profile", "entity": "Waqar Haider"}),
    Case("who is Omar Farooq's bcm",
         {"intent": "reverse_hierarchy", "level": "bcm", "entity": "Omar Farooq"}),
    Case("all advisors under Rabia Anjum",
         {"intent": "roster", "level": "bcm", "entity": "Rabia Anjum"}),
]


# =====================================================================
# 12. NEGATIVE CASES (must NOT be understood as something else)
# =====================================================================

NEGATIVE = [
    # F1 — "late" inside unrelated words. attendance_filter scores 0.98
    # and returns before the parser runs, so a collision hijacks the
    # whole query.
    # RETIRED REFUSAL. F1's point was that a shortcut must not hijack
    # this phrasing; the measure now resolves, so the metric is what it
    # names rather than a refusal.
    Case("how is the answered calls percentage calculated",
         {"intent": "leaderboard", "metric": "answered_calls_rate"}, finding="F1"),
    Case("show me related teams", {"intent": "clarify", "metric": None}, finding="F1"),
    Case("escalate to the manager", {"intent": "clarify", "metric": None}, finding="F1"),
    Case("please translate this reply", {"intent": "clarify", "metric": None}, finding="F1"),
    Case("who is the representative", {"intent": "clarify", "metric": None}, finding="F1"),
    Case("how is this calculated", {"intent": "clarify", "metric": None}, finding="F1"),
    # F9 / Phase 5.3 — comparator and ranking words inside other words.
    Case("what is the turnover 500", {"intent": "clarify", "comparators": ()},
         finding="F9"),
    Case("show me handover 3", {"intent": "clarify", "comparators": ()}, finding="F9"),
    # A SUPERLATIVE is a ranking, not a comparison.
    # A roster is not a ranking, and a ranking is not a roster.
    Case("top 5 advisors in Blue Area",
         {"intent": "leaderboard", "metric": "mtd_cleared", "limit": 5}),
    # Forward hierarchy is not reverse hierarchy.
    Case("who reports to Sadia Rehman",
         {"intent": "breakdown", "level": "unit_head", "entity": "Sadia Rehman"}),
    Case("Sadia Rehman's team",
         {"intent": "breakdown", "level": "unit_head", "entity": "Sadia Rehman"}),
    # Greetings and help must never become analytics.
    Case("hello", {"intent": "shortcut:greeting", "metric": None}),
    Case("thanks", {"intent": "shortcut:thanks", "metric": None}),
    Case("what can you do", {"intent": "shortcut:help", "metric": None}),
]


CATEGORIES: dict[str, list[Case]] = {
    "leaderboards": LEADERBOARDS,
    "comparisons": COMPARISONS,
    "kpi_questions": KPI_QUESTIONS,
    "hierarchy": HIERARCHY,
    "periods": PERIODS,
    "thresholds": THRESHOLDS,
    "attendance": ATTENDANCE,
    "reverse_hierarchy": REVERSE_HIERARCHY,
    "kpi_terminology": KPI_TERMINOLOGY,
    "clarifications": CLARIFICATIONS,
    "ambiguous_entities": AMBIGUOUS,
    "negative_cases": NEGATIVE,
}

ALL_CASES: list[tuple[str, Case]] = [
    (category, case) for category, cases in CATEGORIES.items() for case in cases
]
