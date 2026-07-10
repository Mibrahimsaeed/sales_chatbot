INTENT_SCHEMA = """Return ONLY a JSON object, no other text, no markdown fences, matching this shape:
{
  "intent": "advisor_lookup" | "team_summary" | "company_summary" | "leaderboard" | "attendance_check" | "help" | "unknown",
  "entities": {
    "advisor_name": string or null,
    "team": string or null,
    "company": string or null,
    "metric": "mtd_cleared" or "mtd_new_connect" or "overdue" or null,
    "period": "MTD" or "YTD" or "3M" or null,
    "limit": number or null
  }
}"""


def build_intent_prompt(text: str, known_teams: list[str], known_companies: list[str]) -> str:
    teams_sample = ", ".join(known_teams[:40])
    companies = ", ".join(known_companies)
    return f"""You are an intent classifier for a real-estate sales operations chatbot.
Known teams (not exhaustive): {teams_sample}
Known companies: {companies}

A rule-based classifier already tried this message and wasn't confident enough — that's why you're being asked. Classify the user's message and extract any entities it mentions, using only the categories below.

User message: "{text}"

{INTENT_SCHEMA}"""