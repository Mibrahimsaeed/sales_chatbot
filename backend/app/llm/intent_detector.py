import re


def classify_intent(text: str) -> dict:
    q = text.lower()

    if re.search(r"top.*(revenue|cleared|sale)", q):
        return {"type": "leaderboard", "metric": "mtd_cleared"}
    if re.search(r"top.*connect", q):
        return {"type": "leaderboard", "metric": "mtd_new_connect"}
    if re.search(r"(worst|most).*overdue", q):
        return {"type": "leaderboard", "metric": "overdue"}

    team_match = re.search(r"team\s+(.+)", q)
    if team_match:
        return {"type": "team_summary", "team": team_match.group(1).strip()}

    cleaned = re.sub(r"tell me about|how is|show me|what about", "", text, flags=re.I).strip()
    if len(cleaned) > 2:
        return {"type": "advisor_lookup", "name": cleaned}

    return {"type": "unknown"}