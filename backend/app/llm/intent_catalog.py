"""
Intent catalog — the declarative half of the planner.

WHY THIS EXISTS. Intent selection used to be an ordered list of
`_plan_*` functions where the FIRST one to return a plan won. That made
priority implicit in source order, and every conflict had to be fixed by
hand-coding a guard into whichever function happened to run first
("decline if metric and ranking", "decline if relational", ...). Each
guard was correct in isolation and invisible from anywhere else, so the
same class of bug kept reappearing in new phrasings.

Here every intent instead declares:

  - the SIGNALS it looks for (trigger phrases, required entities)
  - what it REQUIRES to be a candidate at all
  - the WEIGHTS those signals contribute

and query_planner scores every candidate and picks the highest. Conflicts
are resolved by comparing evidence, not by who was checked first, and the
winning score plus its evidence is recorded in the request trace — so
"why did this query route there?" is answerable from a log line rather
than by re-reading control flow.

Weights are tuned so that with NO distinguishing evidence the ordering
reproduces the previous priority list exactly; evidence is what moves a
query off that default.
"""

from __future__ import annotations

import re

from app.llm import hierarchy, relations, token_match

# =====================================================================
# Trigger patterns
# =====================================================================

# RANKING — "top 5", "best", "worst". Split by strength on purpose:
# "top/best/worst" genuinely means a ranking, whereas "show me"/"give me"
# is how people open ANY request and is barely evidence of anything.
# Treating them as equal is what let "show me X's team" look like a
# ranking.
RANKING_STRONG = ("top", "bottom", "best", "worst", "highest", "lowest",
                  "most", "least", "rank", "leaderboard")
RANKING_WEAK = ("give me", "show me")

# Phase 2 — sort direction vocabulary, split by what the word actually
# means. The distinction matters once a metric can declare that LOWER is
# better (overdue, late arrivals):
#
#   ABSOLUTE words name a numeric end. "highest overdue" means the
#   largest number regardless of whether large is good.
#   RELATIVE words name a QUALITY end. "worst overdue" means the most
#   overdue, while "worst revenue" means the least revenue — the same
#   word, opposite directions, resolved against the metric's polarity.
#
# With no word at all the ranking shows the GOOD end, which is what
# "top 5" has always meant and is why overdue must rank ascending.
ASCENDING_ABSOLUTE = ("lowest", "least")
DESCENDING_ABSOLUTE = ("highest", "most")

# FIX 1. Words naming the BAD end of a metric. Resolved against polarity,
# never by reversing the sort:
#
#   worst/bottom revenue  -> the LOWEST revenue   (higher is better)
#   worst/bottom overdue  -> the HIGHEST overdue  (lower is better)
#
# "bottom" was in no list at all, so it contributed neither a direction
# nor ranking evidence: "bottom 5 advisors" became an unbounded
# DESCENDING ranking and returned the top performers. Wrong direction and
# wrong row count, presented confidently.
WORST_RELATIVE = ("worst", "bottom")

# Words naming the GOOD end. They need no entry in _sort_signal — with no
# direction word the metric's own polarity decides (query_compiler.
# default_direction), which already yields the good end. Declared so the
# vocabulary is visible beside its opposite, and so the limit pattern
# below can be built from it.
BEST_RELATIVE = ("top", "best")

# FIX 1. "top 5" / "bottom 5" / "worst 3" — the ranking words that can
# carry an explicit N. Built from the vocabularies above so a word added
# there gains limit support automatically.
#
# "least" and "most" are deliberately EXCLUDED: "at least 80" would
# otherwise read as a limit of 80. They are comparator-adjacent, and no
# one says "least 5 advisors".
LIMIT_RANKING_WORDS = (*BEST_RELATIVE, *WORST_RELATIVE, "highest", "lowest")

FLAT_KEYWORDS = ("flat", "list all", "as a list", "not nested", "without teams", "ungrouped")

# ENUMERATE — "connects of ALL BCMs": every member, not the leaders.
#
# "top advisors by connects" and "connects of all BCMs" both produce a
# ranked list, and both were capped at the same default of 10 — so a
# question that says ALL answered with a tenth of the answer and reported
# it as the whole thing. The two are different requests: one asks who is
# ahead, the other asks for the roll. Only this one lifts the cap;
# pagination then shows a page at a time.
#
# Deliberately narrow. It requires an explicit "all"/"every"/"each" —
# a bare "BCM connects" is not an enumeration, and neither is "most" or
# "top", which carry their own meaning about how many are wanted. An
# explicit "top 5" still wins over this, because a stated number is the
# most specific thing the user can say about size.
ENUMERATE_WORDS = ("all", "every", "each", "entire", "complete", "full list")


# ROSTER — "who is IN this group": the answer is a list of PEOPLE.
# Requires a word for people ("advisors"/"employees"/...) or an explicit
# "who works in". A bare "who is in X's team" is deliberately NOT a
# roster trigger — that asks about the team's shape, which the nested
# breakdown answers.

# The words that name PEOPLE rather than a hierarchy level. `advisor` is
# the level these all denote — hierarchy.LEVEL_KEYWORDS carries the
# level's own names ("advisor", "agent"), and these are the rest of the
# ways the same population is said. Declared once because two callers
# need it: ROSTER_RE below, and the direct-report target scan, which has
# to read "people who report directly to X" as the advisor question it is.
PEOPLE_WORDS = ("advisors", "advisers", "employees", "people", "staff", "agents")

# "people" is deliberately absent here: "<people-word> in X" is a roster
# trigger, but "people in X" is loose enough to catch questions that are
# not roster questions at all.
_SCOPED_PEOPLE_WORDS = tuple(w for w in PEOPLE_WORDS if w != "people")

ROSTER_RE = re.compile(
    r"\b(all|list|show|name|give\s+me)\s+(the\s+|me\s+the\s+)?"
    rf"({'|'.join(PEOPLE_WORDS)})\b"
    rf"|\b({'|'.join(_SCOPED_PEOPLE_WORDS)})\s+(in|from|under|at|of|for|assigned\s+to)\b"
    r"|\bwho\s+(works|work)\s+(in|at|for|under|with)\b",
    re.I,
)

# COMPARISON — two or more entities set against each other.
#
# Previously reachable ONLY through the LLM semantic parser (which emits
# QueryIR intent="comparison"). The rule-based planner had no comparison
# intent at all, so whenever the LLM was unavailable — or simply not
# consulted — "Compare Graana and Agency21" degraded to the metric-help
# message, and "…by revenue" degraded to a plain leaderboard that
# silently ignored both named entities.
COMPARISON_RE = re.compile(
    r"\bcompare\b"
    r"|\bcomparison\b"
    r"|\bdifference\s+between\b"
    r"|\b(vs\.?|versus)\b"
    # "which TEAM is doing better, X or Y". The noun between "which" and
    # the verb is the whole gap: the pattern used to allow only "which"
    # or "which one", so naming the level — the most natural way to ask —
    # defeated it and the query degraded to a one-sided summary of
    # whichever entity grounded first.
    #
    # Up to three words are allowed between, which covers "which of the
    # two teams". The comparative is deliberately limited to
    # better/worse: a SUPERLATIVE ("which team has the highest CR") is a
    # ranking, not a two-sided comparison, and must keep reaching the
    # leaderboard.
    r"|\bwhich\s+(?:\w+\s+){0,3}?(is|are|was|were|has|have|had|does|do|did|"
    r"perform(s|ed|ing)?)\b.*\b(better|worse)\b"
    r"|\bwhich\s+(?:of\s+)?(?:\w+\s+){0,3}?(?:one\s+)?(?:is|are)\s+(?:the\s+)?(?:better|worse)\b"
    r"|\bwho\s+(is|are)\s+(doing\s+)?better\b"
    r"|\bhow\s+do(es)?\s+.+\s+(compare|stack\s+up)\b"
    r"|\b(better|worse)\s+(than|between)\b"
    # Phase 5B — the ENUMERATED superlative. "Who has more revenue, Blue
    # Area or Downtown?" and "Which of BCM X's or BCM Y's groups has more
    # conversions?" are two-sided questions, and routing them to the
    # leaderboard answered with ONE subject's figure — whichever the
    # extractor listed first, never compared against the other. That read
    # as correct on the fixture only because its gazetteer order happened
    # to match the metric order; inverting the data returned the loser.
    #
    # The distinction from the ranking above is the DISJUNCTION, not the
    # comparative word. "Which team has the highest CR" ranks every team
    # and stays a leaderboard; naming the alternatives with "or" is what
    # makes it a comparison of those alternatives. Grounding is not
    # checked here — _score_comparison already requires two resolved
    # targets, so a stray "or" cannot manufacture a comparison.
    # Either order: "who has MORE revenue, A OR B" and "which of A's OR
    # B's groups has MORE conversions" are the same question, and spec
    # query 67 is written the second way.
    r"|\b(who|which)\b[^?]*\b(more|less|higher|lower|greater|better|worse|"
    r"most|least|highest|lowest|best)\b[^?]*\bor\b"
    r"|\b(who|which)\b[^?]*\bor\b[^?]*\b(more|less|higher|lower|greater|"
    r"better|worse|most|least|highest|lowest|best)\b",
    re.I,
)

# FORWARD HIERARCHY — the people UNDER someone, as a nested structure.
RELATIONAL_RE = re.compile(
    r"('s|s')\s*(team|advisors|people|reports|staff|members)"
    r"|\bwho\s+(is\s+in|are\s+in|reports?\s+to|works?\s+under|works?\s+for)\b"
    r"|\b(team|advisors|members|people|reports)\s+(of|under|for)\b"
    r"|\bunder\s+\w+",
    re.I,
)

# TRANSITIVE SCOPE — "which BCMs work under Unit Head X".
#
# Deliberately NOT folded into RELATIONAL_RE above. That expression
# drives _score_hierarchy, which answers "X's team"; widening it to reach
# "below" and "reporting to" would change what those questions resolve to
# as a side effect. This one is read by a single scorer whose other three
# conditions (a named target level strictly below the manager, a grounded
# group entity, and no "directly") are what make it specific — so the
# relation words themselves can stay plain.
SCOPED_UNDER_RE = re.compile(
    r"\b(under|below|beneath|underneath)\b"
    r"|\breport(s|ing)?\s+to\b"
    r"|\bwork(s|ing)?\s+(under|for)\b",
    re.I,
)

# Role words that name a manager WITHOUT naming which level — "who is
# X's boss" asks the same question as "who is X's manager". They belong
# to language, not to any one relation, so they stay here rather than in
# the registry, and they resolve to DEFAULT_REVERSE_LEVEL by falling
# through level detection exactly as they always have.
GENERIC_ROLE_WORDS = ("manager", "boss", "supervisor", "lead")

# Metric synonyms that describe a person's standing in GENERAL rather
# than naming one measure. They earn their place in the ontology because
# "top performer" is a real ranking phrase, but on their own — "the
# performance of X" — they are asking how somebody is doing, which the
# full profile answers and a single percentage does not.
#
# Only the bare words: _SYNONYM_INDEX is longest-first, so a specific
# phrase containing one of them ("performance against target") still
# resolves as itself and is treated as a genuine metric request.
GENERAL_INTEREST_SYNONYMS = frozenset({"performance", "performer"})


def _alias_pattern(alias: str) -> str:
    r"""One alias as regex source. Spaces relax to \s* so "unithead" and
    "unit  head" keep matching, which is what the hand-written
    `unit\s*head` did."""
    return r"\s*".join(re.escape(word) for word in alias.split())


# M2: the reverse-role vocabulary is DERIVED from the relation registry
# rather than written out here.
#
# It used to live in two hand-maintained lists — this alternation and
# REVERSE_LEVEL_PATTERNS below — which had drifted apart: level detection
# knew "portfolio lead", "division head", "branch" and others, while the
# trigger did not, so those questions never reached reverse lookup at all.
# Two lists that must agree, and no mechanism forcing them to, is the
# defect; deriving both from one declaration is the fix. Adding a relation
# with role_aliases now teaches BOTH halves at once.
_MANAGER_ROLE = "|".join(
    _alias_pattern(alias)
    for alias in [*relations.role_aliases(), *GENERIC_ROLE_WORDS]
)

# REVERSE HIERARCHY — the person ABOVE someone. Note the contrast with
# RELATIONAL_RE: "who reports to X" is FORWARD, "who does X report to" is
# REVERSE. Each has its own explicit pattern rather than sharing a
# "report" keyword.
REVERSE_RE = re.compile(
    rf"('s|s')\s*({_MANAGER_ROLE})\b"
    rf"|\bwho\s+(is|are)\s+.*?('s|s')\s*({_MANAGER_ROLE})\b"
    rf"|\b(his|her|their|its)\s+({_MANAGER_ROLE})\b"
    r"|\bwho\s+does\s+.+?\s+report\s+to\b"
    r"|\bwho\s+(is|are)\s+.+?\s+(report(ing)?\s+to|under|managed\s+by)\b"
    rf"|\b({_MANAGER_ROLE})\s+(of|for)\b"
    # "the Unit Head IN AMD" — the same question as "the unit head OF
    # AMD", which this pattern already answered, said with the other
    # preposition. Without it the role word resolved to nothing, the
    # sentence degraded to a plain team lookup, and "how many advisors
    # report to the Unit Head in AMD" was answered with the whole team's
    # headcount — a different question, confidently.
    #
    # THE ARTICLE IS REQUIRED, unlike the (of|for) branch above. "in" is
    # the ordinary scoping preposition ("advisors in AMD", "BCMs in the
    # branch in Islamabad"), and `_MANAGER_ROLE` includes bare words like
    # "branch" and "lead" that appear inside those. "the <role> in" is
    # specific enough to mean the person holding that role.
    rf"|\b(the|our)\s+({_MANAGER_ROLE})\s+in\b"
    # Step 3: "which unit head MANAGES Ahmed" — the role named first, then
    # an active verb. Every other branch above expects the role to trail
    # its subject ("X's unit head") or the subject to trail a passive
    # ("who is X managed by"), so this phrasing matched nothing and
    # "which unit head manages Ahmed" was answered with Ahmed's own
    # profile. The role alternation is the same registry-derived one, so
    # a relation declared later gets this phrasing for free.
    rf"|\b(which|who|what)\s+({_MANAGER_ROLE})\s+"
    r"(manages|manage|oversees|oversee|leads|lead|heads|head|handles|handle|supervises|supervise)\b"
    # Role first, PASSIVE: "which unit head is Fawad Hafeez under". The
    # active form above covers "which unit head manages X"; this covers
    # the same question with the preposition at the end, which is how it
    # is usually said about a manager rather than an advisor.
    rf"|\b(which|who|what)\s+({_MANAGER_ROLE})\s+(is|was|are|were)\s+.+?\s+"
    r"(under|over|above|reporting\s+to)\b",
    re.I,
)

# Which manager level a reverse question asks about. Derived from the
# same declarations as the trigger above, so the two can no longer
# disagree. Ordering is longest-alias-first (see
# relations.role_alias_pairs), which reproduces the old list's hand-
# ordered specificity — "regional manager" ahead of "manager", "business
# center" ahead of "branch" — as a consequence of the data rather than of
# list order someone has to maintain.
REVERSE_LEVEL_PATTERNS: list[tuple[str, re.Pattern]] = [
    (level, re.compile(rf"\b{_alias_pattern(alias)}\b", re.I))
    for alias, level in relations.role_alias_pairs()
]
DEFAULT_REVERSE_LEVEL = "unit_head"

# ANCESTRY — every level above someone, not just one. "Who is above X"
# with no role named is the same question asked without knowing the
# vocabulary, so it belongs here rather than being answered about
# whichever level DEFAULT_REVERSE_LEVEL happens to be.
# Movement over time, as opposed to a figure at a point in time.
#
# Phase 3. `trend` has been in QueryIR's intent set and in
# ir_validator._UNSUPPORTED_INTENTS (with a written reason: the monthly
# snapshots a trend needs are not stored yet) since the redesign — but
# nothing in the rule planner ever produced it, so the refusal was
# unreachable and "show the trend of revenue" silently answered with a
# CURRENT leaderboard. A snapshot presented as a trend is a wrong answer,
# not a degraded one; detecting the request is what lets the system say
# so.
TREND_RE = re.compile(
    r"\btrends?\b"
    r"|\btrending\b"
    r"|\bover\s+time\b"
    r"|\bmonth\s+over\s+month\b|\bmonth-over-month\b"
    r"|\byear\s+over\s+year\b|\byear-over-year\b"
    r"|\bhistory\b|\bhistorical\b"
    r"|\bprogress(ion)?\s+(over|across)\b"
    r"|\b(improving|declining|worsening)\b"
    r"|\bgrowth\s+(rate|over)\b",
    re.I,
)

ANCESTRY_RE = re.compile(
    r"\b(full|whole|entire|complete)\s+(hierarchy|chain|reporting\s+line|management\s+chain)\b"
    r"|\bhierarchy\s+above\b"
    r"|\bwho\s+(is|are|sits?)\s+above\b"
    r"|\breporting\s+(line|chain)\b"
    r"|\bchain\s+of\s+command\b"
    r"|\ball\s+(the\s+)?managers\s+(above|over)\b",
    re.I,
)

# Levels a roster/summary can scope to, most granular first: a query
# naming both a team and its company means the team — the narrower answer
# is the more informative one.
# M5 inserted `unit` and `region` at their granularity: a unit is
# narrower than a zonal head's span, a region is wider than a unit head's
# and narrower than a company. Ordering matters only when a query names
# entities at several levels at once — the narrowest wins, being the more
# informative answer.
# Most granular first — derived from the chain (reversed, advisor
# excluded) plus the attributes. A query naming entities at several
# levels resolves to the narrowest, which is the more informative answer.
GROUP_LEVEL_ORDER = tuple(
    [lvl for lvl in reversed(hierarchy.CHAIN) if lvl != "advisor"]
    + list(hierarchy.ATTRIBUTE_LEVELS)
)

# Level-detection keywords for leaderboard subject level. Order preserved
# from the original implementation so an ambiguous query keeps resolving
# the way it always has.
_LEVEL_ORDER = ["advisor", "team", "company", *hierarchy.NEW_GROUP_LEVELS]
LEVEL_KEYWORDS = {level: hierarchy.LEVEL_KEYWORDS[level] for level in _LEVEL_ORDER}


# =====================================================================
# Scoring weights
#
# Read these as "how much does this piece of evidence argue for this
# intent". They are deliberately few and named — the previous design's
# behaviour lived in a dozen scattered boolean guards, which is exactly
# what made it hard to reason about.
# =====================================================================

# A hard gate, not evidence: an unresolvable ambiguity must beat every
# other reading, because answering ANY of them would be a guess.
W_HARD_GATE = 10.0

# An unambiguous trigger phrase for this intent ("all advisors in",
# "who is X's BM"). The single strongest ordinary signal.
W_EXPLICIT_PHRASE = 0.40

# The entity this intent operates on is grounded in the query.
W_ENTITY = 0.20

# A person resolved to a specific wid.
W_IDENTITY = 0.25

# A metric resolved from the ontology — the defining evidence for a
# leaderboard, and meaningless for every other intent.
W_METRIC = 0.30

W_RANKING_STRONG = 0.25
W_RANKING_WEAK = 0.05

# This intent honours a constraint that every competing reading would
# silently DROP. Weighted above an ordinary explicit phrase because
# dropping a constraint doesn't give a different answer to the same
# question — it gives a SUPERSET, which reads as authoritative and is
# simply wrong. "late advisors in Blue Area" matches roster phrasing, but
# a roster ignores "late" and returns all 54 people in the team.
W_SPECIFIC_CONSTRAINT = 0.30

# Two or more entities were named AND a comparison phrase is present.
# Weighted to beat a leaderboard even when a metric is also named:
# "compare Graana and Agency21 by revenue" is a two-sided question, and
# a ranking answers it with one list that drops the pairing entirely.
W_COMPARISON_PAIR = 0.45

# The metric a ranking falls back to when the user asked for "top N" but
# named no measure. "Top 5 advisors" is unambiguous about WANTING a
# ranking of 5 — only the measure is unstated — and revenue is the
# conventional default in a sales org.
#
# This is a disclosed default, not a silent guess: the reply header
# always names the metric it ranked by ("Top 5 by MTD Revenue Cleared"),
# and the trace records `default_metric` as evidence. The alternative
# behaviour was strictly worse — with no metric the leaderboard wasn't a
# candidate at all, so "top 5 advisors in Blue Area" fell through to the
# roster reading and returned all 54 advisors, dropping both "top" and
# "5".
DEFAULT_RANKING_METRIC = "mtd_cleared"

# "top 5 … by revenue" is more than the sum of its parts: a strong
# ranking word AND a metric together is a leaderboard even when the
# sentence also contains roster or relational phrasing ("top 5 advisors
# in Blue Area by revenue"). This bonus is what replaces the hand-coded
# "decline if metric and ranking" guards that used to live in the roster
# and hierarchy branches.
W_RANK_METRIC_COMBO = 0.25

# Base precedence, applied when evidence alone doesn't separate two
# intents. Ordering here reproduces the previous priority list, so a
# query with no distinguishing signals routes exactly as it did before.
PRIOR = {
    "clarify_ambiguous": 0.40,
    "clarify_person": 0.40,
    # Above roster and reverse_hierarchy, and deliberately: "directly"
    # is an explicit narrowing of a question both of those also match,
    # and the reading that HONOURS the word must win over the two that
    # ignore it. It only ever scores when the word is present.
    "direct_reports": 0.37,
    # Same tier as direct_reports: it is the same mechanism read
    # transitively, and the two are mutually exclusive (one requires
    # "directly", the other declines on it), so the tie never arises.
    # Above `roster` (0.35) because "the BCMs under X" and "the advisors
    # in X" are different questions and the roster reading answered both.
    "scoped_reports": 0.37,
    "comparison": 0.36,
    "roster": 0.35,
    "reverse_hierarchy": 0.33,
    "ancestry": 0.33,
    "hierarchy": 0.32,
    "attendance_filter": 0.28,
    # Same family as advisor_profile — one named person — so the same
    # prior. What separates them is evidence, not precedence: a resolved
    # metric adds W_SPECIFIC_CONSTRAINT, and without one this intent
    # never scores at all.
    "advisor_metric": 0.25,
    "advisor_profile": 0.25,
    "entity_summary": 0.20,
    "leaderboard": 0.18,
}

# Human-readable catalog, consumed by the audit report and by
# test_intent_matrix.py so documentation can't drift from the code.
INTENT_DOCS = {
    "clarify_ambiguous": "A name matches several hierarchy levels; ask which was meant.",
    "clarify_person": "A name matches several real people; ask which one.",
    "comparison": "Two or more entities side by side ('compare Graana and Agency21').",
    "roster": "Enumerate the people in a group ('all advisors in X').",
    "hierarchy": "The group under someone, nested by team (\"X's team\").",
    "reverse_hierarchy": "The person above someone (\"who is X's BM\").",
    "ancestry": "Every level above someone (\"the full hierarchy above X\").",
    "advisor_metric": "ONE metric for one person ('connects of X').",
    "advisor_profile": "One person's own record ('tell me about X').",
    "attendance_filter": "Advisors filtered by attendance status.",
    "entity_summary": "A group's aggregate metrics ('how is X doing').",
    "leaderboard": "A metric ranking ('top 5 by revenue').",
}


def detect_level(q: str) -> str | None:
    """The level a ranking is over, from the words the user used.

    Token-aware (F9's class): "team" is inside "teamwork" and "region"
    inside "regionally". Every table entry already lists its own plural
    ("teams", "advisors", "centres"), so requiring whole tokens costs no
    supported phrasing.
    """
    for level, keywords in LEVEL_KEYWORDS.items():
        if token_match.contains_any(q, keywords):
            return level
    return None


def detect_reverse_level(q: str) -> str:
    for level, pattern in REVERSE_LEVEL_PATTERNS:
        if pattern.search(q):
            return level
    return DEFAULT_REVERSE_LEVEL
