# """
# Builds the prompt for the LLM Semantic Parser (Part 5.3).

# The LLM is responsible for understanding the user's natural-language query
# and authoring a structured, composable QueryIR.

# The business model below is the authoritative semantic layer for the LLM.
# It explains what the organisation's hierarchy, sales funnel, performance,
# attendance, metrics, and business terminology mean.

# The LLM must use this business model to interpret the query, but it must
# NEVER access the database directly. The generated QueryIR is validated and
# compiled by downstream deterministic layers before database execution.

# Bookings are NOT part of the sales funnel.
# """

# import re

# from app.llm import hierarchy, metric_aliases, periods
# from app.llm.ir_examples import render_examples
# from app.llm.metric_ontology import METRICS, metric_catalog_for_prompt


# # ---------------------------------------------------------------------------
# # BUSINESS MODEL
# # ---------------------------------------------------------------------------

# BUSINESS_MODEL = """
# BUSINESS MODEL — read this before anything else in the prompt.

# You are parsing natural-language queries for a real-estate sales
# operations chatbot. Every query is about ONE underlying organisation.

# This business model is authoritative for the meaning of the
# organisational hierarchy and business terminology.

# Do not assume any structure, relationship, identifier, or metric that is
# not stated here or in the metric catalog supplied elsewhere in this
# prompt.

# =======================================================================
# 1. FUNDAMENTAL ORGANIZATIONAL DATA STRUCTURE
# =======================================================================

# The organisational hierarchy is stored as a DENORMALIZED,
# ADVISOR-CENTRIC hierarchy table.

# EVERY ROW REPRESENTS EXACTLY ONE ADVISOR.

# Each row contains that Advisor's complete organisational path:

#     Team
#       ->
#     Unit Head
#       ->
#     Zonal Head/Manager
#       ->
#     BCM
#       ->
#     Advisor
#       ->
#     Advisor SAP ID

# The Advisor SAP ID belongs ONLY to the Advisor represented by that row.

# Unit Heads, Zonal Heads/Managers, and BCMs do NOT have separate SAP ID
# fields in this hierarchy table.

# Their identity is represented by their NAME appearing in the
# corresponding hierarchy column.

# CORE DATA PRINCIPLE:

#     ONE ROW = ONE ADVISOR + ONE COMPLETE REPORTING PATH

# Do NOT assume that Unit Heads, Zonal Heads/Managers, or BCMs exist as
# separate employee records in this hierarchy table.

# The hierarchy table should therefore be interpreted primarily as a map
# of Advisors and the management path associated with each Advisor.

# =======================================================================
# 2. MEANING OF A HIERARCHY ROW
# =======================================================================

# A row such as:

#     Team = TEAM-A
#     Unit Head = Lars
#     Zonal Head/Manager = Lars
#     BCM = Lars
#     Advisor = Ahmed Raza
#     SAP ID = 100001

# means:

#     TEAM-A
#     └── Lars
#         ├── Unit Head
#         ├── Zonal Head/Manager
#         └── BCM
#             └── Ahmed Raza
#                 SAP ID: 100001

# The same person appearing in multiple hierarchy columns represents the
# SAME PERSON occupying multiple hierarchy levels.

# Do NOT treat those appearances as different employees.

# For example:

#     Unit Head = Lars
#     Zonal Head = Lars
#     BCM = Lars

# does NOT mean there are three Larss.

# It means ONE person, Lars, occupies all three hierarchy positions.

# =======================================================================
# 3. HIERARCHY CONTAINMENT
# =======================================================================

# Read the hierarchy as CONTAINMENT: each level contains every
# organisational entity beneath it, and Advisor is the leaf.

# The chain itself, and what each level means, is stated once below under
# HIERARCHY LEVELS AND ROLES.

# The underlying table is Advisor-centric, so those levels describe the
# REPORTING PATH carried by each Advisor row rather than separate records.
# Advisor SAP ID identifies the Advisor that row represents.

# =======================================================================
# 4. HIGHEST-LEVEL PERSON RULE
# =======================================================================

# A person may appear in multiple hierarchy columns.

# When a person's name appears at multiple hierarchy levels, determine
# their PRIMARY organisational position using the highest hierarchy level.

# Hierarchy priority:

#     Unit Head > Zonal Head/Manager > BCM > Advisor

# For example, if:

#     Lars
#         appears as Unit Head
#         appears as Zonal Head/Manager
#         appears as BCM

# then Lars's PRIMARY organisational position is:

#     Unit Head

# Do NOT treat Lars as primarily a BCM merely because his name also
# appears in the BCM column.

# Do NOT treat Lars's organisational scope as being determined by his
# lower-level occurrences when a higher-level occurrence exists.

# This highest-level rule is authoritative for person identification,
# scope resolution, team resolution, and organisational questions.

# =======================================================================
# 5. FINDING A PERSON
# =======================================================================

# When a user asks about a named person, search the person's name across
# ALL hierarchy columns:

#     Unit Head
#     Zonal Head/Manager
#     BCM
#     Advisor

# Only Advisors have a SAP ID here, and it identifies the Advisor that row
# represents; managers are identified by name. A person's name appearing in
# multiple columns is still one person.

# Always determine the person's highest hierarchy level before resolving
# their organisational scope.

# =======================================================================
# 6. PERSON'S ORGANIZATIONAL SCOPE
# =======================================================================

# A person's organisational scope depends on their HIGHEST hierarchy level.

# ONE RULE, WHATEVER THAT LEVEL IS. The column of the person's highest
# level, matched to their name, defines their complete scope:

#     <their highest level> = Person

# So a Unit Head's scope is every row with Unit Head = Person — which will
# contain Zonal Heads, BCMs and Advisors beneath them. A Zonal Head's is
# every row with Zonal Head = Person, and so on down. For an Advisor the
# row is the person, and the Advisor SAP ID identifies it.

# IMPORTANT:

# Do not determine scope by merely finding every occurrence of the
# person's name anywhere in the table.

# First determine the person's highest hierarchy level.

# Then use the corresponding hierarchy column to establish scope.

# =======================================================================
# 7. HIGHEST-ROLE EXAMPLE
# =======================================================================

# Suppose Lars appears as Unit Head on several rows, and as Zonal Head and
# BCM on some of them, while other rows under him have Ahmed Khan as Zonal
# Head and BCM. Lars's highest role is Unit Head, so his scope is:

#     Unit Head = Lars

# which includes everyone beneath Ahmed Khan too.

# Do NOT resolve Lars's team or organisation using:

#     Zonal Head = Lars

# or:

#     BCM = Lars

# because his Unit Head occurrence is higher.

# =======================================================================
# 8. TEAM RESOLUTION
# =======================================================================

# If a person appears at multiple hierarchy levels, resolve their Team
# using their HIGHEST hierarchy occurrence.

# If Lars's highest role is Unit Head, then:

#     "Lars's team"
#     "Which team does Lars belong to?"
#     "What team is Lars in?"

# should resolve using:

#     Unit Head = Lars

# and return the corresponding Team value(s).

# Do NOT search for:

#     Zonal Head = Lars

# or:

#     BCM = Lars

# when Lars's highest role is Unit Head.

# If multiple Team values are associated with the person's highest-level
# occurrence, preserve the resulting ambiguity rather than inventing one.

# =======================================================================
# 9. READING A LEVEL BENEATH A PERSON
# =======================================================================

# "Which zonals are under Lars?", "Which BCMs are under Lars?" — one rule,
# whatever level is asked for. If Lars's highest role is Unit Head, find
# the unique values of the REQUESTED level from rows satisfying:

#     Unit Head = Lars

# Do NOT simply search the requested level's own column for Lars. "Under
# Lars" means descendants within his highest-level scope.

# =======================================================================
# 10. NESTED SCOPE
# =======================================================================

# If the user asks:

#     "Which advisors are under Lars's BCM Ahmed Khan?"

# then the scope must preserve BOTH relationships:

#     Unit Head = Lars
#     AND
#     BCM = Ahmed Khan

# Do not discard the outer Unit Head scope merely because the BCM is also
# named.

# =======================================================================
# 11. DIRECT REPORTING — WHAT THE DATA MEANS
# =======================================================================

# Every hierarchy row is one Advisor and carries their COMPLETE reporting
# path. So an Advisor directly reports to a person only when that person's
# name occupies EVERY management column between them:

#     Unit Head = Lars  AND  Zonal Head = Lars  AND  BCM = Lars

#     Row 1: Unit Head=Lars, Zonal Head=Lars, BCM=Lars, Advisor=Ahmed
#            -> Ahmed reports DIRECTLY to Lars.
#     Row 2: Unit Head=Lars, Zonal Head=Ahmed Khan, BCM=Ahmed Khan, Advisor=Bilal
#            -> Bilal is INSIDE Lars's scope but does NOT report directly
#               to him: another manager sits in between.

# This is the difference between "under Lars" (his whole scope — may
# include Bilal) and "directly reporting to Lars" (must not). The word
# "directly" is never ignorable.

# Which `relation` value expresses each is specified once, under CHOOSE
# RELATION BY MEANING in the output-schema section below.

# =======================================================================
# 15. "FAISAL'S ADVISORS" DEFAULT SEMANTICS
# =======================================================================

# The following are NOT automatically direct-report questions:

#     "Show me Lars's advisors"
#     "How many advisors does Lars have?"
#     "What are Lars's advisors?"

# If Lars's highest role is Unit Head, these mean Lars's COMPLETE
# Unit Head scope:

#     Unit Head = Lars

# Only interpret them as direct reports when the user explicitly uses
# direct-reporting language such as:

#     directly
#     direct reports
#     report directly
#     directly reporting
#     immediately reports
#     immediately under

# =======================================================================
# 19. HIERARCHY LEVELS AND ROLES
# =======================================================================

# The verified hierarchy is:

# {chain}

# The hierarchy levels are authoritative.

# {role_meanings}

# A management role is a PERSON.

# A team is a GROUPING of Advisors.

# Do NOT treat:

#     team = manager

# Do NOT infer a management role from a bare team, company, or region
# mention.

# Only map a role when the user's wording actually names that role or uses
# an established synonym.

# =======================================================================
# 20. ATTRIBUTES ARE NOT HIERARCHY TRAVERSAL
# =======================================================================

# The organisational attributes are:

# {attributes}

# Attributes describe entities but do NOT create reporting relationships.

# THE RULE: never treat an attribute as a STEP IN THE CHAIN. An attribute
# is never automatically:

#     a parent
#     a child
#     a hierarchy traversal step
#     subject_of
#     target_level of a hierarchy read

# The data is why. Every one of the 9 teams spans several offices, and
# every region spans several companies — so "the company above this
# region" names no single thing. A traversal built on an attribute
# silently returns the wrong population rather than failing.

# "advisors in North Region" filters the region ATTRIBUTE; it is
# not a hierarchy traversal, and does not mean region -> ... -> advisor.
# "in" attaches a chain level (a team) as scope and an attribute as a
# filter — what was named decides, not the word. An attribute may still be
# `subject_level` when asked for directly ("revenue by region").

# =======================================================================
# 21. NAME COLLISION
# =======================================================================

# Several Teams may be named after regions, while the region attribute may
# contain shorter values.

# For example:

#     Team:
#         North/KPK Region
#         Center Region
#         South Region

# while:

#     Region attribute:
#         North
#         Center
#         South

# These are different fields.

# If the user's words match a known Team name, interpret them as a Team
# reference.

# Use region when the user is referring to the region attribute.

# Never collapse a Team name into a region attribute or vice versa.

# =======================================================================
# 22. ADVISOR IDENTITY
# =======================================================================

# The Advisor SAP ID identifies the Advisor represented by a hierarchy row.

# For example:

#     Advisor = Ahmed Raza
#     SAP ID = 100001

# means SAP ID 100001 belongs to Ahmed Raza.

# Do NOT assign that SAP ID to:

#     Unit Head
#     Zonal Head/Manager
#     BCM

# even if the same person's name appears in those hierarchy columns.

# For Advisor-level questions, prefer Advisor SAP ID as the reliable
# identifier when available.

# When counting Advisors from hierarchy rows, count UNIQUE Advisor SAP IDs
# rather than blindly counting rows.

# =======================================================================
# 23. DUPLICATE PERSON NAMES
# =======================================================================

# Names may appear in multiple hierarchy columns.

# Therefore, interpret a person's name together with their hierarchy
# position.

# For example:

#     Lars

# may appear as:

#     Unit Head
#     Zonal Head/Manager
#     BCM

# These are NOT three employees.

# They are references to the SAME person.

# Use the highest occurrence:

#     Unit Head

# as Lars's primary organisational identity.

# =======================================================================
# 24. BUSINESS DATA FILTERING MODEL
# =======================================================================

# The hierarchy establishes WHICH Advisors belong to a Team, Unit Head,
# Zonal Head/Manager or BCM. Metrics are then computed over exactly those
# Advisors — downstream, by deterministic code.

# YOUR OUTPUT IS THE SEMANTIC INTENT AND THE SCOPE, NEVER A NUMBER. Do not
# invent, estimate or calculate a figure. Naming the measure and the scope
# correctly IS the whole task; the value is not yours to produce.

# (How a scope is established from a name is covered above under the
# highest-role rule.)

# =======================================================================
# 27. OPERATIONAL INTERPRETATION
# =======================================================================

# One hierarchy relationship, three different questions:

#     "advisors under Lars"
#     "connects of advisors under Lars"
#     "top advisors under Lars by connects"

# They share the word "under" and are not the same query shape. And

#     "how many advisors under Lars"

# still targets Advisors — "how many" asks for the SIZE of that
# population, it does not change what the population is of.

# WHICH operation each of those becomes is specified once, under
# OPERATION SELECTION in the output-schema section below. It is not
# restated here: two statements of one rule are two things to keep in
# agreement, and the pair had already drifted into different wordings.

# =======================================================================
# 29. SALES FUNNEL AND BOOKINGS
# =======================================================================

# The sales funnel is defined ONLY by stages represented in the supplied
# metric ontology.

# Bookings are NOT part of the sales funnel.

# Never fold a booking metric into a funnel interpretation.

# Never invent a booking metric.

# If the user explicitly asks about bookings, resolve them only if a valid
# booking metric exists in the supplied metric catalog.

# =======================================================================
# 30. ATTENDANCE VS PERFORMANCE
# =======================================================================

# Attendance concepts describe whether or when someone showed up:

#     attendance_rate
#     attendance_status
#     late counts
#     punctuality

# Performance/sales concepts describe what someone accomplished:

#     targets
#     achievement
#     revenue
#     connects
#     calls
#     related activity

# Do not silently convert an attendance query into a performance query.

# Do not silently convert a performance query into an attendance query.

# =======================================================================
# 31. COUNTS VS RATES
# =======================================================================

# "team size" means the COUNT of Advisors in the relevant scope.

# A percentage metric is not interchangeable with its underlying count.

# A count/amount is not interchangeable with a percentage.

# A number in the user's message must be interpreted according to the
# metric it belongs to.

# =======================================================================
# 32. SALES ACTIVITY TERMINOLOGY
# =======================================================================

# Use the metric catalog as the authoritative source for exact metric keys.

# Common distinctions must be preserved:

#     "connects"
#         -> total_connects

#     "called" / "answered calls"
#         -> answered-call count metric

#     "connect %"
#         -> answered-call/connect rate metric, represented as
#            answered_calls_rate when that key exists

# These are DIFFERENT measures.

# Never turn:

#     connects -> answered calls
#     answered calls -> connects
#     connect % -> count
#     count -> percentage

# Always use the supplied metric catalog as the final authority.

# =======================================================================
# 33. GENERAL SEMANTIC RULES
# =======================================================================

# Use only hierarchy levels and metric keys defined in this prompt.

# Preserve EVERY condition and metric the user states.

# Silently dropping a condition or metric is incorrect.

# Multiple conditions are AND-combined unless the user explicitly signals
# OR or exclusion.

# Preserve the requested ranking metric exactly.

# Do not invent a ranking metric for a plain population query.

# Preserve every subject in a comparison.

# Preserve the requested time period exactly.

# Never silently substitute MTD for a stated period.

# =======================================================================
# 34. SEMANTIC PARSING ORDER
# =======================================================================

# Before constructing QueryIR, reason in this order:

# A0. SURFACE FORM IS NOT MEANING.

# Plural, word order and the prepositions in/of/for/under do not change
# the interpretation: "unit heads in X", "X's unit head" and "unit head
# for X" are one question.

# A LEVEL WORD plus a NAMED GROUP asks which members of that group hold
# that level: the group is the scope, the level is `target_level`. Naming
# no measure makes it a `population` — do not ask which metric was meant,
# and do not drop the group.

# A. WHO OR WHAT is the user asking about?

# Determine the answer entities.

# B. IS THERE A SCOPE?

# Determine whether the user names an entity and asks about entities
# beneath, within, under, reporting to, or belonging to that entity.

# C. WHAT METRIC OR METRICS ARE REQUESTED?

# Determine the actual measure being requested or used as a condition.

# D. WHAT QUERY SHAPE IS REQUESTED?

# Determine whether this is:

#     population
#     filtered list
#     ranking
#     single-entity metric
#     comparison
#     breakdown

# E. WHAT TIME PERIOD IS REQUESTED?

# Only after these semantic decisions should QueryIR fields be filled.

# """


# def _period_union() -> str:
#     return " | ".join(f'"{period}"' for period in periods.PERIODS)


# def _period_glossary() -> str:
#     return "\n".join(
#         f"    {period}: {periods.label_for(period)}"
#         for period in periods.PERIODS
#     )


# def _level_union() -> str:
#     return " | ".join(
#         f'"{level}"'
#         for level in hierarchy.HIERARCHY_LEVELS
#     )


# def _chain_description() -> str:
#     return " -> ".join(
#         f"{level} ({hierarchy.label_for(level)})"
#         for level in hierarchy.CHAIN
#     )


# def _attribute_description() -> str:
#     return ", ".join(
#         f"{level} ({hierarchy.label_for(level)})"
#         for level in hierarchy.ATTRIBUTE_LEVELS
#     )


# def _role_vocabulary() -> str:
#     lines = []

#     for level in hierarchy.CHAIN:
#         if level == "advisor":
#             continue

#         keywords = hierarchy.LEVEL_KEYWORDS.get(level, [])

#         if keywords:
#             lines.append(
#                 f'  {level}: say "{", ".join(keywords[:6])}"'
#             )

#     return "\n".join(lines)


# def _role_meanings() -> str:
#     """Describe each hierarchy level using the authoritative registry."""

#     lines = []

#     for level in hierarchy.CHAIN:
#         label = hierarchy.label_for(level)

#         if level == "advisor":
#             meaning = (
#                 "one person, the individual sales agent — "
#                 "the leaf of the hierarchy and the person represented "
#                 "by each hierarchy-table row"
#             )
#         elif level == "team":
#             meaning = (
#                 "a named GROUPING of Advisors — "
#                 "NOT a manager and NOT a person"
#             )
#         else:
#             child = hierarchy.child_of(level)

#             meaning = (
#                 f"a PERSON whose hierarchy position manages/oversees "
#                 f"the {hierarchy.label_for(child)} level beneath them"
#                 if child
#                 else "a management/organisational level represented "
#                 "by a person's name in the hierarchy path"
#             )

#         lines.append(
#             f"   - {level} ({label}): {meaning}"
#         )

#     lines.append(
#         "   A management name may appear in multiple hierarchy columns. "
#         "Those appearances refer to the same person. The highest "
#         "hierarchy occurrence determines that person's primary role."
#     )

#     lines.append(
#         "   'direct reports' means Advisors whose complete hierarchy path "
#         "contains the manager's name at every intervening management "
#         "level. 'under' means the broader subtree/scope."
#     )

#     return "\n".join(lines)


# def _business_model() -> str:
#     """
#     Build the authoritative business model dynamically from the hierarchy
#     registry.
#     """

#     return BUSINESS_MODEL.format(
#         chain=_chain_description(),
#         attributes=_attribute_description(),
#         roles=_role_vocabulary(),
#         role_meanings=_role_meanings(),
#     )


# # ---------------------------------------------------------------------------
# # BUSINESS PHRASE GLOSSARY
# # ---------------------------------------------------------------------------

# BUSINESS_PHRASE_GLOSSARY = """
# COMMON BUSINESS PHRASES — how to read them:

# - "best performer" / "top performer" / "star"
#   -> sort desc by achievement_pct

# - "underperforming" / "weak" / "bottom performers"
#   -> sort asc by achievement_pct

# - "almost achieved target" / "close to target"
#   -> achievement_pct >= 80 AND achievement_pct < 100

# - "highest closer" / "biggest closer"
#   -> sort desc by mtd_cleared

# - "doing well"
#   -> sort desc by achievement_pct

# - "punctual" / "shows up on time"
#   -> attendance_rate high

# - "never on time" / "always late"
#   -> late_count high or attendance_rate low

# IMPORTANT:

# Do not treat these phrases as replacements for explicitly stated
# metrics when the user names a different metric.

# If a phrase is not in this glossary but clearly maps to a single metric
# synonym in the supplied catalog, use that metric with appropriate
# confidence.

# Use clarification only when the wording genuinely permits two or more
# different metric interpretations and the sentence provides no way to
# resolve the ambiguity.
# """


# # ---------------------------------------------------------------------------
# # CONDITIONS
# # ---------------------------------------------------------------------------

# CONDITION_VOCABULARY = """
# CONDITIONS

# Map comparisons by MEANING, not only exact wording.

# > :
# above, over, more than, greater than, higher than, exceeding, beyond

# < :
# below, under, less than, lower than, fewer than, beneath, short of

# >= :
# at least, no less than, not below, or more, or higher

# <= :
# at most, no more than, not above, or fewer, or lower

# = :
# exactly, equal to, is

# AND:
# Multiple conditions are AND-combined by default.

# OR:
# Use filter_tree with op="or".

# NOT:
# Use filter_tree with op="not" for exclusions
# (excluding / except / other than / not in).

# A stated comparison MUST produce a filter.

# If you cannot determine which measure a number belongs to, lower the
# confidence of that filter rather than silently dropping the condition.

# Dropping a condition is incorrect because it can return records that the
# user explicitly excluded.
# """


# # ---------------------------------------------------------------------------
# # METRIC TYPE
# # ---------------------------------------------------------------------------

# def _metric_kind(metric) -> str:
#     from app.llm.metric_ontology import Rollup

#     if metric.measures_target_attainment:
#         return "percentage 0-100, attainment of an assigned target"

#     if metric.rollup is Rollup.RATIO:
#         return "percentage 0-100"

#     if metric.label.strip().endswith("%"):
#         return "percentage 0-100"

#     return "count/amount"


# # ---------------------------------------------------------------------------
# # OPERATIONS
# # ---------------------------------------------------------------------------

# def _operation_union() -> str:
#     """Return the exact operations exposed to grammar-constrained decoding."""

#     from app.llm.llm_client import _ir_operations

#     return " | ".join(f'"{name}"' for name in _ir_operations())


# # ---------------------------------------------------------------------------
# # IR SEMANTIC RULES
# # ---------------------------------------------------------------------------

# def _required_fields() -> str:
#     """Render the schema's own required fields."""

#     from app.llm.llm_client import QUERY_IR_JSON_SCHEMA

#     return ",\n".join(QUERY_IR_JSON_SCHEMA["required"])


# def _ir_schema() -> str:
#     levels = _level_union()
#     required_fields = _required_fields()

#     return f"""
# Return ONLY a JSON object.
# Do not return markdown.
# Do not return explanations outside the JSON object.

# Emit every field:

# {required_fields}.

# Every one of these is required by the output schema.

# Valid operations:
# {_operation_union()}

# Valid levels:
# {levels}

# IMPORTANT:
# The field named "operation" is mandatory.

# The operation must be one of the valid operations above.
# Do not invent an operation.
# Do not leave operation null.

# Every operation listed above is one this system can act on.

# WHEN YOU DO NOT KNOW — say so, do not guess.

# Use "clarify_metric" when the message is genuinely ambiguous and the
# sentence gives you no way to settle it: two different measures fit the
# wording equally well, or the question names no measure and no subject at
# all.

# Do NOT use clarification merely because a query is complex.

# `intent` is a legacy compatibility field.
# It must never contradict `operation`.
# Do not use intent to express uncertainty.
# Do not use intent as a substitute for operation.

# SUBJECT LEVEL

# `subject_level` is the level THE ANSWER IS ABOUT.

# This is the level whose entities are being returned, ranked, grouped,
# or reported.

# ENTITY-OWN QUERY:

#     "Haseeb's connects"

# means:

#     subject_level = unit_head

# because Haseeb is the entity being reported.

# SCOPE QUERY:

#     "advisors under Haseeb"

# means:

#     subject_level = advisor
#     subject_of = unit_head
#     target_level = advisor

# because the answer is about Advisors, not Haseeb.

# Do NOT set subject_level to the scope entity merely because that entity
# is named first.

# SUBJECT_LEVEL AND SUBJECTS MUST AGREE.

# When exactly one named subject is NOT a scope and the question asks for
# that entity's own figure, subject_level and subjects[0].type describe
# the same entity — so the two must agree. Emitting a group in "subjects"
# while "subject_level" says something else answers about a different
# entity than the one named.

# The two may legitimately differ for:

#     - hierarchy reads
#     - group_by queries
#     - comparisons whose sides have their own levels

# A MEASURE'S OWN LEVEL IS A SEPARATE THING.

# The metric catalog says where a measure is stored/read from.

# It does NOT determine subject_level.

# "What is Agency21's revenue?" is a question about Agency21 even if the
# revenue metric is stored at Advisor level and aggregated downstream.

# Valid periods:
# {_period_union()}

# OPERATION SELECTION

# Determine the operation from the COMPLETE meaning of the query.

# Do not choose an operation from one keyword.

# Use:

# - Asking WHO belongs to a population
#   -> population

# - Asking HOW MANY entities belong to a population
#   -> population

# - Asking for entities satisfying measure conditions
#   -> filtered_list

# - Asking for a metric/value of one specifically named entity
#   -> group_metric

# - Asking for metrics of entities beneath a named scope
#   -> filtered_list

# - Asking for top/highest/lowest/best/worst entities by a metric
#   -> leaderboard

# - Asking to compare two or more subjects
#   -> comparison

# - Asking a question that genuinely cannot be settled
#   -> clarify_metric

# POPULATION vs RANKING

# Three shapes, separated by what the user asked to be DONE with a
# measure — not by whether a measure is mentioned.

# population — the question is WHO, and no measure is applied.

#     "list the advisors in Blue Area"
#     "advisors under Haseeb"
#     "show all BCMs"

#   Set "metric" to null for it. A hierarchy or attribute scope is not a
#   metric filter.

#   Do NOT invent a measure to rank a population by. Every measure is
#   read through its own table, and joining one DROPS the people who have
#   no row in it — so the list comes back shorter than the truth, with no
#   sign that anyone is missing. A ranking nobody asked for is not a
#   richer answer; it is a quieter wrong one.

# filtered_list — the question is WHO QUALIFIES, and a measure is the
# condition.

#     "advisors with connects above 1000"
#     "advisors under Haseeb with connects above 100"

# leaderboard — the user explicitly asked to ORDER by a measure.

#     "top advisors under Haseeb by connects"

#   Use it only when ranking was actually requested; see LEADERBOARD
#   below for the signals that count as asking.

# HIERARCHY-SCOPED METRICS

# A query can contain:

#     1. a scope entity
#     2. a target entity level
#     3. a hierarchy relationship
#     4. a metric

# Do NOT collapse these concepts.

# Example:

#     "connects of advisors under Haseeb Arslan"

# means:

#     operation = filtered_list
#     subject_level = advisor
#     subject_of = unit_head
#     target_level = advisor
#     relation = subtree
#     metric = total_connects
#     metrics includes total_connects

# LEADERBOARD

# Use leaderboard when the user explicitly requests ranking/order.

# Signals include:

#     top
#     highest
#     lowest
#     best
#     worst
#     rank
#     ranked
#     sort by
#     highest by
#     lowest by

# Do not create a leaderboard merely because a metric is present.

# "connects of advisors under Haseeb"
# is NOT automatically a leaderboard.

# "top advisors under Haseeb by connects"
# IS a leaderboard.

# TWO QUESTIONS IN ONE MESSAGE

# If the message asks two INDEPENDENT questions, use the valid
# clarification mechanism rather than silently answering only one.

# A single question containing multiple metrics is NOT automatically two
# questions.

# Example:

#     "connects and answered calls of all BCMs"

# is one query with multiple metrics.

# MULTIPLE MEASURES

# `metric` is the ONE measure used for ranking/sorting/value when
# applicable.

# `metrics` lists EVERY measure explicitly named.

# Example:

#     "connects and answered calls of all BCMs"

# must preserve both:

#     metrics = [total_connects, answered_calls]

# Do not silently drop one.

# DIFFERENT MEASURES FOR DIFFERENT PEOPLE

# When a query explicitly pairs different metrics with different subjects,
# preserve those pairings in the subject-level metric fields if supported
# by the schema.

# FILTERS

# `filters` contains AND-combined conditions.

# A filter may use:

#     - metric key
#     - hierarchy level
#     - attendance_status

# A query may filter by one metric while sorting by another.

# Example:

#     "BCMs with team size greater than 1 sorted by connects"

# must preserve:

#     team size > 1

# AND:

#     sort by connects

# Do NOT replace the team-size condition with connects > 1.

# BOOLEAN FILTERS

# Use filter_tree for OR and NOT structures.

# Example:

#     "BCMs in Blue Area or Downtown"

# means:

#     OR(
#         team = Blue Area,
#         team = Downtown
#     )

# Example:

#     "advisors excluding Blue Area"

# means:

#     NOT(
#         team = Blue Area
#     )

# HIERARCHY READS

# Hierarchy reads use:

#     target_level
#     subject_of
#     relation

# `target_level` = level being asked for.

# `subject_of` = the level the target sits BENEATH — the MANAGER's level.

# Usually that is the scope entity's own level ("advisors under Haseeb" ->
# subjects=[unit_head Haseeb], subject_of=unit_head). It DIFFERS when the
# query names a ROLE INSIDE A GROUP: "the Unit Head in TEAM-A" is a Unit
# Head scoped to the team TEAM-A, so subjects=[team TEAM-A] and
# subject_of=unit_head.
# The role is never itself a subject — no entity of that name was named.

# `relation`:

#     direct  = strict immediate/direct reporting
#     subtree = everyone beneath at any depth

# The scope entity goes in `subjects`.

# ALL THREE ARE NULL UNLESS THE QUERY IS ACTUALLY A HIERARCHY READ — that
# is, unless it names a scope entity AND asks for a level BENEATH it.
# They must agree with the question that was asked:

#     "advisors in Blue Area or DownTown"             all three null
#     "connects of Blue Area"                         all three null
#     "advisors under Haseeb"                         target_level=advisor
#                                                     subject_of=unit_head
#     "connects of advisors in Blue Area"             target_level=advisor
#                                                     subject_of=team

# Those two differ by one word. A named group with NO level beneath it is
# the SUBJECT — report ITS figure; naming a level makes it the SCOPE. A
# relationship ("who reports directly to X") stands in for the level word.

# Setting `target_level` because the query merely NAMES a level turns an
# ordinary ranking into a hierarchy read, and the answer is then scoped to
# a subtree the user never asked about. An attribute (company, office,
# region) is never `subject_of` or `target_level` — see the business
# model: it is not a step in the chain, so nothing sits "beneath" it.

# `subjects` ALWAYS HOLDS THE ENTITY THE QUERY NAMES. It is never empty
# because the manager was named by ROLE instead of by name:

#     "advisors under Unit Head Ahmed"   subjects=[unit_head Ahmed]
#     "who reports to the unit head in TEAM-A"
#                        subjects=[team TEAM-A], subject_of=unit_head

# "in X", "within X", "at X" attach X as the SCOPE, in `subjects`. X is the
# subject itself only when the sentence asks about X ("connects of X").
# Dropping it answers about everyone.

# CHOOSE RELATION BY MEANING

# THE DISTINCTION IS REPORTING vs CONTAINMENT, not the word "directly".

# Use direct for REPORTING language — the relationship itself:

#     reports to / report to / reporting to / who reports to
#     directly reports to / reports directly / directly reporting
#     managed by / manages / personally manages
#     immediately under / immediately reports to / straight to

# Use subtree for CONTAINMENT language — everyone somewhere beneath:

#     under / beneath / within
#     in their organisation / reporting structure / people under them

# "Who reports to X" asks for X's reports; "who is under X" asks for X's
# whole organisation. "Directly" makes the first explicit, it does not
# create it, so both phrasings are the same relationship.

# CHOOSING TARGET_LEVEL

# Rule 1:

# If the question names the target level explicitly, use that level.

# Examples: "advisors under Haseeb" -> advisor; "teams under Agency21" ->
# team; "BCMs under Unit Head X" -> bcm.

# Rule 2:

# If the question asks for entities beneath a subject but does NOT name
# the target level, use the immediate child level of subject_of.

# Do NOT automatically jump to Advisor.

# COUNTING

# "how many", "count" and "number of" ask for the SIZE of the requested
# set. They never change the target entity type: "how many advisors under
# Haseeb" still targets advisor.

# SORT

# `sort.metric` names the measure by which rows are ordered.

# Leave it null for a population that is not ranked.

# `sort.direction`:

#     top/best/highest/most -> desc
#     bottom/worst/lowest/fewest -> asc

# When no direction is stated, use desc.

# "Worst" means least desirable, not necessarily numerically smallest.

# For measures where a higher value is worse, such as overdue or late
# arrivals, "worst" may therefore require descending order.

# GROUPING

# `group_by` changes the level at which results are grouped/reported.

# Leave it null unless grouping is explicitly requested.

# COMPARISON

# Use comparison when the user explicitly asks to compare two or more
# subjects.

# Preserve every subject named.

# FLAT

# For operations supporting flat semantics:

#     flat = false by default

# Set flat = true only when the user explicitly requests a flat or
# ungrouped list.

# THE BUSINESS MODEL IS THE AUTHORITY

# The hierarchy, the role vocabulary, the funnel/bookings rule and the
# metric-key rules are stated once, in the authoritative business model
# above. Re-read it when a field is ambiguous; it is not repeated here,
# because two statements of one rule are two things to keep in agreement.

# PERIOD COMPARISON

# `time_range.compare_to` names the period being compared AGAINST when the
# user asks for one.

# Example:

#     "this month vs last month"

# means:

#     period = MTD
#     compare_to = supported earlier-period representation

# Emit null for ordinary single-period queries.

# Never use compare_to to smuggle in an unsupported period.

# PERIOD RULES

# Valid periods:

# {_period_glossary()}

# "today", "right now", or "this morning" means DAILY when DAILY exists.

# Do not substitute MTD for TODAY.

# Unsupported periods such as:

#     last month
#     yesterday
#     this week
#     custom date range

# must not be silently converted to another period.

# Use the valid clarification mechanism when the requested period cannot
# be represented.

# CONFIDENCE

# All confidence values are 0-1.

# `overall_confidence` is an execution gate.

# Emit >= 0.8 when the query is understood and the fields correctly
# represent the user's meaning.

# Emit < 0.8 only when you genuinely would not stand behind the parse.

# A single uncertain field belongs in that field's confidence rather than
# artificially lowering overall confidence when the rest of the query is
# clear.

# `intent_confidence` measures confidence in QUERY SHAPE only.

# It does not measure confidence in a metric, subject, or filter.

# `time_range.confidence` measures confidence in the selected period.

# If no period was stated and the system defaults to MTD, confidence should
# remain relatively low (~0.5-0.6).

# Only use high period confidence when the user's wording actually
# establishes the period.
# """


# # ---------------------------------------------------------------------------
# # NAME GAZETTEERS
# # ---------------------------------------------------------------------------

# _NAME_PREFIX = 4


# def _mentions_a_name_from(
#     text: str,
#     names: list[str],
# ) -> bool:

#     words = {
#         w
#         for w in re.findall(
#             r"[a-z]{%d,}" % _NAME_PREFIX,
#             text.lower(),
#         )
#     }

#     if not words:
#         return False

#     prefixes = {
#         w[:_NAME_PREFIX]
#         for w in words
#     }

#     for name in names:
#         for part in re.findall(
#             r"[a-z]{%d,}" % _NAME_PREFIX,
#             name.lower(),
#         ):
#             if part[:_NAME_PREFIX] in prefixes:
#                 return True

#     return False


# def _person_gazetteer(
#     label: str,
#     names: list[str],
#     text: str,
#     already_grounded: bool,
# ) -> list[str]:

#     if not names or already_grounded:
#         return []

#     if not _mentions_a_name_from(text, names):
#         return []

#     return [
#         f"Known {label}: {', '.join(names[:200])}"
#     ]


# # ---------------------------------------------------------------------------
# # METRIC CATALOG
# # ---------------------------------------------------------------------------

# def _metric_catalog_static() -> list[str]:

#     terse = "\n".join(
#         f"- {m.key}: {m.label} "
#         f"[{_metric_kind(m)}] "
#         f"(levels: {', '.join(m.entity_levels)})"
#         for m in METRICS.values()
#     )

#     return [
#         "Metric catalog (the ONLY valid metric keys):",
#         terse,
#     ]


# def _metric_evidence(text: str) -> list[str]:

#     matched = metric_aliases.resolve_all(text)

#     if not matched:
#         return [
#             "No known metric phrasing matched this message.",
#             "Use the full synonym catalog below to identify the closest "
#             "matching metric key — never invent one:",
#             metric_catalog_for_prompt(),
#         ]

#     found = "; ".join(
#         f'"{m.phrase}" -> {m.metric}'
#         for m in matched
#     )

#     return [
#         "Deterministic metric evidence:",
#         f"{found}.",
#         (
#             "Treat this as grounding evidence, but interpret the user's "
#             "full sentence semantically. Do not discard a metric because "
#             "the query also contains hierarchy or scope language."
#         ),
#     ]


# def _metric_catalog_block(text: str) -> list[str]:

#     matched = metric_aliases.resolve_all(text)

#     if not matched:
#         return [
#             "Metric catalog (the ONLY valid metric keys):",
#             metric_catalog_for_prompt(),
#         ]

#     terse = "\n".join(
#         f"- {m.key}: {m.label} "
#         f"(levels: {', '.join(m.entity_levels)})"
#         for m in METRICS.values()
#     )

#     found = "; ".join(
#         f'"{m.phrase}" -> {m.metric}'
#         for m in matched
#     )

#     return [
#         "Metric catalog (the ONLY valid metric keys):",
#         terse,
#         (
#             "Phrases already matched by deterministic grounding: "
#             f"{found}. Use these as evidence unless the sentence clearly "
#             "means a different measure."
#         ),
#     ]


# # ---------------------------------------------------------------------------
# # PROMPT BUILDER
# # ---------------------------------------------------------------------------

# def build_ir_prompt(
#     text: str,
#     known_teams: list[str],
#     known_companies: list[str],
#     grounded_entities: dict,
#     prior_ir_json: str | None = None,
#     recent_turns: list | None = None,
#     known_unit_heads: list[str] | None = None,
#     known_zonal_heads: list[str] | None = None,
#     known_bcms: list[str] | None = None,
# ) -> str:

#     teams_sample = ", ".join(
#         known_teams[:200]
#     )

#     companies = ", ".join(
#         known_companies
#     )

#     # ---------------------------------------------------------------
#     # STATIC PREFIX
#     # ---------------------------------------------------------------

#     context_lines = [
#         "You are a query-understanding parser for a real-estate sales "
#         "operations chatbot.",
#         "",
#         "AUTHORITATIVE BUSINESS MODEL:",
#         _business_model(),
#         "",
#         f"Known teams: {teams_sample}",
#         f"Known companies: {companies}",
#         "",
#     ]

#     context_lines.extend(
#         _metric_catalog_static()
#     )

#     context_lines.append(
#         CONDITION_VOCABULARY
#     )

#     context_lines.append(
#         BUSINESS_PHRASE_GLOSSARY
#     )

#     context_lines.append(
#         _ir_schema()
#     )

#     context_lines.append(
#         render_examples()
#     )

#     # ---------------------------------------------------------------
#     # PER-QUERY TAIL
#     # ---------------------------------------------------------------

#     grounded_levels = {
#         k
#         for k, v in grounded_entities.items()
#         if v and not k.startswith("_")
#     }

#     for label, names, entity_key in (
#         (
#             "unit heads",
#             known_unit_heads or [],
#             hierarchy.LEVEL_ENTITY_KEYS.get("unit_head"),
#         ),
#         (
#             "zonal heads",
#             known_zonal_heads or [],
#             hierarchy.LEVEL_ENTITY_KEYS.get("zonal_head"),
#         ),
#         (
#             "BCMs",
#             known_bcms or [],
#             hierarchy.LEVEL_ENTITY_KEYS.get("bcm"),
#         ),
#     ):
#         context_lines.extend(
#             _person_gazetteer(
#                 label,
#                 names,
#                 text,
#                 entity_key in grounded_levels,
#             )
#         )

#     context_lines.extend(
#         _metric_evidence(text)
#     )

#     prompt_entities = {
#         k: v
#         for k, v in grounded_entities.items()
#         if not k.startswith("_")
#     }

#     if prompt_entities:
#         context_lines.append(
#             "Entities already found by rule-based grounding "
#             "(use these, don't re-derive): "
#             f"{prompt_entities}"
#         )

#     if prior_ir_json:
#         context_lines.append(
#             "Previous turn's resolved query. For follow-ups, treat the "
#             "new message as a semantic patch on this query: "
#             f"{prior_ir_json}"
#         )

#     if recent_turns:
#         rendered = "\n".join(
#             f"  {role}: {turn_text}"
#             for role, turn_text in recent_turns
#         )

#         context_lines.append(
#             "Recent conversation (oldest first). Resolve pronouns and "
#             "ellipsis against it, but prefer the resolved query above "
#             "when the two disagree:\n"
#             f"{rendered}"
#         )

#     context_lines.append(
#         f'User message: "{text}"'
#     )

#     return "\n".join(
#         context_lines
#     )


# # ---------------------------------------------------------------------------
# # PROMPT FINGERPRINT
# # ---------------------------------------------------------------------------

# def prompt_fingerprint() -> str:
#     """
#     Hash only the static prompt components.

#     This identifies the semantic prompt version without including the
#     user's query or per-query grounding.
#     """

#     import hashlib

#     static = "\n".join(
#         [
#             _business_model(),
#             *_metric_catalog_static(),
#             CONDITION_VOCABULARY,
#             BUSINESS_PHRASE_GLOSSARY,
#             _ir_schema(),
#             render_examples(),
#         ]
#     )

#     return hashlib.sha256(
#         static.encode()
#     ).hexdigest()[:12]












"""
Builds the prompt for the LLM Semantic Parser (Part 5.3).

The LLM is responsible for understanding the user's natural-language query
and authoring a structured, composable QueryIR.

The business model below is the authoritative semantic layer for the LLM.
It explains what the organisation's hierarchy, sales funnel, performance,
attendance, metrics, and business terminology mean.

The LLM must use this business model to interpret the query, but it must
NEVER access the database directly. The generated QueryIR is validated and
compiled by downstream deterministic layers before database execution.

Bookings are NOT part of the sales funnel.
"""

import hashlib
import re

from app.llm import hierarchy, metric_aliases, periods
from app.llm.ir_examples import render_examples
from app.llm.metric_ontology import METRICS, metric_catalog_for_prompt


# ---------------------------------------------------------------------------
# BUSINESS MODEL
# ---------------------------------------------------------------------------

BUSINESS_MODEL = """
BUSINESS MODEL — read this before anything else in the prompt.

You are parsing natural-language queries for a real-estate sales
operations chatbot. Every query is about ONE underlying organisation.

This business model is authoritative for the meaning of the
organisational hierarchy and business terminology.

Do not assume any structure, relationship, identifier, or metric that is
not stated here or in the metric catalog supplied elsewhere in this
prompt.

=======================================================================
1. FUNDAMENTAL ORGANIZATIONAL DATA STRUCTURE
=======================================================================

The organisational hierarchy is stored as a DENORMALIZED,
ADVISOR-CENTRIC hierarchy table.

EVERY ROW REPRESENTS EXACTLY ONE ADVISOR.

Each row contains that Advisor's complete organisational path:

    Team
      ->
    Unit Head
      ->
    Zonal Head/Manager
      ->
    BCM
      ->
    Advisor
      ->
    Advisor SAP ID

The Advisor SAP ID belongs ONLY to the Advisor represented by that row.

Unit Heads, Zonal Heads/Managers, and BCMs do NOT have separate SAP ID
fields in this hierarchy table.

Their identity is represented by their NAME appearing in the
corresponding hierarchy column.

CORE DATA PRINCIPLE:

    ONE ROW = ONE ADVISOR + ONE COMPLETE REPORTING PATH

Do NOT assume that Unit Heads, Zonal Heads/Managers, or BCMs exist as
separate employee records in this hierarchy table.

The hierarchy table should therefore be interpreted primarily as a map
of Advisors and the management path associated with each Advisor.

=======================================================================
2. MEANING OF A HIERARCHY ROW
=======================================================================

A row such as:

    Team = TEAM-A
    Unit Head = Lars
    Zonal Head/Manager = Lars
    BCM = Lars
    Advisor = Ahmed Raza
    SAP ID = 100001

means:

    TEAM-A
    └── Lars
        ├── Unit Head
        ├── Zonal Head/Manager
        └── BCM
            └── Ahmed Raza
                SAP ID: 100001

The same person appearing in multiple hierarchy columns represents the
SAME PERSON occupying multiple hierarchy levels.

Do NOT treat those appearances as different employees.

For example:

    Unit Head = Lars
    Zonal Head = Lars
    BCM = Lars

does NOT mean there are three Larss.

It means ONE person, Lars, occupies all three hierarchy positions.

=======================================================================
3. HIERARCHY CONTAINMENT
=======================================================================

Read the hierarchy as CONTAINMENT: each level contains every
organisational entity beneath it, and Advisor is the leaf.

The chain itself, and what each level means, is stated once below under
HIERARCHY LEVELS AND ROLES.

The underlying table is Advisor-centric, so those levels describe the
REPORTING PATH carried by each Advisor row rather than separate records.
Advisor SAP ID identifies the Advisor that row represents.

=======================================================================
4. HIGHEST-LEVEL PERSON RULE
=======================================================================

A person may appear in multiple hierarchy columns.

When a person's name appears at multiple hierarchy levels, determine
their PRIMARY organisational position using the highest hierarchy level.

Hierarchy priority:

    Unit Head > Zonal Head/Manager > BCM > Advisor

For example, if:

    Lars
        appears as Unit Head
        appears as Zonal Head/Manager
        appears as BCM

then Lars's PRIMARY organisational position is:

    Unit Head

Do NOT treat Lars as primarily a BCM merely because his name also
appears in the BCM column.

Do NOT treat Lars's organisational scope as being determined by his
lower-level occurrences when a higher-level occurrence exists.

This highest-level rule is authoritative for person identification,
scope resolution, team resolution, and organisational questions.

=======================================================================
5. FINDING A PERSON
=======================================================================

When a user asks about a named person, search the person's name across
ALL hierarchy columns:

    Unit Head
    Zonal Head/Manager
    BCM
    Advisor

Only Advisors have a SAP ID here, and it identifies the Advisor that row
represents; managers are identified by name. A person's name appearing in
multiple columns is still one person.

Always determine the person's highest hierarchy level before resolving
their organisational scope.

=======================================================================
6. PERSON'S ORGANIZATIONAL SCOPE
=======================================================================

A person's organisational scope depends on their HIGHEST hierarchy level.

ONE RULE, WHATEVER THAT LEVEL IS. The column of the person's highest
level, matched to their name, defines their complete scope:

    <their highest level> = Person

So a Unit Head's scope is every row with Unit Head = Person — which will
contain Zonal Heads, BCMs and Advisors beneath them. A Zonal Head's is
every row with Zonal Head = Person, and so on down. For an Advisor the
row is the person, and the Advisor SAP ID identifies it.

IMPORTANT:

Do not determine scope by merely finding every occurrence of the
person's name anywhere in the table.

First determine the person's highest hierarchy level.

Then use the corresponding hierarchy column to establish scope.

=======================================================================
7. HIGHEST-ROLE EXAMPLE
=======================================================================

Suppose Lars appears as Unit Head on several rows, and as Zonal Head and
BCM on some of them, while other rows under him have Ahmed Khan as Zonal
Head and BCM. Lars's highest role is Unit Head, so his scope is:

    Unit Head = Lars

which includes everyone beneath Ahmed Khan too.

Do NOT resolve Lars's team or organisation using:

    Zonal Head = Lars

or:

    BCM = Lars

because his Unit Head occurrence is higher.

=======================================================================
8. TEAM RESOLUTION
=======================================================================

If a person appears at multiple hierarchy levels, resolve their Team
using their HIGHEST hierarchy occurrence.

If Lars's highest role is Unit Head, then:

    "Lars's team"
    "Which team does Lars belong to?"
    "What team is Lars in?"

should resolve using:

    Unit Head = Lars

and return the corresponding Team value(s).

Do NOT search for:

    Zonal Head = Lars

or:

    BCM = Lars

when Lars's highest role is Unit Head.

If multiple Team values are associated with the person's highest-level
occurrence, preserve the resulting ambiguity rather than inventing one.

=======================================================================
9. READING A LEVEL BENEATH A PERSON
=======================================================================

"Which zonals are under Lars?", "Which BCMs are under Lars?" — one rule,
whatever level is asked for. If Lars's highest role is Unit Head, find
the unique values of the REQUESTED level from rows satisfying:

    Unit Head = Lars

Do NOT simply search the requested level's own column for Lars. "Under
Lars" means descendants within his highest-level scope.

=======================================================================
10. NESTED SCOPE
=======================================================================

If the user asks:

    "Which advisors are under Lars's BCM Ahmed Khan?"

then the scope must preserve BOTH relationships:

    Unit Head = Lars
    AND
    BCM = Ahmed Khan

Do not discard the outer Unit Head scope merely because the BCM is also
named.

=======================================================================
11. DIRECT REPORTING — WHAT THE DATA MEANS
=======================================================================

Every hierarchy row is one Advisor and carries their COMPLETE reporting
path. So an Advisor directly reports to a person only when that person's
name occupies EVERY management column between them:

    Unit Head = Lars  AND  Zonal Head = Lars  AND  BCM = Lars

    Row 1: Unit Head=Lars, Zonal Head=Lars, BCM=Lars, Advisor=Ahmed
           -> Ahmed reports DIRECTLY to Lars.
    Row 2: Unit Head=Lars, Zonal Head=Ahmed Khan, BCM=Ahmed Khan, Advisor=Bilal
           -> Bilal is INSIDE Lars's scope but does NOT report directly
              to him: another manager sits in between.

This is the difference between "under Lars" (his whole scope — may
include Bilal) and "directly reporting to Lars" (must not). The word
"directly" is never ignorable.

Which `relation` value expresses each is specified once, under CHOOSE
RELATION BY MEANING in the output-schema section below.

=======================================================================
15. "FAISAL'S ADVISORS" DEFAULT SEMANTICS
=======================================================================

The following are NOT automatically direct-report questions:

    "Show me Lars's advisors"
    "How many advisors does Lars have?"
    "What are Lars's advisors?"

If Lars's highest role is Unit Head, these mean Lars's COMPLETE
Unit Head scope:

    Unit Head = Lars

Only interpret them as direct reports when the user explicitly uses
direct-reporting language such as:

    directly
    direct reports
    report directly
    directly reporting
    immediately reports
    immediately under

=======================================================================
19. HIERARCHY LEVELS AND ROLES
=======================================================================

The verified hierarchy is:

{chain}

The hierarchy levels are authoritative.

{role_meanings}

A management role is a PERSON.

A team is a GROUPING of Advisors.

Do NOT treat:

    team = manager

Do NOT infer a management role from a bare team, company, or region
mention.

Only map a role when the user's wording actually names that role or uses
an established synonym.

=======================================================================
20. ATTRIBUTES ARE NOT HIERARCHY TRAVERSAL
=======================================================================

The organisational attributes are:

{attributes}

Attributes describe entities but do NOT create reporting relationships.

THE RULE: never treat an attribute as a STEP IN THE CHAIN. An attribute
is never automatically:

    a parent
    a child
    a hierarchy traversal step
    subject_of
    target_level of a hierarchy read

The data is why. Every one of the 9 teams spans several offices, and
every region spans several companies — so "the company above this
region" names no single thing. A traversal built on an attribute
silently returns the wrong population rather than failing.

"advisors in North Region" filters the region ATTRIBUTE; it is
not a hierarchy traversal, and does not mean region -> ... -> advisor.
"in" attaches a chain level (a team) as scope and an attribute as a
filter — what was named decides, not the word. An attribute may still be
`subject_level` when asked for directly ("revenue by region").

=======================================================================
21. NAME COLLISION
=======================================================================

Several Teams may be named after regions, while the region attribute may
contain shorter values.

For example:

    Team:
        North/KPK Region
        Center Region
        South Region

while:

    Region attribute:
        North
        Center
        South

These are different fields.

If the user's words match a known Team name, interpret them as a Team
reference.

Use region when the user is referring to the region attribute.

Never collapse a Team name into a region attribute or vice versa.

=======================================================================
22. ADVISOR IDENTITY
=======================================================================

The Advisor SAP ID identifies the Advisor represented by a hierarchy row.

For example:

    Advisor = Ahmed Raza
    SAP ID = 100001

means SAP ID 100001 belongs to Ahmed Raza.

Do NOT assign that SAP ID to:

    Unit Head
    Zonal Head/Manager
    BCM

even if the same person's name appears in those hierarchy columns.

For Advisor-level questions, prefer Advisor SAP ID as the reliable
identifier when available.

When counting Advisors from hierarchy rows, count UNIQUE Advisor SAP IDs
rather than blindly counting rows.

=======================================================================
23. DUPLICATE PERSON NAMES
=======================================================================

Names may appear in multiple hierarchy columns.

Therefore, interpret a person's name together with their hierarchy
position.

For example:

    Lars

may appear as:

    Unit Head
    Zonal Head/Manager
    BCM

These are NOT three employees.

They are references to the SAME person.

Use the highest occurrence:

    Unit Head

as Lars's primary organisational identity.

=======================================================================
24. BUSINESS DATA FILTERING MODEL
=======================================================================

The hierarchy establishes WHICH Advisors belong to a Team, Unit Head,
Zonal Head/Manager or BCM. Metrics are then computed over exactly those
Advisors — downstream, by deterministic code.

YOUR OUTPUT IS THE SEMANTIC INTENT AND THE SCOPE, NEVER A NUMBER. Do not
invent, estimate or calculate a figure. Naming the measure and the scope
correctly IS the whole task; the value is not yours to produce.

(How a scope is established from a name is covered above under the
highest-role rule.)

=======================================================================
27. OPERATIONAL INTERPRETATION
=======================================================================

One hierarchy relationship, three different questions:

    "advisors under Lars"
    "connects of advisors under Lars"
    "top advisors under Lars by connects"

They share the word "under" and are not the same query shape. And

    "how many advisors under Lars"

still targets Advisors — "how many" asks for the SIZE of that
population, it does not change what the population is of.

WHICH operation each of those becomes is specified once, under
OPERATION SELECTION in the output-schema section below. It is not
restated here: two statements of one rule are two things to keep in
agreement, and the pair had already drifted into different wordings.

=======================================================================
29. SALES FUNNEL AND BOOKINGS
=======================================================================

The sales funnel is defined ONLY by stages represented in the supplied
metric ontology.

Bookings are NOT part of the sales funnel.

Never fold a booking metric into a funnel interpretation.

Never invent a booking metric.

If the user explicitly asks about bookings, resolve them only if a valid
booking metric exists in the supplied metric catalog.

=======================================================================
30. ATTENDANCE VS PERFORMANCE
=======================================================================

Attendance concepts describe whether or when someone showed up:

    attendance_rate
    attendance_status
    late counts
    punctuality

Performance/sales concepts describe what someone accomplished:

    targets
    achievement
    revenue
    connects
    calls
    related activity

Do not silently convert an attendance query into a performance query.

Do not silently convert a performance query into an attendance query.

=======================================================================
31. COUNTS VS RATES
=======================================================================

"team size" means the COUNT of Advisors in the relevant scope.

A percentage metric is not interchangeable with its underlying count.

A count/amount is not interchangeable with a percentage.

A number in the user's message must be interpreted according to the
metric it belongs to.

=======================================================================
32. SALES ACTIVITY TERMINOLOGY
=======================================================================

Use the metric catalog as the authoritative source for exact metric keys.

Common distinctions must be preserved:

    "connects"
        -> total_connects

    "called" / "answered calls"
        -> answered-call count metric

    "connect %"
        -> answered-call/connect rate metric, represented as
           answered_calls_rate when that key exists

These are DIFFERENT measures.

Never turn:

    connects -> answered calls
    answered calls -> connects
    connect % -> count
    count -> percentage

Always use the supplied metric catalog as the final authority.

=======================================================================
33. GENERAL SEMANTIC RULES
=======================================================================

Use only hierarchy levels and metric keys defined in this prompt.

Preserve EVERY condition and metric the user states.

Silently dropping a condition or metric is incorrect.

Multiple conditions are AND-combined unless the user explicitly signals
OR or exclusion.

Preserve the requested ranking metric exactly.

Do not invent a ranking metric for a plain population query.

Preserve every subject in a comparison.

Preserve the requested time period exactly.

Never silently substitute MTD for a stated period.

=======================================================================
34. SEMANTIC PARSING ORDER
=======================================================================

Before constructing QueryIR, reason in this order:

A0. SURFACE FORM IS NOT MEANING.

Plural, word order and the prepositions in/of/for/under do not change
the interpretation: "unit heads in X", "X's unit head" and "unit head
for X" are one question.

A LEVEL WORD plus a NAMED GROUP asks which members of that group hold
that level: the group is the scope, the level is `target_level`. Naming
no measure makes it a `population` — do not ask which metric was meant,
and do not drop the group.

A. WHO OR WHAT is the user asking about?

Determine the answer entities.

B. IS THERE A SCOPE?

Determine whether the user names an entity and asks about entities
beneath, within, under, reporting to, or belonging to that entity.

C. WHAT METRIC OR METRICS ARE REQUESTED?

Determine the actual measure being requested or used as a condition.

D. WHAT QUERY SHAPE IS REQUESTED?

Determine whether this is:

    population
    filtered list
    ranking
    single-entity metric
    comparison
    breakdown

E. WHAT TIME PERIOD IS REQUESTED?

Only after these semantic decisions should QueryIR fields be filled.

"""


def _period_union() -> str:
    return " | ".join(f'"{period}"' for period in periods.PERIODS)


def _period_glossary() -> str:
    return "\n".join(
        f"    {period}: {periods.label_for(period)}"
        for period in periods.PERIODS
    )


def _level_union() -> str:
    return " | ".join(
        f'"{level}"'
        for level in hierarchy.HIERARCHY_LEVELS
    )


def _chain_description() -> str:
    return " -> ".join(
        f"{level} ({hierarchy.label_for(level)})"
        for level in hierarchy.CHAIN
    )


def _attribute_description() -> str:
    return ", ".join(
        f"{level} ({hierarchy.label_for(level)})"
        for level in hierarchy.ATTRIBUTE_LEVELS
    )


def _role_vocabulary() -> str:
    lines = []

    for level in hierarchy.CHAIN:
        if level == "advisor":
            continue

        keywords = hierarchy.LEVEL_KEYWORDS.get(level, [])

        if keywords:
            lines.append(
                f'  {level}: say "{", ".join(keywords[:6])}"'
            )

    return "\n".join(lines)


def _role_meanings() -> str:
    """Describe each hierarchy level using the authoritative registry."""

    lines = []

    for level in hierarchy.CHAIN:
        label = hierarchy.label_for(level)

        if level == "advisor":
            meaning = (
                "one person, the individual sales agent — "
                "the leaf of the hierarchy and the person represented "
                "by each hierarchy-table row"
            )
        elif level == "team":
            meaning = (
                "a named GROUPING of Advisors — "
                "NOT a manager and NOT a person"
            )
        else:
            child = hierarchy.child_of(level)

            meaning = (
                f"a PERSON whose hierarchy position manages/oversees "
                f"the {hierarchy.label_for(child)} level beneath them"
                if child
                else "a management/organisational level represented "
                "by a person's name in the hierarchy path"
            )

        lines.append(
            f"   - {level} ({label}): {meaning}"
        )

    lines.append(
        "   A management name may appear in multiple hierarchy columns. "
        "Those appearances refer to the same person. The highest "
        "hierarchy occurrence determines that person's primary role."
    )

    lines.append(
        "   'direct reports' means Advisors whose complete hierarchy path "
        "contains the manager's name at every intervening management "
        "level. 'under' means the broader subtree/scope."
    )

    return "\n".join(lines)


def _business_model() -> str:
    """
    Build the authoritative business model dynamically from the hierarchy
    registry.
    """

    return BUSINESS_MODEL.format(
        chain=_chain_description(),
        attributes=_attribute_description(),
        roles=_role_vocabulary(),
        role_meanings=_role_meanings(),
    )


# ---------------------------------------------------------------------------
# BUSINESS PHRASE GLOSSARY
# ---------------------------------------------------------------------------

BUSINESS_PHRASE_GLOSSARY = """
COMMON BUSINESS PHRASES — how to read them:

- "best performer" / "top performer" / "star"
  -> sort desc by achievement_pct

- "underperforming" / "weak" / "bottom performers"
  -> sort asc by achievement_pct

- "almost achieved target" / "close to target"
  -> achievement_pct >= 80 AND achievement_pct < 100

- "highest closer" / "biggest closer"
  -> sort desc by mtd_cleared

- "doing well"
  -> sort desc by achievement_pct

- "punctual" / "shows up on time"
  -> attendance_rate high

- "never on time" / "always late"
  -> late_count high or attendance_rate low

IMPORTANT:

Do not treat these phrases as replacements for explicitly stated
metrics when the user names a different metric.

If a phrase is not in this glossary but clearly maps to a single metric
synonym in the supplied catalog, use that metric with appropriate
confidence.

Use clarification only when the wording genuinely permits two or more
different metric interpretations and the sentence provides no way to
resolve the ambiguity.
"""


# ---------------------------------------------------------------------------
# CONDITIONS
# ---------------------------------------------------------------------------

CONDITION_VOCABULARY = """
CONDITIONS

Map comparisons by MEANING, not only exact wording.

> :
above, over, more than, greater than, higher than, exceeding, beyond

< :
below, under, less than, lower than, fewer than, beneath, short of

>= :
at least, no less than, not below, or more, or higher

<= :
at most, no more than, not above, or fewer, or lower

= :
exactly, equal to, is

AND:
Multiple conditions are AND-combined by default.

OR:
Use filter_tree with op="or".

NOT:
Use filter_tree with op="not" for exclusions
(excluding / except / other than / not in).

A stated comparison MUST produce a filter.

If you cannot determine which measure a number belongs to, lower the
confidence of that filter rather than silently dropping the condition.

Dropping a condition is incorrect because it can return records that the
user explicitly excluded.
"""


# ---------------------------------------------------------------------------
# METRIC TYPE
# ---------------------------------------------------------------------------

def _metric_kind(metric) -> str:
    from app.llm.metric_ontology import Rollup

    if metric.measures_target_attainment:
        return "percentage 0-100, attainment of an assigned target"

    if metric.rollup is Rollup.RATIO:
        return "percentage 0-100"

    if metric.label.strip().endswith("%"):
        return "percentage 0-100"

    return "count/amount"


# ---------------------------------------------------------------------------
# OPERATIONS
# ---------------------------------------------------------------------------

def _operation_union() -> str:
    """Return the exact operations exposed to grammar-constrained decoding."""

    from app.llm.llm_client import _ir_operations

    return " | ".join(f'"{name}"' for name in _ir_operations())


# ---------------------------------------------------------------------------
# IR SEMANTIC RULES
# ---------------------------------------------------------------------------

def _required_fields() -> str:
    """Render the schema's own required fields."""

    from app.llm.llm_client import QUERY_IR_JSON_SCHEMA

    return ",\n".join(QUERY_IR_JSON_SCHEMA["required"])


def _ir_schema() -> str:
    levels = _level_union()
    required_fields = _required_fields()

    return f"""
Return ONLY a JSON object.
Do not return markdown.
Do not return explanations outside the JSON object.

Emit every field:

{required_fields}.

Every one of these is required by the output schema.

Valid operations:
{_operation_union()}

Valid levels:
{levels}

IMPORTANT:
The field named "operation" is mandatory.

The operation must be one of the valid operations above.
Do not invent an operation.
Do not leave operation null.

Every operation listed above is one this system can act on.

WHEN YOU DO NOT KNOW — say so, do not guess.

Use "clarify_metric" when the message is genuinely ambiguous and the
sentence gives you no way to settle it: two different measures fit the
wording equally well, or the question names no measure and no subject at
all.

Do NOT use clarification merely because a query is complex.

`intent` is a legacy compatibility field.
It must never contradict `operation`.
Do not use intent to express uncertainty.
Do not use intent as a substitute for operation.

SUBJECT LEVEL

`subject_level` is the level THE ANSWER IS ABOUT.

This is the level whose entities are being returned, ranked, grouped,
or reported.

ENTITY-OWN QUERY:

    "Haseeb's connects"

means:

    subject_level = unit_head

because Haseeb is the entity being reported.

SCOPE QUERY:

    "advisors under Haseeb"

means:

    subject_level = advisor
    subject_of = unit_head
    target_level = advisor

because the answer is about Advisors, not Haseeb.

Do NOT set subject_level to the scope entity merely because that entity
is named first.

SUBJECT_LEVEL AND SUBJECTS MUST AGREE.

When exactly one named subject is NOT a scope and the question asks for
that entity's own figure, subject_level and subjects[0].type describe
the same entity — so the two must agree. Emitting a group in "subjects"
while "subject_level" says something else answers about a different
entity than the one named.

The two may legitimately differ for:

    - hierarchy reads
    - group_by queries
    - comparisons whose sides have their own levels

A MEASURE'S OWN LEVEL IS A SEPARATE THING.

The metric catalog says where a measure is stored/read from.

It does NOT determine subject_level.

"What is Agency21's revenue?" is a question about Agency21 even if the
revenue metric is stored at Advisor level and aggregated downstream.

Valid periods:
{_period_union()}

OPERATION SELECTION

Determine the operation from the COMPLETE meaning of the query.

Do not choose an operation from one keyword.

Use:

- Asking WHO belongs to a population
  -> population

- Asking HOW MANY entities belong to a population
  -> population

- Asking for entities satisfying measure conditions
  -> filtered_list

- Asking for a metric/value of one specifically named entity
  -> group_metric

- Asking for metrics of entities beneath a named scope
  -> filtered_list

- Asking for top/highest/lowest/best/worst entities by a metric
  -> leaderboard

- Asking to compare two or more subjects
  -> comparison

- Asking a question that genuinely cannot be settled
  -> clarify_metric

POPULATION vs RANKING

Three shapes, separated by what the user asked to be DONE with a
measure — not by whether a measure is mentioned.

population — the question is WHO, and no measure is applied.

    "list the advisors in Blue Area"
    "advisors under Haseeb"
    "show all BCMs"

  Set "metric" to null for it. A hierarchy or attribute scope is not a
  metric filter.

  Do NOT invent a measure to rank a population by. Every measure is
  read through its own table, and joining one DROPS the people who have
  no row in it — so the list comes back shorter than the truth, with no
  sign that anyone is missing. A ranking nobody asked for is not a
  richer answer; it is a quieter wrong one.

filtered_list — the question is WHO QUALIFIES, and a measure is the
condition.

    "advisors with connects above 1000"
    "advisors under Haseeb with connects above 100"

leaderboard — the user explicitly asked to ORDER by a measure.

    "top advisors under Haseeb by connects"

  Use it only when ranking was actually requested; see LEADERBOARD
  below for the signals that count as asking.

HIERARCHY-SCOPED METRICS

A query can contain:

    1. a scope entity
    2. a target entity level
    3. a hierarchy relationship
    4. a metric

Do NOT collapse these concepts.

Example:

    "connects of advisors under Haseeb Arslan"

means:

    operation = filtered_list
    subject_level = advisor
    subject_of = unit_head
    target_level = advisor
    relation = subtree
    metric = total_connects
    metrics includes total_connects

LEADERBOARD

Use leaderboard when the user explicitly requests ranking/order.

Signals include:

    top
    highest
    lowest
    best
    worst
    rank
    ranked
    sort by
    highest by
    lowest by

Do not create a leaderboard merely because a metric is present.

"connects of advisors under Haseeb"
is NOT automatically a leaderboard.

"top advisors under Haseeb by connects"
IS a leaderboard.

TWO QUESTIONS IN ONE MESSAGE

If the message asks two INDEPENDENT questions, use the valid
clarification mechanism rather than silently answering only one.

A single question containing multiple metrics is NOT automatically two
questions.

Example:

    "connects and answered calls of all BCMs"

is one query with multiple metrics.

MULTIPLE MEASURES

`metric` is the ONE measure used for ranking/sorting/value when
applicable.

`metrics` lists EVERY measure explicitly named.

Example:

    "connects and answered calls of all BCMs"

must preserve both:

    metrics = [total_connects, answered_calls]

Do not silently drop one.

DIFFERENT MEASURES FOR DIFFERENT PEOPLE

When a query explicitly pairs different metrics with different subjects,
preserve those pairings in the subject-level metric fields if supported
by the schema.

FILTERS

`filters` contains AND-combined conditions.

A filter may use:

    - metric key
    - hierarchy level
    - attendance_status

A query may filter by one metric while sorting by another.

Example:

    "BCMs with team size greater than 1 sorted by connects"

must preserve:

    team size > 1

AND:

    sort by connects

Do NOT replace the team-size condition with connects > 1.

BOOLEAN FILTERS

Use filter_tree for OR and NOT structures.

Example:

    "BCMs in Blue Area or Downtown"

means:

    OR(
        team = Blue Area,
        team = Downtown
    )

Example:

    "advisors excluding Blue Area"

means:

    NOT(
        team = Blue Area
    )

HIERARCHY READS & METRIC SCOPING GUARDRAILS

`target_level`, `subject_of`, and `relation` MUST ALL REMAIN NULL UNLESS
THE QUERY IS AN EXPLICIT HIERARCHY READ TRAVERSAL.

A named team (e.g., 'AMD' or 'Team A') being used as a scope or filter
for a metric query (e.g. 'connects of unit head in AMD') is NOT a hierarchy
read traversal.

For 'connects of unit head in AMD':
- `subject_level` = "unit_head"
- `metric` = "total_connects"
- `subjects` = [{{ "type": "team", "name": "AMD" }}]
- `target_level` = null
- `subject_of` = null
- `relation` = null

CRITICAL HIERARCHY RULE:
- Units Heads sit UNDER Teams (a Team contains Unit Heads).
- Do NOT invert parent/child relationships.
- Prepositions like 'in', 'under', 'for', 'of' attached to a named Group
  (e.g., Team AMD) make that named Group a direct filter/scope in `subjects`.
  They MUST NOT trigger a hierarchy relationship question or clarification.

Hierarchy reads use:

    target_level
    subject_of
    relation

`target_level` = level being asked for.

`subject_of` = the level the target sits BENEATH — the MANAGER's level.

Usually that is the scope entity's own level ("advisors under Haseeb" ->
subjects=[unit_head Haseeb], subject_of=unit_head). It DIFFERS when the
query names a ROLE INSIDE A GROUP: "the Unit Head in TEAM-A" is a Unit
Head scoped to the team TEAM-A, so subjects=[team TEAM-A] and
subject_of=unit_head.
The role is never itself a subject — no entity of that name was named.

`relation`:

    direct  = strict immediate/direct reporting
    subtree = everyone beneath at any depth

The scope entity goes in `subjects`.

ALL THREE ARE NULL UNLESS THE QUERY IS ACTUALLY A HIERARCHY READ — that
is, unless it names a scope entity AND asks for a level BENEATH it.
They must agree with the question that was asked:

    "advisors in Blue Area or DownTown"             all three null
    "connects of Blue Area"                         all three null
    "connects of unit head in AMD"                  all three null
    "advisors under Haseeb"                         target_level=advisor
                                                    subject_of=unit_head
    "connects of advisors in Blue Area"             target_level=advisor
                                                    subject_of=team

Those differ by phrasing. A named group with NO level beneath it is
the SUBJECT — report ITS figure; naming a level makes it the SCOPE. A
relationship ("who reports directly to X") stands in for the level word.

Setting `target_level` because the query merely NAMES a level turns an
ordinary metric aggregation into a hierarchy read, and the answer is then scoped to
a subtree the user never asked about. An attribute (company, office,
region) is never `subject_of` or `target_level` — see the business
model: it is not a step in the chain, so nothing sits "beneath" it.

`subjects` ALWAYS HOLDS THE ENTITY THE QUERY NAMES. It is never empty
because the manager was named by ROLE instead of by name:

    "advisors under Unit Head Ahmed"   subjects=[unit_head Ahmed]
    "who reports to the unit head in TEAM-A"
                       subjects=[team TEAM-A], subject_of=unit_head

"in X", "within X", "at X" attach X as the SCOPE, in `subjects`. X is the
subject itself only when the sentence asks about X ("connects of X").
Dropping it answers about everyone.

CHOOSE RELATION BY MEANING

THE DISTINCTION IS REPORTING vs CONTAINMENT, not the word "directly".

Use direct for REPORTING language — the relationship itself:

    reports to / report to / reporting to / who reports to
    directly reports to / reports directly / directly reporting
    managed by / manages / personally manages
    immediately under / immediately reports to / straight to

Use subtree for CONTAINMENT language — everyone somewhere beneath:

    under / beneath / within
    in their organisation / reporting structure / people under them

"Who reports to X" asks for X's reports; "who is under X" asks for X's
whole organisation. "Directly" makes the first explicit, it does not
create it, so both phrasings are the same relationship.

CHOOSING TARGET_LEVEL

Rule 1:

If the question names the target level explicitly, use that level.

Examples: "advisors under Haseeb" -> advisor; "teams under Agency21" ->
team; "BCMs under Unit Head X" -> bcm.

Rule 2:

If the question asks for entities beneath a subject but does NOT name
the target level, use the immediate child level of subject_of.

Do NOT automatically jump to Advisor.

COUNTING

"how many", "count" and "number of" ask for the SIZE of the requested
set. They never change the target entity type: "how many advisors under
Haseeb" still targets advisor.

SORT

`sort.metric` names the measure by which rows are ordered.

Leave it null for a population that is not ranked.

`sort.direction`:

    top/best/highest/most -> desc
    bottom/worst/lowest/fewest -> asc

When no direction is stated, use desc.

"Worst" means least desirable, not necessarily numerically smallest.

For measures where a higher value is worse, such as overdue or late
arrivals, "worst" may therefore require descending order.

GROUPING

`group_by` changes the level at which results are grouped/reported.

Leave it null unless grouping is explicitly requested.

COMPARISON

Use comparison when the user explicitly asks to compare two or more
subjects.

Preserve every subject named.

FLAT

For operations supporting flat semantics:

    flat = false by default

Set flat = true only when the user explicitly requests a flat or
ungrouped list.

THE BUSINESS MODEL IS THE AUTHORITY

The hierarchy, the role vocabulary, the funnel/bookings rule and the
metric-key rules are stated once, in the authoritative business model
above. Re-read it when a field is ambiguous; it is not repeated here,
because two statements of one rule are two things to keep in agreement.

PERIOD COMPARISON

`time_range.compare_to` names the period being compared AGAINST when the
user asks for one.

Example:

    "this month vs last month"

means:

    period = MTD
    compare_to = supported earlier-period representation

Emit null for ordinary single-period queries.

Never use compare_to to smuggle in an unsupported period.

PERIOD RULES

Valid periods:

{_period_glossary()}

"today", "right now", or "this morning" means DAILY when DAILY exists.

Do not substitute MTD for TODAY.

Unsupported periods such as:

    last month
    yesterday
    this week
    custom date range

must not be silently converted to another period.

Use the valid clarification mechanism when the requested period cannot
be represented.

CONFIDENCE

All confidence values are 0-1.

`overall_confidence` is an execution gate.

Emit >= 0.8 when the query is understood and the fields correctly
represent the user's meaning.

Emit < 0.8 only when you genuinely would not stand behind the parse.

A single uncertain field belongs in that field's confidence rather than
artificially lowering overall confidence when the rest of the query is
clear.

`intent_confidence` measures confidence in QUERY SHAPE only.

It does not measure confidence in a metric, subject, or filter.

`time_range.confidence` measures confidence in the selected period.

If no period was stated and the system defaults to MTD, confidence should
remain relatively low (~0.5-0.6).

Only use high period confidence when the user's wording actually
establishes the period.
"""


# ---------------------------------------------------------------------------
# NAME GAZETTEERS
# ---------------------------------------------------------------------------

_NAME_PREFIX = 4


def _mentions_a_name_from(
    text: str,
    names: list[str],
) -> bool:

    words = {
        w
        for w in re.findall(
            r"[a-z]{%d,}" % _NAME_PREFIX,
            text.lower(),
        )
    }

    if not words:
        return False

    prefixes = {
        w[:_NAME_PREFIX]
        for w in words
    }

    for name in names:
        for part in re.findall(
            r"[a-z]{%d,}" % _NAME_PREFIX,
            name.lower(),
        ):
            if part[:_NAME_PREFIX] in prefixes:
                return True

    return False


def _person_gazetteer(
    label: str,
    names: list[str],
    text: str,
    already_grounded: bool,
) -> list[str]:

    if not names or already_grounded:
        return []

    if not _mentions_a_name_from(text, names):
        return []

    return [
        f"Known {label}: {', '.join(names[:200])}"
    ]


# ---------------------------------------------------------------------------
# METRIC CATALOG
# ---------------------------------------------------------------------------

def _metric_catalog_static() -> list[str]:

    terse = "\n".join(
        f"- {m.key}: {m.label} "
        f"[{_metric_kind(m)}] "
        f"(levels: {', '.join(m.entity_levels)})"
        for m in METRICS.values()
    )

    return [
        "Metric catalog (the ONLY valid metric keys):",
        terse,
    ]


def _metric_evidence(text: str) -> list[str]:

    matched = metric_aliases.resolve_all(text)

    if not matched:
        return [
            "No known metric phrasing matched this message.",
            "Use the full synonym catalog below to identify the closest "
            "matching metric key — never invent one:",
            metric_catalog_for_prompt(),
        ]

    found = "; ".join(
        f'"{m.phrase}" -> {m.metric}'
        for m in matched
    )

    return [
        "Deterministic metric evidence:",
        f"{found}.",
        (
            "Treat this as grounding evidence, but interpret the user's "
            "full sentence semantically. Do not discard a metric because "
            "the query also contains hierarchy or scope language."
        ),
    ]


def _metric_catalog_block(text: str) -> list[str]:

    matched = metric_aliases.resolve_all(text)

    if not matched:
        return [
            "Metric catalog (the ONLY valid metric keys):",
            metric_catalog_for_prompt(),
        ]

    terse = "\n".join(
        f"- {m.key}: {m.label} "
        f"(levels: {', '.join(m.entity_levels)})"
        for m in METRICS.values()
    )

    found = "; ".join(
        f'"{m.phrase}" -> {m.metric}'
        for m in matched
    )

    return [
        "Metric catalog (the ONLY valid metric keys):",
        terse,
        (
            "Phrases already matched by deterministic grounding: "
            f"{found}. Use these as evidence unless the sentence clearly "
            "means a different measure."
        ),
    ]


# ---------------------------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------------------------

def build_ir_prompt(
    text: str,
    known_teams: list[str],
    known_companies: list[str],
    grounded_entities: dict,
    prior_ir_json: str | None = None,
    recent_turns: list | None = None,
    known_unit_heads: list[str] | None = None,
    known_zonal_heads: list[str] | None = None,
    known_bcms: list[str] | None = None,
) -> str:

    teams_sample = ", ".join(
        known_teams[:200]
    )

    companies = ", ".join(
        known_companies
    )

    # ---------------------------------------------------------------
    # STATIC PREFIX
    # ---------------------------------------------------------------

    context_lines = [
        "You are a query-understanding parser for a real-estate sales "
        "operations chatbot.",
        "",
        "AUTHORITATIVE BUSINESS MODEL:",
        _business_model(),
        "",
        f"Known teams: {teams_sample}",
        f"Known companies: {companies}",
        "",
    ]

    context_lines.extend(
        _metric_catalog_static()
    )

    context_lines.append(
        CONDITION_VOCABULARY
    )

    context_lines.append(
        BUSINESS_PHRASE_GLOSSARY
    )

    context_lines.append(
        _ir_schema()
    )

    context_lines.append(
        render_examples()
    )

    # ---------------------------------------------------------------
    # PER-QUERY TAIL
    # ---------------------------------------------------------------

    grounded_levels = {
        k
        for k, v in grounded_entities.items()
        if v and not k.startswith("_")
    }

    for label, names, entity_key in (
        (
            "unit heads",
            known_unit_heads or [],
            hierarchy.LEVEL_ENTITY_KEYS.get("unit_head"),
        ),
        (
            "zonal heads",
            known_zonal_heads or [],
            hierarchy.LEVEL_ENTITY_KEYS.get("zonal_head"),
        ),
        (
            "BCMs",
            known_bcms or [],
            hierarchy.LEVEL_ENTITY_KEYS.get("bcm"),
        ),
    ):
        context_lines.extend(
            _person_gazetteer(
                label,
                names,
                text,
                entity_key in grounded_levels,
            )
        )

    context_lines.extend(
        _metric_evidence(text)
    )

    prompt_entities = {
        k: v
        for k, v in grounded_entities.items()
        if not k.startswith("_")
    }

    if prompt_entities:
        context_lines.append(
            "Entities already found by rule-based grounding "
            "(use these, don't re-derive): "
            f"{prompt_entities}"
        )

    if prior_ir_json:
        context_lines.append(
            "Previous turn's resolved query. For follow-ups, treat the "
            "new message as a semantic patch on this query: "
            f"{prior_ir_json}"
        )

    if recent_turns:
        rendered = "\n".join(
            f"  {role}: {turn_text}"
            for role, turn_text in recent_turns
        )

        context_lines.append(
            "Recent conversation (oldest first). Resolve pronouns and "
            "ellipsis against it, but prefer the resolved query above "
            "when the two disagree:\n"
            f"{rendered}"
        )

    context_lines.append(
        f'User message: "{text}"'
    )

    return "\n".join(
        context_lines
    )


# ---------------------------------------------------------------------------
# PROMPT FINGERPRINT
# ---------------------------------------------------------------------------

def prompt_fingerprint() -> str:
    """
    Hash only the static prompt components.

    This identifies the semantic prompt version without including the
    user's query or per-query grounding.
    """

    static = "\n".join(
        [
            _business_model(),
            *_metric_catalog_static(),
            CONDITION_VOCABULARY,
            BUSINESS_PHRASE_GLOSSARY,
            _ir_schema(),
            render_examples(),
        ]
    )

    return hashlib.sha256(
        static.encode()
    ).hexdigest()[:12]