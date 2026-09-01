""""Directly reports to X" means X's IMMEDIATE reports, not X's subtree.

Two defects, independent, both upstream of any traversal.

DIRECTLY WAS A NO-OP. The word appeared in no vocabulary, so every
phrasing below produced the same plan as "X's team". That is not a near
miss: hierarchy.scope_filter is one column match on a denormalised row,
which IS the whole subtree, so "how many advisors directly report to the
Unit Head" answered with everyone beneath him however many managers sat
in between.

A ROLE WITH A SCOPE RESOLVED TO NOBODY. "the Unit Head in AMD" named a
role and a scope; the role contributed nothing, the sentence degraded to
a plain team lookup, and the reply was the team's own headcount — a
different question, answered confidently. The capability existed
(get_manager_of_group, which "the unit head OF AMD" already used); only
the other preposition was missing.

The fixture mirrors the production shape that makes this subtle: a
manager who is ALSO his own sub-level. Faisal is Unit Head of everyone
and his own BCM for a handful, which is why self-exclusion and the
descend-when-empty rule both need testing rather than assuming the tree
is clean.
"""

import pytest

from app.database.models import Advisor
from app.llm import entity_extractor, hierarchy
from app.llm.entity_extractor import extract_entities
from app.llm.nlu_pipeline import Resolution, resolve
from app.llm.preprocessing import normalize
from app.llm.query_planner import build_query_plan
from app.services import chat_service, hierarchy_service

# ---------------------------------------------------------------------
# Path-agnostic semantic accessors
# ---------------------------------------------------------------------
#
# These queries are now parsed by the LLM into a QueryIR rather than by
# the rule planner, so `resolution.plan` is None for them. Asserting on
# an internal planner action pinned the ROUTE, which is exactly the
# coupling the migration removes — and it would fail for a perfectly
# correct answer.
#
# What the tests actually care about is the SEMANTICS the query resolved
# to: which level was enumerated, whether it was the immediate reports or
# the whole subtree, and who came back. These read that from whichever
# representation carried it.


def _semantics(resolution):
    """(target_level, relation) for a hierarchy read, from either path."""
    ir = getattr(resolution, "ir", None)
    if ir is not None and ir.is_hierarchy_read():
        return ir.target_level, ir.relation
    plan = getattr(resolution, "plan", None)
    if plan is None:
        return None, None
    action = getattr(plan, "action", None)
    relation = {"direct_reports": "direct",
                "scoped_reports": "subtree",
                "roster": "subtree"}.get(action)
    return getattr(plan, "target_level", None), relation


def _target_level(resolution, response=None):
    """The level that was actually enumerated.

    The plan leaves `target_level` unset and lets the service default it
    to the level below, resolving it into the RESPONSE; the IR states it
    up front. Reading the response first means this asserts the level the
    answer is actually about, whichever path produced it.
    """
    if isinstance(response, dict):
        data = response.get("data")
        if isinstance(data, dict) and data.get("target_level"):
            return data["target_level"]
    return _semantics(resolution)[0]


def _relation(resolution):
    return _semantics(resolution)[1]


def _result_names(response):
    """Member names, whichever shape the answer came back in."""
    data = response.get("data")
    if isinstance(data, list):
        return sorted(r.get("name") for r in data if r.get("name"))
    if isinstance(data, dict):
        members = data.get("members") or data.get("advisors") or []
        return sorted(m.get("name") for m in members if m.get("name"))
    return []


def _result_count(response):
    data = response.get("data")
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and "count" in data:
        return data["count"]
    return len(_result_names(response))


# team -> unit_head(rm) -> zonal_head(portfolio_lead) -> bcm(management_lead)
#
# UH "Uma" heads team AMD. She is her own Zonal Head for one branch and
# her own BCM for two advisors, exactly as production managers are.
_ORG = [
    # wid, name,        team,  rm,     portfolio_lead, management_lead
    (1, "Uma",          "AMD", "Uma",  "Uma",          "Uma"),
    (2, "Direct One",   "AMD", "Uma",  "Uma",          "Uma"),
    (3, "Direct Two",   "AMD", "Uma",  "Uma",          "Uma"),
    (4, "Zed",          "AMD", "Uma",  "Zed",          "Zed"),
    (5, "Under Zed",    "AMD", "Uma",  "Zed",          "Zed"),
    (6, "Bee",          "AMD", "Uma",  "Zed",          "Bee"),
    (7, "Under Bee",    "AMD", "Uma",  "Zed",          "Bee"),
    (8, "Outsider",     "OTH", "Otto", "Otto",         "Otto"),
]


@pytest.fixture()
def org(db_session):
    for wid, name, team, rm, pl, ml in _ORG:
        db_session.add(Advisor(wid=wid, name=name, team=team, company="Graana",
                               rm=rm, portfolio_lead=pl, management_lead=ml,
                               in_master_sheet=True))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    return db_session


def _reply(db, text):
    cleaned = normalize(text)
    entities = extract_entities(cleaned, db)
    plan = build_query_plan(cleaned, entities)
    response = chat_service._dispatch(
        db, Resolution(kind="plan", plan=plan, entities=entities))
    return plan, str(response.get("reply") or "")


def _resolved_reply(db, text):
    """The WHOLE pipeline, nlu_pipeline.resolve included.

    _reply above calls build_query_plan directly, which skips the routing
    layer that decides whether a plan is served or handed to the semantic
    parser. An action missing from nlu_pipeline._RULE_BASED_ACTIONS is
    routed away and, with the LLM unreachable, answers "I'm not tracking
    that one" — a plan built perfectly and then discarded. Every test
    asserting an ANSWER goes through here, because that is the only path
    the running app takes.
    """
    resolution = resolve(text, db, session_id=None)
    response = chat_service._dispatch(db, resolution)
    return resolution, str(response.get("reply") or "")


def _names(reports):
    return [m["name"] for m in reports["members"]]


# ------------------------------------------------- direct, every level
def test_unit_head_direct_reports_are_zonal_heads_only(org):
    """Uma heads 7 advisors. Her IMMEDIATE reports are Zonal Heads, and
    there is exactly one other than herself."""
    reports = hierarchy_service.get_direct_reports(org, "unit_head", "Uma")
    assert reports["target_level"] == "zonal_head"
    assert _names(reports) == ["Zed"]
    assert reports["count"] == 1


def test_zonal_head_direct_reports_are_bcms_only(org):
    reports = hierarchy_service.get_direct_reports(org, "zonal_head", "Zed")
    assert reports["target_level"] == "bcm"
    assert _names(reports) == ["Bee"]


def test_bcm_direct_reports_are_advisors_only(org):
    reports = hierarchy_service.get_direct_reports(org, "bcm", "Bee")
    assert reports["target_level"] == "advisor"
    assert _names(reports) == ["Under Bee"]


def test_an_advisor_has_no_reports(org):
    """The leaf. None rather than an empty list, so the caller can say
    "not a question about this level" instead of "nobody"."""
    assert hierarchy_service.get_direct_reports(org, "advisor", "Under Bee") is None
    assert hierarchy.child_of("advisor") is None


def test_an_explicit_target_level_is_honoured(org):
    """"How many ADVISORS report directly to the Unit Head" asks about
    the leaf, not about the level below her."""
    reports = hierarchy_service.get_direct_reports(org, "unit_head", "Uma",
                                                   target_level="advisor")
    assert reports["target_level"] == "advisor"
    assert _names(reports) == ["Direct One", "Direct Two"]


# ------------------------------------------------------ self-exclusion
def test_a_manager_is_never_their_own_direct_report(org):
    """Uma is her own Zonal Head and her own BCM, and has an advisor row.
    She must appear in none of her own report lists — otherwise a
    headcount of her reports counts her as one of them."""
    for target in ("zonal_head", "advisor"):
        reports = hierarchy_service.get_direct_reports(org, "unit_head", "Uma",
                                                       target_level=target)
        assert "Uma" not in _names(reports), target
    assert "Zed" not in _names(
        hierarchy_service.get_direct_reports(org, "zonal_head", "Zed"))


def test_self_exclusion_does_not_hide_real_reports(org):
    """Zed is the only BCM beneath himself as Zonal Head, so his BCM
    level is empty once self is excluded — but "Under Zed" genuinely
    names him as their immediate manager. Answering "nobody" would be
    false in the only sense the question means."""
    reports = hierarchy_service.get_direct_reports(org, "zonal_head", "Zed",
                                                   target_level="bcm")
    assert _names(reports) == ["Bee"]

    solo = hierarchy_service.get_direct_reports(org, "bcm", "Zed")
    assert solo["target_level"] == "advisor"
    assert _names(solo) == ["Under Zed"]


# ------------------------------------- direct vs the subtree it isn't
def test_direct_is_a_strict_subset_of_the_full_scope(org):
    """The distinction the fix exists for, asserted as the relationship
    rather than as two hand-written numbers."""
    from app.llm import aggregation

    everyone = aggregation.headcount(org, "unit_head", "Uma")
    direct = hierarchy_service.get_direct_reports(org, "unit_head", "Uma",
                                                  target_level="advisor")
    assert everyone == 7
    assert direct["count"] == 2
    assert direct["count"] < everyone


def test_the_full_scope_reading_is_unchanged(org):
    """"X's team" and "people under X" keep the subtree they always had —
    scope_filter is untouched, and this is what guards that."""
    from app.llm import aggregation

    assert aggregation.headcount(org, "unit_head", "Uma") == 7
    roster = hierarchy_service.get_level_roster(org, "unit_head", "Uma")
    assert roster["count"] == 7


def test_count_and_list_come_from_one_population(org):
    reports = hierarchy_service.get_direct_reports(org, "unit_head", "Uma")
    assert reports["count"] == len(reports["members"])


# ------------------------------------------------------ every phrasing
# Phrasings that name NO population, so the target stays the default:
# the level immediately below the manager. "people directly under Uma"
# is deliberately not here — a people-word names the leaf, and is
# covered by the people-word tests further down.
_PHRASINGS = [
    "who directly reports to Uma",
    "who reports directly to Uma",
    "directly works under Uma",
    "direct reports of Uma",
    "Uma's direct reports",
    "how many directly report to Uma",
]


@pytest.mark.parametrize("text", _PHRASINGS)
def test_every_direct_phrasing_routes_the_same_way(text, org):
    resolution, reply = _resolved_reply(org, text)
    assert _relation(resolution) == "direct", text
    assert "Zed" in reply
    # The subtree's other members must not appear.
    assert "Under Bee" not in reply


def test_a_people_word_names_the_leaf(org):
    """"people directly under X" asks about advisors, not about the level
    below X — the same reading as "advisors directly under X"."""
    _, people = _resolved_reply(org, "people directly under Uma")
    _, advisors = _resolved_reply(org, "advisors directly under Uma")
    assert people == advisors
    assert "Direct One" in people and "Zed" not in people


def test_the_word_directly_is_what_changes_the_answer(org):
    """Same sentence, one word apart. Without it the reading stays the
    subtree one it has always been."""
    direct_plan, _ = _reply(org, "who directly reports to Uma")
    plain_plan, _ = _reply(org, "who reports to Uma")
    assert direct_plan.action == "direct_reports"
    assert plain_plan.action != "direct_reports"


# --------------------------------------------- role within a scope (b)
def test_a_role_named_within_a_scope_resolves_to_the_person(org):
    """"the Unit Head in AMD" — the role word contributed nothing before
    this, and the sentence degraded to a team lookup."""
    plan, reply = _reply(org, "who directly reports to the Unit Head in AMD")
    assert plan.action == "direct_reports"
    assert plan.level == "unit_head"
    assert plan.subject_level == "team"
    assert "Uma" in reply and "Zed" in reply


def test_the_reported_query_counts_direct_advisors_only(org):
    """The query this work started from. The answer is Uma's two direct
    advisors, not the seven in her subtree and not the team's headcount."""
    plan, reply = _reply(org, "how many advisors directly report to the Unit Head in AMD")
    assert plan.action == "direct_reports"
    assert plan.target_level == "advisor"
    assert "2 Advisors" in reply
    assert "Direct One" in reply and "Direct Two" in reply
    assert "Under Bee" not in reply


def test_in_and_of_resolve_the_role_the_same_way(org):
    """"of" already worked; "in" is the same question."""
    for preposition in ("in", "of"):
        _, reply = _reply(org, f"who directly reports to the Unit Head {preposition} AMD")
        assert "Zed" in reply, preposition


def test_a_scope_with_several_role_holders_asks_which(org):
    """Two Unit Heads in one team contradicts nothing — it just means the
    question has two answers, and picking one would be a guess."""
    org.add(Advisor(wid=9, name="Second UH", team="AMD", company="Graana",
                    rm="Second UH", portfolio_lead="Second UH",
                    management_lead="Second UH", in_master_sheet=True))
    org.commit()
    entity_extractor._cache["loaded_at"] = 0

    _, reply = _reply(org, "who directly reports to the Unit Head in AMD")
    assert "more than one" in reply.lower()
    assert "Uma" in reply and "Second UH" in reply


# --------------------------------------------------- untouched readings
def test_a_plain_roster_is_unchanged(org):
    plan, _ = _reply(org, "all advisors in AMD")
    assert plan.action == "roster"


def test_a_plain_reverse_lookup_is_unchanged(org):
    """"the unit head of AMD" kept reverse_hierarchy — the `in` branch
    must not have stolen the `of` one."""
    plan, _ = _reply(org, "the unit head of AMD")
    assert plan.action == "reverse_hierarchy"


def test_direct_scope_filter_derives_from_the_chain(org):
    """No hardcoded pairs: the predicate is built from parent_of, so a
    rebound CHAIN carries it without an edit here."""
    assert hierarchy.direct_scope_filter("advisor", "anyone") is None
    for level in ("unit_head", "zonal_head", "bcm"):
        assert hierarchy.direct_scope_filter(level, "x") is not None


# ------------------------------------- relative clauses and people-words
#
# "how many advisors THAT directly report to the Unit Head in AMD" got
# "I'm not tracking that one". The clause was a red herring: the plan was
# built correctly either way, and BOTH this and the version without
# "that" were failing in the running app. `direct_reports` was missing
# from nlu_pipeline._RULE_BASED_ACTIONS, so every phrasing was routed to
# the semantic parser and, with the LLM unreachable, discarded.

_ADVISOR_VARIANTS = [
    "how many advisors directly report to the Unit Head in AMD",
    "how many advisors that directly report to the Unit Head in AMD",
    "how many advisors who directly report to the Unit Head in AMD",
    "advisors that directly report to the Unit Head in AMD",
    "advisors who directly report to the Unit Head in AMD",
    "advisors directly reporting to the Unit Head in AMD",
    "people that directly report to the Unit Head in AMD",
    "people who directly report to the Unit Head in AMD",
    "staff who directly report to the Unit Head in AMD",
]


@pytest.mark.parametrize("text", _ADVISOR_VARIANTS)
def test_relative_clause_variants_all_answer_identically(text, org):
    """Through the FULL pipeline, not just the planner."""
    resolution, reply = _resolved_reply(org, text)
    assert resolution.kind == "plan", text
    assert _relation(resolution) == "direct", text
    assert resolution.plan.target_level == "advisor", text
    assert "2 Advisors" in reply, text
    assert "Direct One" in reply and "Direct Two" in reply
    assert "Under Bee" not in reply


def test_every_variant_gives_the_same_answer_as_every_other(org):
    """Stated as the equality rather than as nine copies of one string."""
    replies = {_resolved_reply(org, text)[1] for text in _ADVISOR_VARIANTS}
    assert len(replies) == 1, f"variants disagreed: {replies}"


def test_direct_reports_is_served_on_the_rule_path(org):
    """The root cause, asserted where it lives. Without this entry the
    action is routed to the semantic parser and the plan is thrown away.
    """
    from app.llm import nlu_pipeline

    assert "direct_reports" in nlu_pipeline._RULE_BASED_ACTIONS
    plan = build_query_plan(*_planned(org, _ADVISOR_VARIANTS[1]))
    assert nlu_pipeline._is_rule_based(plan)


def _planned(db, text):
    cleaned = normalize(text)
    return cleaned, extract_entities(cleaned, db)


def test_a_bare_who_question_still_targets_the_next_level_down(org):
    """No people-word and no level word means the default target stands:
    the level below the manager, not the leaf."""
    _, reply = _resolved_reply(org, "who directly reports to the Unit Head in AMD")
    assert "Zonal Head" in reply and "Zed" in reply


def test_people_words_are_read_from_one_declaration(org):
    """The target scan and ROSTER_RE share cat.PEOPLE_WORDS, so a word
    added there is understood by both rather than by whichever was
    remembered."""
    from app.llm import intent_catalog as cat

    assert "people" in cat.PEOPLE_WORDS and "staff" in cat.PEOPLE_WORDS
    for word in cat.PEOPLE_WORDS:
        _, reply = _resolved_reply(org, f"{word} who directly report to the Unit Head in AMD")
        assert "2 Advisors" in reply, word


def test_a_plain_roster_still_reaches_the_roster_action(org):
    """ROSTER_RE was rebuilt from the shared vocabulary — same pattern,
    and this is what proves it still triggers."""
    resolution, reply = _resolved_reply(org, "all advisors in AMD")
    assert _relation(resolution) == "subtree"
    assert "7 advisor(s)" in reply


# ------------------------------- a manager named outright, every level
#
# "how many advisors directly report to <a real BCM>" answered with that
# BCM's OWN PROFILE. Two causes, both in the routing layer above the
# planner, and both invisible to a test that called build_query_plan:
#
#   _UNDER_RE listed "reports to" and "reporting to" but not the bare
#   "report to" — the form a plural subject produces ("advisors ... report
#   to X") — so the relation went unrecognised and "advisors" was read as
#   naming WHO X IS rather than what to return.
#
#   _asks_for_the_group did not count a direct question as a question
#   about the people under someone, so a phrasing with no level word at
#   all pinned the name to `advisor` for the same reason.

_MANAGERS = [("bcm", "Bee", 1), ("zonal_head", "Zed", 1), ("unit_head", "Uma", 2)]


@pytest.mark.parametrize("level,who,expected", _MANAGERS)
def test_advisor_count_for_a_named_manager_at_every_level(level, who, expected, org):
    resolution, reply = _resolved_reply(org, f"how many advisors directly report to {who}")

    assert _relation(resolution) == "direct", who
    assert _target_level(resolution) == "advisor", who
    assert reply.startswith(f"{expected} Advisor"), reply


@pytest.mark.parametrize("level,who,expected", _MANAGERS)
def test_the_reply_count_equals_get_direct_reports(level, who, expected, org):
    """The number in the sentence and the service's population are the
    same thing, asserted as an equality rather than two literals."""
    resolution, reply = _resolved_reply(org, f"how many advisors directly report to {who}")
    plan = resolution.plan
    reports = hierarchy_service.get_direct_reports(
        org, plan.level, plan.entity_value, plan.target_level)

    assert reports["count"] == expected
    assert reply.startswith(f"{reports['count']} ")
    for member in reports["members"]:
        assert member["name"] in reply


@pytest.mark.parametrize("level,who,_expected", _MANAGERS)
def test_only_immediate_children_are_returned(level, who, _expected, org):
    """Every returned advisor names the manager as their OWN immediate
    manager, and the result is a strict subset of the manager's subtree.
    """
    from app.llm import aggregation

    reports = hierarchy_service.get_direct_reports(org, level, who,
                                                   target_level="advisor")
    returned = {m["name"] for m in reports["members"]}

    immediate = {
        a.name for a in org.query(Advisor)
        .filter(Advisor.management_lead.ilike(who),
                Advisor.in_master_sheet.is_(True)).all()
        if a.name.lower() != who.lower()
    }
    assert returned == immediate, who
    assert who not in returned, "self must never appear"
    assert len(returned) < aggregation.headcount(org, level, who)


@pytest.mark.parametrize("level,who,_expected", _MANAGERS)
def test_a_named_manager_resolves_through_the_full_pipeline(level, who, _expected, org):
    """Guards the routing layer specifically: an action the planner builds
    correctly must still be SERVED, not handed to the semantic parser."""
    resolution, _ = _resolved_reply(org, f"who directly reports to {who}")
    assert _relation(resolution) == "direct", who


def test_the_bare_infinitive_reads_as_a_relation(org):
    """"report to" alongside the inflected forms the pattern already had.
    Missing it is what made "advisors ... report to X" a profile lookup."""
    from app.llm.nlu_pipeline import _UNDER_RE

    for phrase in ("report to", "reports to", "reporting to", "under", "beneath", "below"):
        assert _UNDER_RE.search(f"advisors directly {phrase} Uma"), phrase
    # The comparator guard is intact: a number after the word is a
    # threshold, not a manager.
    assert not _UNDER_RE.search("advisors under 50 connects")


def test_a_manager_profile_question_is_still_a_profile_question(org):
    """The pin these fixes bypass must still happen when the turn really
    is about the person: a measure named, no direct wording."""
    resolution, _ = _resolved_reply(org, "connects of Uma")
    assert _relation(resolution) != "direct"
