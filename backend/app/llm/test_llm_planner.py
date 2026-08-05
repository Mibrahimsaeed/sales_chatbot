"""
LLM planner regression suite.

The LLM is MOCKED throughout — these tests verify the planner's
CONTRACT (schema, validation, adaptation, fallback), not a model's
output quality, which is not a property a test suite can pin down.

What's asserted, per the migration requirements:
  - the 13 required queries produce the right plan from a realistic
    planner response
  - every invalid planner output is rejected rather than executed
  - the feature flag switches planners, and failure ALWAYS falls back to
    the rule-based planner
  - the planner can never emit SQL or answer the user
"""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod, SalesFunnel
from app.llm import (
    advisor_resolver, conversation_memory, entity_extractor, llm_planner,
    narrative, semantic_parser,
)
from app.llm.planner_schema import LLMQueryPlan, QUERY_PLAN_JSON_SCHEMA
from app.llm.llm_planner import PlannerRejection, plan_query, to_query_plan, validate_plan
from app.services.chat_service import handle_chat_message


def _plan(**overrides) -> dict:
    base = {
        "intent": "advisor_profile", "entities": [], "metric": None, "period": None,
        "filters": [], "sort": None, "limit": None, "confidence": 0.95, "clarification": None,
    }
    base.update(overrides)
    return base


def _entity(type_, value):
    return {"type": type_, "value": value}


@pytest.fixture(autouse=True)
def _enable_planner(monkeypatch):
    monkeypatch.setattr(llm_planner.settings, "use_llm_planner", True)


@pytest.fixture()
def planner_db(db_session, monkeypatch):
    def advisor(wid, name, team, company, **kw):
        db_session.add(Advisor(wid=wid, name=name, team=team, company=company, **kw))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=10, mtd_followup_connect=0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD, target=1000, cleared=500))

    advisor(1, "Kaleem Ahmed", "Blue Area", "Graana", bm="Adeel Dogar", rm="Adeel Dogar")
    advisor(2, "Reportee One", "Blue Area", "Graana", bm="Kaleem Ahmed", rm="Kaleem Ahmed")
    advisor(3, "Reportee Two", "Downtown", "Graana", bm="Kaleem Ahmed", rm="Kaleem Ahmed")
    advisor(4, "Yasir Ali", "North/KPK", "Agency21", bm="Kaleem Ahmed", rm="Kaleem Ahmed")
    advisor(5, "Ali Murtaza", "Downtown", "Agency21", bm="Musab Sial", rm="Musab Sial")
    advisor(6, "Adeel Mubarik", "Gamma", "Agency21", bm="Adeel Dogar", rm="Adeel Dogar")
    db_session.commit()

    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


def _mock_llm(monkeypatch, response: dict | None):
    monkeypatch.setattr(llm_planner, "call_llm_structured", lambda *a, **k: response)


# =====================================================================
# The 13 required queries — planner output -> plan action
# =====================================================================

REQUIRED = [
    ("Who works under Kaleem",
     _plan(intent="roster", entities=[_entity("unit_head", "Kaleem Ahmed")]), "roster"),
    ("Show advisors in Blue Area",
     _plan(intent="roster", entities=[_entity("team", "Blue Area")]), "roster"),
    ("Top 5 advisors in Graana",
     _plan(intent="leaderboard", entities=[_entity("company", "Graana")],
           metric="mtd_cleared", limit=5,
           sort={"metric": "mtd_cleared", "direction": "desc"}), "leaderboard"),
    ("Compare Graana and Agency21",
     _plan(intent="comparison",
           entities=[_entity("company", "Graana"), _entity("company", "Agency21")]), "comparison"),
    ("Compare Graana and Agency21 by revenue",
     _plan(intent="comparison", metric="mtd_cleared",
           entities=[_entity("company", "Graana"), _entity("company", "Agency21")]), "comparison"),
    ("Who reports to Kaleem",
     _plan(intent="hierarchy", entities=[_entity("unit_head", "Kaleem Ahmed")]), "breakdown"),
    ("Who does Kaleem report to",
     _plan(intent="reverse_hierarchy", entities=[_entity("advisor", "Kaleem Ahmed")]),
     "reverse_hierarchy"),
    ("Tell me about Yasir Ali",
     _plan(intent="advisor_profile", entities=[_entity("advisor", "Yasir Ali")]), "lookup"),
    ("Show Adeel Dogar's team",
     _plan(intent="hierarchy", entities=[_entity("unit_head", "Adeel Dogar")]), "breakdown"),
    ("Who is BM of Ali Murtaza",
     _plan(intent="reverse_hierarchy", entities=[_entity("advisor", "Ali Murtaza")]),
     "reverse_hierarchy"),
    ("Late advisors in Blue Area",
     _plan(intent="attendance_filter", entities=[_entity("team", "Blue Area")],
           filters=[{"field": "attendance_status", "operator": "=", "value": "Late"}]),
     "attendance_filter"),
    ("Attendance of Agency21",
     _plan(intent="entity_summary", entities=[_entity("company", "Agency21")]), "summary"),
    ("Top performers under Kaleem",
     _plan(intent="leaderboard", entities=[_entity("unit_head", "Kaleem Ahmed")],
           metric="achievement_pct",
           sort={"metric": "achievement_pct", "direction": "desc"}), "leaderboard"),
]


@pytest.mark.parametrize("query,response,expected_action", REQUIRED,
                         ids=[q for q, _r, _a in REQUIRED])
def test_required_queries_produce_the_right_plan(planner_db, monkeypatch, query, response, expected_action):
    _mock_llm(monkeypatch, response)
    entities = entity_extractor.extract_entities(query.lower(), planner_db)
    plan = plan_query(query, entities, planner_db)
    assert plan is not None, "planner returned nothing"
    assert plan.action == expected_action


@pytest.mark.parametrize("query,response,expected_action", REQUIRED,
                         ids=[q for q, _r, _a in REQUIRED])
def test_required_queries_execute_end_to_end(planner_db, monkeypatch, query, response, expected_action):
    """Past the plan: entity resolution, SQL, and a formatted response.
    A plan that can't be executed is not a working planner."""
    _mock_llm(monkeypatch, response)
    result = handle_chat_message(planner_db, query, session_id=None)
    assert result["reply"], "no reply produced"
    assert result["type"] not in ("unknown",), f"{query!r} -> {result['type']}"


# =====================================================================
# Validation — every rejection is a case where executing would be wrong
# =====================================================================

def test_unknown_metric_is_rejected():
    with pytest.raises(PlannerRejection, match="unknown_metric"):
        validate_plan(LLMQueryPlan.model_validate(
            _plan(intent="leaderboard", metric="vibes")))


def test_unknown_sort_metric_is_rejected():
    with pytest.raises(PlannerRejection, match="unknown_sort_metric"):
        validate_plan(LLMQueryPlan.model_validate(
            _plan(intent="leaderboard", metric="mtd_cleared",
                  sort={"metric": "made_up", "direction": "desc"})))


def test_unknown_filter_field_is_rejected():
    with pytest.raises(PlannerRejection, match="unknown_filter_field"):
        validate_plan(LLMQueryPlan.model_validate(
            _plan(intent="roster", entities=[_entity("team", "Blue Area")],
                  filters=[{"field": "astrological_sign", "operator": "=", "value": "leo"}])))


def test_missing_entities_are_rejected():
    for intent in ("advisor_profile", "hierarchy", "reverse_hierarchy", "roster", "entity_summary"):
        with pytest.raises(PlannerRejection, match="missing_entities_for"):
            validate_plan(LLMQueryPlan.model_validate(_plan(intent=intent, entities=[])))


def test_comparison_with_one_entity_is_rejected():
    with pytest.raises(PlannerRejection, match="comparison_needs_two"):
        validate_plan(LLMQueryPlan.model_validate(
            _plan(intent="comparison", entities=[_entity("company", "Graana")])))


def test_leaderboard_without_a_metric_is_rejected():
    with pytest.raises(PlannerRejection, match="leaderboard_needs_a_metric"):
        validate_plan(LLMQueryPlan.model_validate(_plan(intent="leaderboard")))


def test_attendance_filter_without_a_status_is_rejected():
    with pytest.raises(PlannerRejection, match="attendance_filter_needs_a_status"):
        validate_plan(LLMQueryPlan.model_validate(
            _plan(intent="attendance_filter", entities=[_entity("team", "Blue Area")])))


def test_low_confidence_is_rejected():
    with pytest.raises(PlannerRejection, match="low_confidence"):
        validate_plan(LLMQueryPlan.model_validate(
            _plan(intent="advisor_profile", entities=[_entity("advisor", "X")], confidence=0.2)))


def test_invalid_period_fails_schema_validation():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        LLMQueryPlan.model_validate(_plan(period="LAST_MONTH"))


def test_invalid_intent_fails_schema_validation():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        LLMQueryPlan.model_validate(_plan(intent="run_arbitrary_sql"))


def test_rejected_plan_becomes_a_clarification_not_an_execution(planner_db, monkeypatch):
    _mock_llm(monkeypatch, _plan(intent="leaderboard", metric="nonsense_metric"))
    entities = entity_extractor.extract_entities("top advisors by nonsense", planner_db)
    plan = plan_query("top advisors by nonsense", entities, planner_db)
    assert plan.action == "unresolved"
    assert "llm_rejected" in plan.intent_evidence[0]


# =====================================================================
# The planner cannot write SQL or answer the user
# =====================================================================

def test_schema_has_no_field_that_could_carry_sql():
    fields = set(LLMQueryPlan.model_fields)
    assert not any("sql" in f.lower() or "query" in f.lower() for f in fields)
    assert set(QUERY_PLAN_JSON_SCHEMA["properties"]) == fields | set()


def test_extra_fields_from_the_model_are_ignored_not_executed():
    """A model that tries to smuggle SQL in an unexpected key must have
    it dropped, not passed along."""
    plan = LLMQueryPlan.model_validate(
        {**_plan(intent="greeting"), "sql": "DROP TABLE advisors"})
    assert not hasattr(plan, "sql")


def test_planner_never_supplies_an_id(planner_db, monkeypatch):
    """Entities carry TYPE + TEXT only — identity resolution stays with
    the WID resolver, which knows about duplicate names."""
    _mock_llm(monkeypatch, _plan(intent="advisor_profile",
                                 entities=[_entity("advisor", "Yasir Ali")]))
    entities = entity_extractor.extract_entities("tell me about yasir ali", planner_db)
    plan = plan_query("tell me about Yasir Ali", entities, planner_db)
    # the wid came from the RESOLVER, not from the planner response
    assert plan.entity_wid == 4


def test_ambiguous_person_still_asks_regardless_of_planner(planner_db, monkeypatch):
    """The LLM cannot know a name maps to several real people."""
    planner_db.add(Advisor(wid=99, name="Yasir Ali", team="Downtown", company="Graana"))
    planner_db.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()

    _mock_llm(monkeypatch, _plan(intent="advisor_profile",
                                 entities=[_entity("advisor", "Yasir Ali")]))
    entities = entity_extractor.extract_entities("tell me about yasir ali", planner_db)
    plan = plan_query("tell me about Yasir Ali", entities, planner_db)
    assert plan.action == "clarify_person"


# =====================================================================
# Feature flag + fallback
# =====================================================================

def test_flag_off_uses_the_rule_based_planner(planner_db, monkeypatch):
    monkeypatch.setattr(llm_planner.settings, "use_llm_planner", False)
    called = []
    monkeypatch.setattr(llm_planner, "call_llm_structured",
                        lambda *a, **k: called.append(1) or _plan())
    handle_chat_message(planner_db, "Show advisors in Blue Area", session_id=None)
    assert called == [], "LLM planner ran despite the flag being off"


def test_provider_unavailable_falls_back_to_rule_based(planner_db, monkeypatch):
    """The property that makes the flag safe to flip: worst case is
    today's behaviour, never an error."""
    _mock_llm(monkeypatch, None)
    result = handle_chat_message(planner_db, "Show advisors in Blue Area", session_id=None)
    assert result["type"] == "roster"


def test_planner_exception_falls_back_to_rule_based(planner_db, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("provider exploded")
    monkeypatch.setattr(llm_planner, "call_llm_structured", _boom)
    result = handle_chat_message(planner_db, "Show advisors in Blue Area", session_id=None)
    assert result["type"] == "roster"


def test_structurally_invalid_response_falls_back(planner_db, monkeypatch):
    _mock_llm(monkeypatch, {"intent": "not_a_real_intent"})
    result = handle_chat_message(planner_db, "Show advisors in Blue Area", session_id=None)
    assert result["type"] == "roster"


# Queries where the two planners DELIBERATELY differ. This list is the
# answer to "what does flipping the flag actually change" — the whole
# point of recording it rather than forcing agreement is that a silent
# behaviour change is what makes an A/B rollout unsafe.
KNOWN_DIVERGENCES = {
    # RESOLVED — "Top 5 advisors in Graana" is no longer a divergence.
    #
    # It used to be ("leaderboard", "roster"): the rule planner had no
    # leaderboard candidate without a named metric, so the roster reading
    # won and the "top 5" RANKING INTENT WAS SILENTLY DROPPED — it
    # returned every advisor in Graana. The scored-intent work gave the
    # rule planner a default metric on an explicit ranking word, so it
    # now reads the ranking and agrees with the LLM planner.
    #
    # Deliberately left as a comment rather than deleted: this entry was
    # the record of a known rule-planner weakness, and "the weakness is
    # gone" is worth more here than a silently shorter dict. The
    # agreement itself is now pinned by
    # test_both_planners_agree_except_where_documented, which asserts
    # every query NOT in this dict reaches the same action on both paths.

    # "Attendance of Agency21" — the capability gap this entry recorded is
    # now CLOSED on the rule side, and the entry documents that rather
    # than being deleted.
    #
    # It used to read: rule-based -> leaderboard, ranking Agency21's
    # advisors by attendance rate (answering "who has the best
    # attendance", not "what is Agency21's attendance"); LLM ->
    # entity_summary, whose fixed KPI set excludes attendance. The note
    # ended "the shape that would actually answer it — ONE metric for ONE
    # entity — doesn't exist yet in either planner."
    #
    # Phase 7 created that shape: `group_metric` is a first-class intent,
    # so the rule planner now answers with Agency21's OWN attendance
    # figure. The LLM side still returns entity_summary, so the two still
    # diverge — but the rule side is now the more correct one, which is
    # the reverse of when this was written.
    "Attendance of Agency21": ("summary", "group_metric"),
}


def test_both_planners_agree_except_where_documented(planner_db, monkeypatch):
    """A/B comparison is only meaningful if a flag flip is a rollback
    rather than a behaviour change. Every difference must be a KNOWN,
    justified one — an undocumented divergence fails this test."""
    from app.llm.query_planner import build_query_plan

    for query, response, _expected in REQUIRED:
        entities = entity_extractor.extract_entities(query.lower(), planner_db)
        rule_based = build_query_plan(query.lower(), entities)
        _mock_llm(monkeypatch, response)
        llm_based = plan_query(query, entities, planner_db)

        if rule_based.action == "unresolved":
            continue  # the rule-based planner simply can't express this one

        if query in KNOWN_DIVERGENCES:
            expected_llm, expected_rule = KNOWN_DIVERGENCES[query]
            assert (llm_based.action, rule_based.action) == (expected_llm, expected_rule), (
                f"{query!r}: divergence changed — llm={llm_based.action} rule={rule_based.action}"
            )
            continue

        assert llm_based.action == rule_based.action, (
            f"{query!r}: UNDOCUMENTED divergence — llm={llm_based.action} "
            f"rule={rule_based.action}. Either fix it or add it to KNOWN_DIVERGENCES."
        )


def test_the_documented_divergence_is_the_llm_planner_being_more_correct(planner_db, monkeypatch):
    """"Top 5 advisors in Graana" asks for a RANKING, and BOTH planners
    now honour it.

    INVERTED. This test used to assert the rule planner returned "roster"
    — dropping "top 5" entirely and listing every advisor in Graana — and
    documented that as the LLM planner being more correct. The
    scored-intent work closed that gap: an explicit ranking word now
    supplies a default metric, so the rule planner reads the ranking too.

    The assertion is kept rather than deleted because the ranking must
    not be silently dropped again by either path, and that is easiest to
    state where the regression was first recorded.
    """
    from app.llm.query_planner import build_query_plan

    entities = entity_extractor.extract_entities("top 5 advisors in graana", planner_db)
    rule_based = build_query_plan("top 5 advisors in graana", entities)
    assert rule_based.action == "leaderboard"
    assert rule_based.limit == 5

    _mock_llm(monkeypatch, _plan(
        intent="leaderboard", entities=[_entity("company", "Graana")],
        metric="mtd_cleared", limit=5,
        sort={"metric": "mtd_cleared", "direction": "desc"}))
    plan = plan_query("Top 5 advisors in Graana", entities, planner_db)
    assert plan.action == "leaderboard"
    assert plan.limit == 5


# =====================================================================
# Logging
# =====================================================================

def test_planner_exchange_is_traced(planner_db, monkeypatch):
    from app.core import tracing

    captured = []
    monkeypatch.setattr(tracing, "_emit", lambda t: captured.append(t.to_dict()))
    _mock_llm(monkeypatch, _plan(intent="roster", entities=[_entity("team", "Blue Area")]))
    handle_chat_message(planner_db, "Show advisors in Blue Area", session_id="t")

    planner_trace = captured[-1]["planner"]
    assert planner_trace["prompt_chars"] > 0
    assert planner_trace["raw_response"]["intent"] == "roster"
    assert planner_trace["validated_plan"]["intent"] == "roster"
    assert planner_trace["rejected"] is None
    assert planner_trace["elapsed_ms"] is not None


def test_rejection_reason_is_traced(planner_db, monkeypatch):
    from app.core import tracing

    captured = []
    monkeypatch.setattr(tracing, "_emit", lambda t: captured.append(t.to_dict()))
    _mock_llm(monkeypatch, _plan(intent="leaderboard", metric="bogus"))
    handle_chat_message(planner_db, "top advisors by bogus", session_id="t")

    assert captured[-1]["planner"]["rejected"] == "unknown_metric:bogus"
