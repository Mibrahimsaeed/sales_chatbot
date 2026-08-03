"""ir_patcher unit tests plus the multi-turn follow-up chain through
nlu_pipeline.resolve() (the report's canonical sequence, with the period
turn using 'ytd' since 'last month' needs the paused trend phase)."""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod
from app.llm import conversation_memory, entity_extractor, nlu_pipeline, semantic_parser
from app.llm.ir_patcher import try_patch
from app.llm.query_ir import Filter, MetricRef, QueryIR, Sort


def _leaderboard_ir(**overrides) -> QueryIR:
    base = dict(
        intent="leaderboard",
        subject_level="advisor",
        metric=MetricRef(key="mtd_cleared"),
        sort=Sort(metric="mtd_cleared", direction="desc"),
        limit=10,
    )
    base.update(overrides)
    return QueryIR(**base)


# ---- pure-function rule tests ----

def test_top_n_patch():
    patched = try_patch(_leaderboard_ir(), "top 5", {}, plan_action="unresolved")
    assert patched.limit == 5
    assert patched.sort.direction == "desc"


def test_bottom_n_flips_direction():
    patched = try_patch(_leaderboard_ir(), "bottom 3", {}, plan_action="unresolved")
    assert patched.limit == 3
    assert patched.sort.direction == "asc"


def test_sort_direction_patch():
    patched = try_patch(_leaderboard_ir(), "now sort ascending", {}, plan_action="unresolved")
    assert patched.sort.direction == "asc"


def test_period_patch():
    patched = try_patch(_leaderboard_ir(), "ytd", {}, plan_action="unresolved")
    assert patched.time_range.period == "YTD"


def test_only_company_patch_requires_grounded_entity():
    entities = {"companies": ["Graana"], "company": "Graana"}
    patched = try_patch(_leaderboard_ir(), "only Graana", entities, plan_action="summary")
    assert patched is not None
    assert Filter(field="company", operator="=", value="Graana") in patched.filters


def test_only_prefix_replaces_existing_same_type_filter():
    prior = _leaderboard_ir(filters=[Filter(field="company", operator="=", value="IMARAT")])
    entities = {"companies": ["Graana"], "company": "Graana"}
    patched = try_patch(prior, "only Graana", entities, plan_action="summary")
    company_filters = [f for f in patched.filters if f.field == "company"]
    assert company_filters == [Filter(field="company", operator="=", value="Graana")]


def test_bare_entity_without_only_prefix_is_not_a_patch():
    entities = {"companies": ["Graana"], "company": "Graana"}
    assert try_patch(_leaderboard_ir(), "Graana", entities, plan_action="summary") is None


def test_only_unit_head_patch_uses_generic_hierarchy_entity_keys():
    entities = {"unit_heads": ["Zeeshan Tariq"], "unit_head": "Zeeshan Tariq"}
    patched = try_patch(_leaderboard_ir(), "only Zeeshan Tariq", entities, plan_action="unresolved")
    assert patched is not None
    assert Filter(field="unit_head", operator="=", value="Zeeshan Tariq") in patched.filters


def test_breakdown_plan_action_treated_like_summary_for_only_prefix():
    entities = {"unit_heads": ["Zeeshan Tariq"], "unit_head": "Zeeshan Tariq"}
    patched = try_patch(_leaderboard_ir(), "only Zeeshan Tariq", entities, plan_action="breakdown")
    assert patched is not None

    # a bare (no "only") mention with plan_action="breakdown" is a NEW
    # question, not a patch — same as plan_action="summary" already was
    assert try_patch(_leaderboard_ir(), "Zeeshan Tariq", entities, plan_action="breakdown") is None


def test_remove_filters_patch():
    prior = _leaderboard_ir(filters=[Filter(field="company", operator="=", value="Graana")])
    patched = try_patch(prior, "all companies", {}, plan_action="unresolved")
    assert patched.filters == []


def test_self_standing_query_is_declined():
    assert try_patch(_leaderboard_ir(), "top teams by overdue", {}, plan_action="leaderboard") is None


def test_long_sentence_is_declined():
    text = "could you please show me all of the advisors who did really well"
    assert try_patch(_leaderboard_ir(), text, {}, plan_action="unresolved") is None


def test_unmatched_short_text_is_declined():
    assert try_patch(_leaderboard_ir(), "hmm okay", {}, plan_action="unresolved") is None


def test_prior_ir_is_not_mutated():
    prior = _leaderboard_ir()
    try_patch(prior, "top 3", {}, plan_action="unresolved")
    assert prior.limit == 10


# ---- multi-turn chain through resolve() ----

@pytest.fixture()
def chain_db(db_session, monkeypatch):
    db_session.add_all([
        Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana"),
        Advisor(wid=2, name="Ali Raza", team="Downtown", company="IMARAT"),
    ])
    db_session.add(Performance(wid=1, period=PerformancePeriod.MTD, cleared=900))
    db_session.add(Performance(wid=2, period=PerformancePeriod.MTD, cleared=500))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()


def test_follow_up_chain_patches_without_extra_llm_calls(chain_db, monkeypatch):
    llm_calls = []

    def fake_llm(prompt, schema, schema_name):
        llm_calls.append(prompt)
        return {
            "intent": "leaderboard",
            "subject_level": "advisor",
            "subjects": [],
            "metric": {"key": "mtd_cleared", "confidence": 0.95},
            "filters": [],
            "time_range": {"mode": "snapshot", "period": "MTD", "compare_to": None},
            "sort": {"metric": "mtd_cleared", "direction": "desc"},
            "limit": 10,
            "group_by": None,
            "overall_confidence": 0.95,
        }

    monkeypatch.setattr(semantic_parser, "call_llm_structured", fake_llm)
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "llm_first")
    session = "chain-1"

    # turn 1: full parse (one LLM call)
    r1 = nlu_pipeline.resolve("show top advisors by revenue", chain_db, session_id=session)
    assert r1.kind == "ir"
    assert len(llm_calls) == 1

    # turn 2: "only Graana" — deterministic patch, no new LLM call
    r2 = nlu_pipeline.resolve("only Graana", chain_db, session_id=session)
    assert r2.kind == "ir"
    assert len(llm_calls) == 1
    assert any(f.field == "company" and f.value == "Graana" for f in r2.ir.filters)

    # turn 3: "top 5" — patch again, filters retained
    r3 = nlu_pipeline.resolve("top 5", chain_db, session_id=session)
    assert r3.kind == "ir"
    assert len(llm_calls) == 1
    assert r3.ir.limit == 5
    assert any(f.field == "company" and f.value == "Graana" for f in r3.ir.filters)

    # turn 4: "ytd" — period patch, everything else retained
    r4 = nlu_pipeline.resolve("ytd", chain_db, session_id=session)
    assert r4.kind == "ir"
    assert len(llm_calls) == 1
    assert r4.ir.time_range.period == "YTD"
    assert r4.ir.limit == 5


# ---------------------------------------------------------------------
# The bare-modifier gate
# ---------------------------------------------------------------------
#
# "top 5" reaches try_patch with plan_action="leaderboard": the planner
# sees an explicit ranking word and supplies a default metric. That
# verdict describes what the planner WOULD DO with the message, not what
# the message contains, and reading it as an ellipsis test sent a pure
# limit change through a full LLM parse — dropping the prior turn's
# filters. _is_bare_modifier() closes that gap, and these pin its edges
# so it cannot widen into swallowing real questions.


def test_a_bare_limit_change_is_a_modifier():
    from app.llm.ir_patcher import _is_bare_modifier

    assert _is_bare_modifier("top 5", {})
    assert _is_bare_modifier("bottom 3", {})
    assert _is_bare_modifier("ascending", {})
    assert _is_bare_modifier("all teams", {})


def test_a_message_naming_a_measure_is_not_a_modifier():
    """"top 5 by revenue" is a self-standing question — it says what to
    rank by, so it must go through the normal parse path."""
    from app.llm.ir_patcher import _is_bare_modifier

    assert not _is_bare_modifier("top 5 by revenue", {})
    assert not _is_bare_modifier("top 3 by overdue", {})


def test_a_message_naming_a_subject_is_not_a_modifier():
    """A grounded entity means the message can stand alone."""
    from app.llm.ir_patcher import _is_bare_modifier

    assert not _is_bare_modifier("top 5", {"team": "Blue Area"})
    assert not _is_bare_modifier("top 5", {"advisor_wids": [1]})


def test_a_message_with_no_modifier_pattern_is_not_a_modifier():
    from app.llm.ir_patcher import _is_bare_modifier

    assert not _is_bare_modifier("who is the best", {})
    assert not _is_bare_modifier("graana", {})


def test_a_self_standing_query_still_declines_the_patch(chain_db):
    """The guardrail the module was built around: a new question after a
    leaderboard is a new question, not a patch."""
    from app.llm.entity_extractor import extract_entities
    from app.llm.ir_patcher import try_patch
    from app.llm.query_ir import MetricRef, QueryIR, Sort
    from app.llm.query_planner import build_query_plan

    prior = QueryIR(intent="leaderboard", subject_level="advisor",
                    metric=MetricRef(key="mtd_cleared"),
                    sort=Sort(metric="mtd_cleared"), limit=10)
    text = "show top teams by overdue"
    entities = extract_entities(text, chain_db)
    plan = build_query_plan(text, entities)

    assert try_patch(prior, text, entities, plan.action) is None
