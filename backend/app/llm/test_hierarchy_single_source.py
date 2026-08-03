"""Step 3 — the prompt, the schema and the registry are one hierarchy.

Three copies of the org chart had drifted apart, and the drift was
invisible because nothing compared them:

  F2  llm_client.QUERY_IR_JSON_SCHEMA built its `subject_level` /
      `subjects[].type` / `group_by` enums from hierarchy.HIERARCHY_LEVELS,
      which was `list(CHAIN)` — the chain ONLY. That schema is sent with
      strict:True, so decoding is grammar-constrained: the model could not
      emit "company", "office" or "region" at all. "Top companies by
      revenue" was forced into a chain level nobody asked for. The one
      test covering this compared the schema to query_ir.Level with `<=`,
      which a narrowing satisfies.

  F3  prompt_builder stated the PRE-Phase-3 chain in prose —
      "company -> region -> unit_head -> zonal_head -> unit ->
      business_center -> team -> advisor" — with team at the bottom, a
      `unit` level Phase 3 deleted, `business_center` (now the `office`
      attribute), and no `bcm` at all. So the prompt instructed the model
      to use values its own grammar rejected, and never mentioned a real
      level.

The registry is now the only declaration; the schema enum, the prompt
prose and query_ir.Level are all generated from it. These tests assert
the three agree, so a future edit to one cannot silently diverge again.
"""

import pytest

from app.llm import hierarchy, prompt_builder
from app.llm.entity_extractor import extract_entities
from app.llm.llm_client import QUERY_IR_JSON_SCHEMA
from app.llm.preprocessing import normalize
from app.llm.query_ir import LEVEL_NAMES, Level, QueryIR
from app.llm.query_planner import build_query_plan


def _schema_levels(path: str) -> list[str]:
    """The enum the grammar actually constrains, for one schema field."""
    props = QUERY_IR_JSON_SCHEMA["properties"]
    if path == "subject_level":
        return props["subject_level"]["enum"]
    if path == "subjects":
        return props["subjects"]["items"]["properties"]["type"]["enum"]
    if path == "group_by":
        return next(o["enum"] for o in props["group_by"]["anyOf"] if "enum" in o)
    raise AssertionError(path)


# ---------------------------------------------------------------------
# THE validation test: prompt == schema == registry
# ---------------------------------------------------------------------

@pytest.mark.parametrize("field", ["subject_level", "subjects", "group_by"])
def test_schema_hierarchy_equals_the_registry(field):
    """Equality, not a subset. `<=` is what let the schema narrow to the
    chain without any test noticing."""
    assert _schema_levels(field) == hierarchy.HIERARCHY_LEVELS, field


def test_prompt_hierarchy_equals_the_registry():
    """Every addressable level appears in the prompt's level union, and
    the union contains nothing the schema would reject."""
    schema_text = prompt_builder._ir_schema()
    union = prompt_builder._level_union()

    offered = {token.strip().strip('"') for token in union.split("|")}
    assert offered == set(hierarchy.HIERARCHY_LEVELS)

    for level in hierarchy.HIERARCHY_LEVELS:
        assert f'"{level}"' in schema_text, level


def test_prompt_chain_equals_the_registry_chain():
    """The prose the model reads, in the registry's order."""
    described = prompt_builder._chain_description()
    assert described.split(" -> ") == [
        f"{level} ({hierarchy.label_for(level)})" for level in hierarchy.CHAIN
    ]


def test_query_ir_level_equals_the_registry_plus_aliases():
    expected = set(hierarchy.HIERARCHY_LEVELS) | set(hierarchy.LEVEL_ALIASES)
    assert set(Level.__args__) == expected
    assert set(LEVEL_NAMES) == expected


def test_the_prompt_never_instructs_a_value_the_grammar_rejects():
    """The F3 failure mode itself: the prompt told the model to use
    `unit` and `business_center`, and strict decoding forbade both."""
    schema_text = prompt_builder._ir_schema()
    allowed = set(_schema_levels("subject_level"))

    for token in prompt_builder._level_union().split("|"):
        assert token.strip().strip('"') in allowed, token


# ---------------------------------------------------------------------
# Task 4 — the specific guarantees
# ---------------------------------------------------------------------

def test_bcm_exists_everywhere():
    """A real chain level that the prompt never mentioned."""
    assert "bcm" in hierarchy.CHAIN
    assert "bcm" in hierarchy.HIERARCHY_LEVELS
    assert "bcm" in _schema_levels("subject_level")
    assert "bcm" in Level.__args__
    assert "bcm" in prompt_builder._ir_schema()


def test_unit_is_gone_everywhere():
    """Phase 3 deleted it (Advisor.unit had zero production rows); the
    prompt kept teaching it."""
    assert "unit" not in hierarchy.HIERARCHY_LEVELS
    assert "unit" not in _schema_levels("subject_level")
    assert "unit" not in Level.__args__
    # As a standalone level token in the prompt's union.
    assert '"unit"' not in prompt_builder._level_union()


def test_business_center_is_an_accepted_alias_but_not_a_level():
    """Backward compatibility: a stored QueryIR or an older client may
    still say `business_center`, and it must validate and canonicalise —
    but it is not offered to the LLM as a level, because `office` is the
    canonical name for that column."""
    assert "business_center" in Level.__args__
    assert QueryIR(intent="leaderboard", subject_level="business_center").subject_level == "business_center"
    assert hierarchy.canonical_level("business_center") == "office"

    assert "business_center" not in hierarchy.HIERARCHY_LEVELS
    assert "business_center" not in _schema_levels("subject_level")


@pytest.mark.parametrize("attribute", ["company", "office", "region"])
def test_attributes_are_valid_addressable_levels(attribute):
    """F2 directly. These are groupable and rankable; the schema forbade
    all three."""
    assert attribute in hierarchy.ATTRIBUTE_LEVELS
    assert attribute in hierarchy.HIERARCHY_LEVELS
    assert attribute in _schema_levels("subject_level")
    assert attribute in Level.__args__
    assert QueryIR(intent="leaderboard", subject_level=attribute).subject_level == attribute


@pytest.mark.parametrize("attribute", ["company", "office", "region"])
def test_attributes_still_do_not_nest(attribute):
    """Addressable is not the same as being in the chain — the prompt's
    old "company -> region -> unit_head" claimed they contained one
    another, which is what Phase 1 disproved."""
    assert not hierarchy.is_chain_level(attribute)
    assert hierarchy.parent_of(attribute) is None
    assert hierarchy.child_of(attribute) is None


# ---------------------------------------------------------------------
# The three required examples, end to end
# ---------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Top companies by revenue", "company"),
    ("Top BCMs by revenue", "bcm"),
])
def test_a_ranking_resolves_to_the_level_the_user_named(db_session, text, expected):
    cleaned = normalize(text)
    plan = build_query_plan(cleaned, extract_entities(cleaned, db_session))

    assert plan.action == "leaderboard"
    assert plan.level == expected
    # And the level the planner chose is one the LLM could also have
    # produced — the two paths must be able to express the same answer.
    assert plan.level in _schema_levels("subject_level")


def test_which_unit_head_manages_x_resolves_to_unit_head(db_session):
    """The third required example. "which <role> manages X" named the
    role FIRST, which no REVERSE_RE branch matched — every other branch
    expects the role to trail its subject ("X's unit head"). The query
    was answered with X's own profile."""
    from app.database.models import Advisor
    from app.llm import entity_extractor

    db_session.add(Advisor(wid=1, name="Ahmed Raza", team="Blue Area",
                           rm="UH Ali", in_master_sheet=True))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0

    cleaned = normalize("Which Unit Head manages Ahmed Raza?")
    plan = build_query_plan(cleaned, extract_entities(cleaned, db_session))

    assert plan.action == "reverse_hierarchy"
    assert plan.level == "unit_head"


@pytest.mark.parametrize("text,expected", [
    ("Which Unit Head manages Ahmed Raza?", "unit_head"),
    ("which zonal head oversees Ahmed Raza", "zonal_head"),
    ("which bcm leads Ahmed Raza", "bcm"),
    # The phrasings that already worked must keep working.
    ("who is Ahmed Raza's unit head", "unit_head"),
    ("who does Ahmed Raza report to", "unit_head"),
])
def test_reverse_lookup_vocabulary(db_session, text, expected):
    from app.database.models import Advisor
    from app.llm import entity_extractor

    db_session.add(Advisor(wid=1, name="Ahmed Raza", team="Blue Area",
                           rm="UH Ali", in_master_sheet=True))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0

    cleaned = normalize(text)
    plan = build_query_plan(cleaned, extract_entities(cleaned, db_session))

    assert plan.action == "reverse_hierarchy", text
    assert plan.level == expected, text


# ---------------------------------------------------------------------
# No hardcoded second copy survives
# ---------------------------------------------------------------------

def test_the_prompt_states_no_hierarchy_the_registry_disagrees_with():
    """The regression guard, aimed at the STRUCTURAL claims — the level
    union and the chain prose. If someone re-introduces a hand-written
    list there it will name a level the registry no longer has, which is
    exactly how F3 happened.

    Deliberately not a scan of the whole prompt: "unit" legitimately
    appears further down as a user SYNONYM for unit_head (the registry
    declares it in LEVEL_KEYWORDS), and a keyword is not a level.
    """
    union = prompt_builder._level_union()
    chain = prompt_builder._chain_description()

    for dead in ("unit", "business_center"):
        assert f'"{dead}"' not in union, dead
        assert f"{dead} (" not in chain, dead

    # The old chain read "... -> team -> advisor" with team at the
    # BOTTOM. The verified one starts there.
    assert chain.startswith("team (")
    assert chain.endswith("advisor (Advisor)")
    assert "company" not in chain and "region" not in chain


def test_adding_a_level_would_reach_every_consumer():
    """The property that makes this a single source: every consumer reads
    the registry rather than restating it."""
    from app.llm import llm_client, planner_prompt

    assert llm_client.HIERARCHY_LEVELS is hierarchy.HIERARCHY_LEVELS
    for level in hierarchy.HIERARCHY_LEVELS:
        assert level in planner_prompt._entity_types()


# ---------------------------------------------------------------------
# Phase 5.2 — the PLANNER path, and drift prevention
# ---------------------------------------------------------------------

def test_the_planner_schema_equals_the_registry():
    """F2 on the second LLM path. planner_schema had its own hand-written
    level list that omitted `bcm` and `office`/`region` while still naming
    the retired `business_center` — and its JSON enum is grammar-enforced
    too, so the planner could not emit a BCM or a region at all."""
    from app.llm import planner_schema

    enum = planner_schema.QUERY_PLAN_JSON_SCHEMA["properties"]["entities"]["items"]["properties"]["type"]["enum"]
    assert list(enum) == list(hierarchy.HIERARCHY_LEVELS)


def test_the_planner_entity_type_accepts_the_registry_plus_aliases():
    from app.llm import planner_schema

    expected = set(hierarchy.HIERARCHY_LEVELS) | set(hierarchy.LEVEL_ALIASES)
    assert set(planner_schema.EntityType.__args__) == expected


def test_every_llm_facing_level_enum_is_the_same_list():
    """The property requirement 7 asks for, stated once over ALL of them:
    prompt == schema == registry, for both LLM paths."""
    from app.llm import planner_schema

    registry = list(hierarchy.HIERARCHY_LEVELS)
    ir_enums = [
        _schema_levels("subject_level"), _schema_levels("subjects"), _schema_levels("group_by"),
    ]
    planner_enum = planner_schema.QUERY_PLAN_JSON_SCHEMA["properties"]["entities"]["items"]["properties"]["type"]["enum"]

    for enum in [*ir_enums, planner_enum]:
        assert list(enum) == registry


@pytest.mark.parametrize("level", [
    "advisor", "bcm", "zonal_head", "unit_head", "team", "company", "office", "region",
])
def test_requirement_4_every_level_is_supported_everywhere(level):
    """The eight levels named in the phase requirements, checked at every
    layer a query passes through."""
    from app.llm import planner_schema

    assert level in hierarchy.HIERARCHY_LEVELS, "registry"
    assert level in _schema_levels("subject_level"), "IR schema"
    assert level in planner_schema.QUERY_PLAN_JSON_SCHEMA["properties"]["entities"]["items"]["properties"]["type"]["enum"], "planner schema"
    assert level in Level.__args__, "QueryIR.Level"
    assert level in planner_schema.EntityType.__args__, "planner EntityType"
    assert level in prompt_builder._ir_schema(), "IR prompt"
    assert level in hierarchy.LEVEL_COLUMNS, "column binding"
    assert level in hierarchy.LEVEL_KEYWORDS, "user vocabulary"


def test_every_level_appears_in_a_worked_example():
    """Requirement 6. The schema accepted bcm/office/region and the prompt
    described them, while every worked example still showed only
    advisor/team/unit_head/zonal_head — so the shapes the model actually
    imitates never included them. A level nobody demonstrates is a level
    the model will not reach for."""
    from app.llm.ir_examples import render_examples

    rendered = render_examples()
    for level in hierarchy.HIERARCHY_LEVELS:
        assert level in rendered, f"no example demonstrates {level}"


def test_no_module_hardcodes_a_level_list():
    """Drift prevention (requirement 8), enforced structurally rather than
    by review. Every LLM-facing level list is derived; a literal one
    reappearing is how F2 and F3 happened in the first place.

    Scoped to the modules that actually feed the model or the trace —
    metric_ontology's per-metric `entity_levels` are binding declarations,
    not a hierarchy definition, and are deliberately not covered.
    """
    import ast
    import pathlib

    watched = [
        "app/llm/planner_schema.py", "app/llm/llm_client.py",
        "app/llm/prompt_builder.py", "app/llm/planner_prompt.py",
        "app/llm/query_ir.py", "app/core/tracing.py", "app/api/leaderboard.py",
    ]
    chain = set(hierarchy.CHAIN)

    for module in watched:
        tree = ast.parse(pathlib.Path(module).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.List, ast.Tuple)):
                continue
            values = {
                el.value for el in node.elts
                if isinstance(el, ast.Constant) and isinstance(el.value, str)
            }
            # Two or more chain levels written out together is a level
            # list, whatever it is called.
            assert len(values & chain) < 2, (
                f"{module} hardcodes levels {sorted(values & chain)} — derive them "
                "from hierarchy.HIERARCHY_LEVELS instead"
            )


def test_the_reverse_lookup_level_set_is_derived():
    """llm_planner picked the manager level from a literal tuple that
    omitted `bcm`, so "who is X's BCM" fell through to the default level
    and answered about a different manager."""
    from app.llm.llm_planner import _reverse_level
    from app.llm.planner_schema import LLMQueryPlan, PlannedEntity

    for level in ("unit_head", "zonal_head", "bcm"):
        plan = LLMQueryPlan(
            intent="reverse_hierarchy",
            entities=[PlannedEntity(type=level, value="Someone")],
        )
        assert _reverse_level(plan) == level, level


def test_the_trace_records_every_level():
    """A misroute involving bcm/office/region was invisible in the trace
    built to explain misroutes."""
    from app.core.tracing import _traced_entity_keys

    keys = _traced_entity_keys()
    for level in hierarchy.GROUP_LEVELS:
        assert level in keys, level
        assert hierarchy.LEVEL_ENTITY_KEYS[level] in keys, level
