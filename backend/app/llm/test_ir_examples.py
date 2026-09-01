"""Every few-shot example shown to the LLM must be a real, valid QueryIR
that survives the same validation pipeline the LLM's actual output goes
through — otherwise the prompt teaches the model a shape the backend
rejects."""

import pytest

from app.database.models import Advisor
from app.llm import entity_extractor
from app.llm.ir_examples import EXAMPLES, render_examples
from app.llm.ir_validator import validate_ir
from app.llm.llm_client import QUERY_IR_JSON_SCHEMA
from app.llm.metric_ontology import METRICS
from app.llm.query_ir import QueryIR


@pytest.fixture()
def example_gazetteer_db(db_session):
    # the fictional entities the examples reference must exist in the
    # gazetteer for validate_ir's subject grounding to accept them
    db_session.add_all([
        Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana"),
        Advisor(wid=2, name="Ali Raza", team="Downtown", company="IMARAT"),
        # the zonal_head comparison example grounds its two subjects against
        # Advisor.zm (see hierarchy.py's column mapping) — these rows exist
        # purely so validate_ir's subject grounding has a real match.
        Advisor(wid=3, name="Fake Advisor Three", team="Blue Area", company="Graana", zm="Ahmed Ali", portfolio_lead="Ahmed Ali"),
        Advisor(wid=4, name="Fake Advisor Four", team="Downtown", company="IMARAT", zm="Bilal Khan", portfolio_lead="Bilal Khan"),
        # the "breakdown" example grounds its single subject against
        # Advisor.bm — same reasoning as the zonal_head rows above.
        Advisor(wid=5, name="Fake Advisor Five", team="Blue Area", company="Graana", bm="Zeeshan Tariq", rm="Zeeshan Tariq"),
    ])
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def test_every_example_parses_as_query_ir():
    """CONTRACT 2. Every example must be a real QueryIR.

    Six examples carried `intent: "population"`, which is not a member of
    the Intent literal, so they could not be built by the very model they
    were teaching. A few-shot example is the strongest signal in the
    prompt; an invalid one teaches a shape the backend rejects.
    """
    for i, ex in enumerate(EXAMPLES):
        ir = QueryIR.model_validate(ex["ir"])
        assert ir.intent
        if ex["prior_ir"]:
            QueryIR.model_validate(ex["prior_ir"])


def test_every_example_is_emittable_under_the_output_grammar():
    """CONTRACT 1. Structured Outputs runs with strict:True, so decoding
    is CONSTRAINED to the schema — an operation or intent outside the enum
    is not merely discouraged, it is unrepresentable.

    Eight examples named values outside it: `breakdown` and `clarify` as
    operations, and `population` as an intent. Imitating them exactly was
    impossible, so on those shapes the model had to depart from the
    examples and was given nothing to depart TOWARD.

    Asserted against the schema itself rather than a copied list, so the
    two cannot drift.
    """
    operations_enum = QUERY_IR_JSON_SCHEMA["properties"]["operation"]["enum"]
    intents_enum = QUERY_IR_JSON_SCHEMA["properties"]["intent"]["enum"]

    for ex in EXAMPLES:
        ir = ex["ir"]
        assert ir.get("operation") in operations_enum, (
            f"{ex['utterance']!r}: operation {ir.get('operation')!r} is not in "
            f"the output grammar {operations_enum} — the model cannot emit it"
        )
        assert ir.get("intent") in intents_enum, (
            f"{ex['utterance']!r}: intent {ir.get('intent')!r} is not in "
            f"the output grammar {intents_enum} — the model cannot emit it"
        )


def test_every_offered_operation_is_actually_demonstrated():
    """Prose is the weakest way to specify a structure; an operation the
    prompt offers but never shows is specified only in prose.

    `group_metric` — "one named entity's own figure", the most ordinary
    analytical question there is, and one of only three plan actions that
    reach the model at all — had no example.
    """
    shown = {ex["ir"].get("operation") for ex in EXAMPLES}
    offered = set(QUERY_IR_JSON_SCHEMA["properties"]["operation"]["enum"])
    assert offered <= shown, f"offered but never demonstrated: {sorted(offered - shown)}"


def test_no_hierarchy_example_inverts_the_chain():
    """A hierarchy read enumerates a level BENEATH a subject, so the
    target must sit lower in the chain than the scope.

    Three examples put `team` beneath `bcm` / `zonal_head`. `team` is the
    ROOT (team -> unit_head -> zonal_head -> bcm -> advisor), so they
    described a traversal running upwards — a shape `ir_validator` refuses
    outright. They could never have been imitated successfully, and what
    they taught was an org chart this business does not have.
    """
    from app.llm import hierarchy

    for ex in EXAMPLES:
        target = ex["ir"].get("target_level")
        scope = ex["ir"].get("subject_of")
        if not target or not scope:
            continue
        depth_target, depth_scope = hierarchy.depth(target), hierarchy.depth(scope)
        assert depth_target is not None and depth_scope is not None, (
            f"{ex['utterance']!r}: {scope!r} or {target!r} is not a chain level — "
            "an attribute cannot be traversed through"
        )
        assert depth_target > depth_scope, (
            f"{ex['utterance']!r}: {target!r} is not beneath {scope!r} in "
            f"{hierarchy.CHAIN} — the validator refuses this pairing"
        )


def test_every_example_metric_key_exists_in_ontology():
    for ex in EXAMPLES:
        metric = ex["ir"]["metric"]
        if metric:
            assert metric["key"] in METRICS, f"{ex['utterance']}: unknown metric {metric['key']}"
        # A filter field is a metric key OR a hierarchy level OR
        # attendance_status — exactly what the prompt tells the model and
        # what query_compiler accepts.
        #
        # The non-metric list used to be hand-written here and had gone
        # stale the same way every other copy did: it named the retired
        # `business_center` and omitted bcm/office/region, so a legitimate
        # example filtering on a region was rejected by the test rather
        # than by the system. Derived now.
        from app.llm import hierarchy

        level_fields = set(hierarchy.HIERARCHY_LEVELS) | set(hierarchy.LEVEL_ALIASES)
        for f in ex["ir"]["filters"]:
            if f["field"] not in level_fields and f["field"] != "attendance_status":
                assert f["field"] in METRICS, f"{ex['utterance']}: unknown filter field {f['field']}"


def test_examples_survive_full_validation(example_gazetteer_db):
    for ex in EXAMPLES:
        result = validate_ir(QueryIR.model_validate(ex["ir"]), example_gazetteer_db)
        if ex["expect_valid"]:
            assert result.is_valid, f"{ex['utterance']}: unexpectedly invalid — {result.missing}"
        else:
            assert not result.is_valid, f"{ex['utterance']}: expected to trip the validator"


def test_render_examples_includes_every_utterance():
    block = render_examples()
    for ex in EXAMPLES:
        assert ex["utterance"] in block
