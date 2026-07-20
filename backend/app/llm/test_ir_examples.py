"""Every few-shot example shown to the LLM must be a real, valid QueryIR
that survives the same validation pipeline the LLM's actual output goes
through — otherwise the prompt teaches the model a shape the backend
rejects."""

import pytest

from app.database.models import Advisor
from app.llm import entity_extractor
from app.llm.ir_examples import EXAMPLES, render_examples
from app.llm.ir_validator import validate_ir
from app.llm.metric_ontology import METRICS
from app.llm.query_ir import QueryIR


@pytest.fixture()
def example_gazetteer_db(db_session):
    # the fictional entities the examples reference must exist in the
    # gazetteer for validate_ir's subject grounding to accept them
    db_session.add_all([
        Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana"),
        Advisor(wid=2, name="Ali Raza", team="Downtown", company="IMARAT"),
    ])
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def test_every_example_parses_as_query_ir():
    for ex in EXAMPLES:
        ir = QueryIR.model_validate(ex["ir"])
        assert ir.intent
        if ex["prior_ir"]:
            QueryIR.model_validate(ex["prior_ir"])


def test_every_example_metric_key_exists_in_ontology():
    for ex in EXAMPLES:
        metric = ex["ir"]["metric"]
        if metric:
            assert metric["key"] in METRICS, f"{ex['utterance']}: unknown metric {metric['key']}"
        for f in ex["ir"]["filters"]:
            if f["field"] not in ("team", "company", "advisor", "attendance_status"):
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
