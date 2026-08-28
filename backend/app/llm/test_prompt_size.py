"""
Prompt size is a behavioural property, not a style preference.

Every token in the parser prompt is re-read by the model on every single
turn, and prompt evaluation is the dominant term in this system's latency
(see llm_client._log_llm_call's prompt_eval_ms). Before TASK 3 an ordinary
analytical query built a ~32,400-character prompt; three quarters of it
was information the model could not act on differently:

  - the QueryIR shape, restated in prose although Ollama already
    constrains decoding with QUERY_IR_JSON_SCHEMA;
  - 44 metric synonym lists, which ARE metric_aliases.ALIASES, matched
    deterministically before the LLM is ever called;
  - 269 people's names, on queries that name no person.

These tests pin the reductions so the prompt cannot silently grow back,
and — more importantly — pin the things that must NOT be dropped to
achieve them.
"""

import pytest

from app.llm import hierarchy
from app.llm.ir_examples import EXAMPLES, render_examples
from app.llm.prompt_builder import _ir_schema, build_ir_prompt

TEAMS = ["Blue Area", "DownTown"]
COMPANIES = ["Graana", "IMARAT"]
# Deliberately realistic: the real gazetteer is ~269 names.
PEOPLE = [f"Person Number{i}" for i in range(90)]


def _prompt(text, **kw):
    kw.setdefault("grounded_entities", {})
    return build_ir_prompt(text, TEAMS, COMPANIES, **kw)


def _tokens(text):
    """Ollama reports real token counts; offline we approximate at 4
    chars/token, which is only ever used for RELATIVE assertions here."""
    return len(text) // 4


ANALYTICAL = "top 5 advisors by connects"


def test_an_ordinary_analytical_prompt_stays_well_under_the_old_size():
    """The regression this file exists for. 8,104 tokens before."""
    assert _tokens(_prompt(ANALYTICAL)) < 6000


def test_naming_no_person_omits_every_person_gazetteer():
    """~1,100 tokens of names on a query with no name in it."""
    body = _prompt(
        ANALYTICAL,
        known_unit_heads=PEOPLE,
        known_zonal_heads=PEOPLE,
        known_bcms=PEOPLE,
    )
    assert "Person Number1" not in body


def test_naming_a_person_still_gets_that_gazetteer_in_full():
    body = _prompt(
        "how is Person Number7 doing",
        known_bcms=PEOPLE,
    )
    for name in PEOPLE:
        assert name in body, f"{name} was dropped from a list the message reaches into"


def test_a_misspelled_name_still_pulls_the_gazetteer_in():
    """Entity extraction fuzzy-matches; the prompt must not pre-empt it."""
    body = _prompt("how is Persson Number7 doing", known_bcms=PEOPLE)
    assert "Person Number7" in body


def test_an_already_grounded_level_drops_its_gazetteer():
    key = hierarchy.LEVEL_ENTITY_KEYS["bcm"]
    body = _prompt("how is Person Number7 doing", known_bcms=PEOPLE,
                   grounded_entities={key: ["Person Number7"]})
    assert "Person Number3" not in body
    # ...but the resolved value itself is still stated.
    assert "Person Number7" in body


def test_a_resolved_metric_phrase_retires_the_synonym_table():
    body = _prompt(ANALYTICAL)
    assert "phrasings:" not in body
    assert "connects" in body


def test_an_unresolved_phrase_keeps_the_full_synonym_table():
    """"struggling" matches no alias — the table is the only bridge left."""
    body = _prompt("who is struggling")
    assert "phrasings:" in body


def test_every_metric_key_is_always_offered_whichever_catalog_is_used():
    """Trimming synonyms must never trim the VOCABULARY: the model has to
    stay able to pick a metric the alias resolver never considered."""
    from app.llm.metric_ontology import METRICS

    for text in (ANALYTICAL, "who is struggling"):
        body = _prompt(text)
        for key in METRICS:
            assert key in body, f"{key} unreachable for {text!r}"


def test_the_schema_block_no_longer_restates_what_the_grammar_enforces():
    schema = _ir_schema()
    assert '"operator": "="' not in schema
    assert '"mode": "snapshot"|"compare"' not in schema


@pytest.mark.parametrize("guidance", [
    "OPERATION.",
    "POPULATION vs RANKING.",
    "TWO QUESTIONS IN ONE MESSAGE.",
    "MULTIPLE MEASURES.",
    "BOOLEAN FILTERS.",
    "GROUPING.",
    "PERIOD COMPARISON.",
])
def test_semantics_the_json_schema_cannot_express_are_retained(guidance):
    assert guidance in _ir_schema()


def test_no_worked_example_was_dropped():
    """Compaction removed default-valued FIELDS, never examples."""
    block = render_examples()
    assert len(EXAMPLES) == 18
    for ex in EXAMPLES:
        assert ex["utterance"] in block


def test_examples_omit_defaults_but_keep_substance():
    block = render_examples()
    assert '"subjects": []' not in block
    assert '"filter_tree": null' not in block
    assert '"intent": "leaderboard"' in block
    assert "you must still emit every field" in block
