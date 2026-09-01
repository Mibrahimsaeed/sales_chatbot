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
    """The regression this file exists for. 8,104 tokens before.

    The budget has been raised twice, both times deliberately: to 7,000
    for the compositional few-shot examples and the business-model block,
    and to 9,000 for the expanded business model plus the metric-null
    examples, then to 10,000 for the hierarchy-read field documentation.

    THE TEST BELOW IS WHAT MAKES THAT A TRADE RATHER THAN A RELAXATION.
    What costs latency is not the prompt's SIZE but the part re-evaluated
    on every call, and that is ~66 tokens: 99% of the prompt is a stable
    prefix the provider can reuse. A prompt that is larger and 99%
    cacheable is cheaper per query than a smaller one that is not.

    Raise this ceiling only alongside evidence that the prefix property
    below still holds — otherwise growth is paid on every single turn.

    RAISED TO 15,000 IN PHASE 2, under that rule. The restored guardrails
    (POPULATION vs RANKING, MULTIPLE MEASURES, the subject_level/subjects
    agreement, SEPARATE THING, the sort-direction default, the confidence
    threshold) and the three group_metric examples cost roughly 1,800
    tokens. Every one of them sits in the STATIC prefix, and the prefix
    property below was re-measured at 99.4% immediately after — so the
    marginal cost per query is unchanged and the marginal benefit is a
    class of wrong parse each rule was written to prevent.
    """
    assert _tokens(_prompt(ANALYTICAL)) < 15000


def test_almost_all_of_the_prompt_is_a_reusable_prefix():
    """THE latency property, and the reason the budget above could move.

    Ollama reuses the KV state of the longest prompt PREFIX it has
    already seen, so every token before the first per-query byte is free
    on the next call. The prompt used to interleave the entity dict, the
    prior IR and the conversation window with the examples and the
    schema, and put the user's message ahead of the ~1,900-token schema
    block — leaving a ~923-token stable prefix and re-prefilling 82% of
    the prompt every turn (15-19s against qwen3:8b).

    Every static block now precedes every per-query one, and the user's
    message is last. This pins that: if someone appends a static block
    after the message, or moves a per-query segment up into the prefix,
    the shared prefix collapses and this fails.
    """
    a = _prompt(ANALYTICAL, grounded_entities={"metric": "total_connects"})
    b = _prompt("top 3 teams by revenue", grounded_entities={"team": "Blue Area"})

    shared = 0
    for shared, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            break

    assert shared / len(a) > 0.9, (
        f"only {shared} of {len(a)} chars are shared prefix "
        f"({100 * shared / len(a):.0f}%) — a static block has moved below a "
        "per-query one, so every query now re-evaluates it"
    )


def test_the_user_message_is_the_last_thing_in_the_prompt():
    """Two reasons, both load-bearing: it maximises the reusable prefix,
    and under a context limit the question can never be the thing that
    gets truncated. It previously sat before the schema block."""
    body = _prompt(ANALYTICAL)
    assert body.rstrip().endswith(f'User message: "{ANALYTICAL}"')


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


# The section headings, spelled EXACTLY as the prompt spells them.
# "POPULATION VS RANKING" and "TWO INDEPENDENT QUESTIONS" were this
# file's own transcriptions of headings test_phases_1_to_4.py pins as
# "POPULATION vs RANKING" and "TWO QUESTIONS IN ONE MESSAGE" — two test
# files asserting the same fact in two spellings, which is how one of
# them ends up wrong. These follow the prompt.
@pytest.mark.parametrize("guidance", [
    "OPERATION",
    "POPULATION vs RANKING",
    "TWO QUESTIONS IN ONE MESSAGE",
    "MULTIPLE MEASURES",
    "BOOLEAN FILTERS",
    "GROUPING",
    "PERIOD COMPARISON",
    # Added with the Phase 2 guardrail restoration — each is a rule the
    # JSON schema cannot carry, and each was measured causing a specific
    # wrong parse before it existed.
    "SUBJECT_LEVEL AND SUBJECTS MUST AGREE",
    "SEPARATE THING",
    "WHEN YOU DO NOT KNOW",
    "SORT",
])
def test_semantics_the_json_schema_cannot_express_are_retained(guidance):
    assert guidance in _ir_schema()


def test_no_worked_example_was_dropped():
    """Compaction removed default-valued FIELDS, never examples."""
    block = render_examples()
    # A TRIPWIRE, not a budget: it exists so an example cannot vanish in a
    # reorganisation without someone deciding to. It has to be edited
    # deliberately when the corpus grows, which is the point — it was 27
    # when written, and Phase 2 removed one unemittable `breakdown`
    # example and added three `group_metric` ones (an operation the model
    # is offered and had never been shown), plus the hierarchy and
    # boolean examples added since.
    assert len(EXAMPLES) == 37
    for ex in EXAMPLES:
        assert ex["utterance"] in block


def test_every_compositional_shape_has_a_worked_example():
    """The prompt DESCRIBED filter_tree, metrics[] and per-subject
    metrics in prose and never once showed one, which is the weakest way
    to specify a nested structure to a small local model — the audit
    measured "advisors in Blue Area or DownTown" losing its disjunction
    entirely.

    Prose is not enough and this is what says so: each of these fields
    must appear filled in at least one example, or the field is being
    asked for without ever being demonstrated.
    """
    def any_example(predicate):
        return any(predicate(ex["ir"]) for ex in EXAMPLES)

    assert any_example(lambda ir: (ir.get("filter_tree") or {}).get("op") == "or"), \
        "no example shows a disjunction"
    assert any_example(lambda ir: (ir.get("filter_tree") or {}).get("op") == "not"), \
        "no example shows an exclusion"
    assert any_example(lambda ir: len(ir.get("metrics") or []) > 1), \
        "no example shows a two-measure question"
    assert any_example(
        lambda ir: any(s.get("metric") for s in (ir.get("subjects") or []))
    ), "no example shows a measure bound to one subject"
    assert any_example(
        lambda ir: len({f["field"] for f in (ir.get("filters") or [])}) > 1
    ), "no example shows conditions on two different measures"


def test_examples_omit_defaults_but_keep_substance():
    block = render_examples()
    assert '"subjects": []' not in block
    assert '"filter_tree": null' not in block
    assert '"intent": "leaderboard"' in block
    assert "you must still emit every field" in block
