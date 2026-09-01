"""The validator rewrites meaning. Those rewrites must be legible.

`ir_validator` does not only reject — it REPAIRS. It normalises the intent
from the registry, copies a subject's level over `subject_level`, corrects
a near-miss metric key, re-types a subject the parser mislabelled, drops
one it cannot ground, prunes an unusable filter, re-labels a metric-free
list as a population, and replaces a placeholder sort direction. Each is
defensible; every one of them CHANGES WHAT THE QUERY MEANS.

Until now they were invisible. The logs held the model's raw output and
the final IR with nothing in between, so "the LLM got it wrong" and "we
rewrote it afterwards" were indistinguishable after the fact — the single
hardest thing to establish when a production answer is wrong.

`ir.repairs` records each as {field, from, to, why}, so the raw parse is
reconstructible by replaying them backwards from the final IR.
"""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod
from app.llm import entity_extractor, hierarchy, routing
from app.llm.conversation_context import Ellipsis, TurnSpec, merge
from app.llm.ir_validator import (
    build_targeted_clarification, pick_clarification_slot, validate_ir,
)
from app.llm.query_ir import (
    Filter, FilterGroup, MetricRef, QueryIR, Sort, Subject, TimeRange,
)


@pytest.fixture()
def org(db_session):
    for wid, name, team in [(1, "Ali Raza", "AMD"), (2, "Sana Tariq", "AMD")]:
        db_session.add(Advisor(wid=wid, name=name, team=team, company="Graana",
                               in_master_sheet=True))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=10 * wid, pct=10.0))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _ir(**overrides):
    base = dict(intent="filtered_list", operation="filtered_list",
                subject_level="advisor", sort=Sort(metric=None), limit=None)
    base.update(overrides)
    return QueryIR(**base)


def _fields(ir):
    return [r["field"] for r in ir.repairs]


# =====================================================================
# 5.1 — a structural contradiction is not a question about a name
# =====================================================================
class TestTheInvertedHierarchyMessage:
    """`missing[]` entries are STRUCTURED: `_ask_for` splits a "subject:"
    entry on ":" into a level and a value. This one was built as

        f"subject:{ir.subject_of}:{ir.target_level} is not beneath it"

    so the value became the literal string "team is not beneath it", and
    "How many people are in ZH1's team?" was answered with

        "which zonal head you meant by 'team is not beneath it'?"

    A contradiction in the parse, rendered as a question about a name
    nobody typed.
    """

    def _inverted(self):
        return _ir(operation="population", metric=None, subject_level="team",
                   target_level="team", subject_of="zonal_head",
                   subjects=[Subject(type="zonal_head", value="ZH1",
                                     match_confidence=1.0)])

    def test_the_contradiction_gets_its_own_slot(self, org):
        missing = validate_ir(self._inverted(), org).missing
        assert any(m.startswith("inverted_hierarchy:") for m in missing), missing

    def test_no_slot_smuggles_prose_into_a_value(self, org):
        """The shape of the bug, guarded generally: a ':'-delimited slot's
        value must not be a sentence."""
        for item in validate_ir(self._inverted(), org).missing:
            parts = item.split(":")
            for part in parts[1:]:
                assert " is not " not in part, item

    def test_the_question_names_neither_a_field_nor_a_phantom_entity(self, org):
        missing = validate_ir(self._inverted(), org).missing
        message = build_targeted_clarification(missing)

        assert "is not beneath it" not in message
        for leak in ("target_level", "subject_of", "inverted_hierarchy", "missing"):
            assert leak not in message, message

    def test_it_explains_which_way_the_chain_actually_runs(self, org):
        message = build_targeted_clarification(
            ["inverted_hierarchy:zonal_head:team"])
        assert "Zonal Heads sit under Teams" in message

    def test_the_contradiction_outranks_the_other_slots(self, org):
        """The same parse also sets `subjects`, and "which two things would
        you like to compare?" is a worse question still. Nothing the user
        can supply fixes an impossible shape, so it is asked about first."""
        missing = validate_ir(self._inverted(), org).missing
        assert pick_clarification_slot(missing).startswith("inverted_hierarchy:")


# =====================================================================
# 5.2 — only a comparison inherits its operation
# =====================================================================
class TestOperationInheritanceIsGatedOnComparison:
    """The condition read "the previous turn had ANY subjects", while its
    own comment said "the previous turn was a comparison". Most turns have
    subjects, so an elliptical follow-up took the PREVIOUS turn's
    operation over the one it had just been parsed as."""

    def _prior(self, operation, intent, n_subjects=1):
        subjects = [Subject(type="team", value=v, match_confidence=1.0)
                    for v in ["AMD", "Blue Area"][:n_subjects]]
        return _ir(operation=operation, intent=intent, subject_level="team",
                   subjects=subjects, metric=MetricRef(key="mtd_cleared"),
                   sort=Sort(metric="mtd_cleared"))

    def _elliptical(self):
        return TurnSpec(metric=False, subject=False, level_word=False,
                        period=False, limit=True, ranking=True, comparison=False)

    def test_a_follow_up_after_a_group_metric_keeps_its_own_shape(self):
        """"What is AMD's revenue?" then "top 5" is a request to RANK. It
        came back as one group's figure."""
        current = _ir(operation="leaderboard", intent="leaderboard",
                      subject_level="advisor", limit=5)
        merge(self._prior("group_metric", "filtered_list"), current,
              self._elliptical(), Ellipsis(True, "test"))

        assert current.resolved_operation() == "leaderboard"

    def test_a_follow_up_after_a_leaderboard_keeps_its_own_shape(self):
        current = _ir(operation="population", intent="filtered_list", metric=None)
        merge(self._prior("leaderboard", "leaderboard"), current,
              self._elliptical(), Ellipsis(True, "test"))

        assert current.resolved_operation() == "population"

    def test_a_comparison_still_carries_forward(self):
        """The case the inheritance exists for: dropping the sides degrades
        a two-sided question into a ranking of everything."""
        current = _ir(operation="leaderboard", intent="leaderboard")
        merge(self._prior("comparison", "comparison", n_subjects=2), current,
              self._elliptical(), Ellipsis(True, "test"))

        assert current.resolved_operation() == "comparison"

    def test_the_previous_subjects_still_carry_either_way(self):
        """Only the OPERATION is now gated. Subject inheritance is a
        separate rule and must not have been narrowed with it."""
        current = _ir(operation="leaderboard", intent="leaderboard")
        merge(self._prior("group_metric", "filtered_list"), current,
              self._elliptical(), Ellipsis(True, "test"))

        assert [s.value for s in current.subjects] == ["AMD"]


# =====================================================================
# 5.3 — an unstated direction is the measure's own
# =====================================================================
class TestSortDirection:
    """`sort.metric` is null exactly when the query expressed no ranking.
    The direction beside it is then a placeholder — and gpt-4o-mini emits
    "asc" for it. The compiler applies it literally, and `primary_metric()`
    falls back to a measure named in a FILTER, so "advisors with connects
    above 1000" ranked the qualifying advisors WORST-FIRST."""

    def test_an_unstated_direction_follows_the_metric_for_higher_is_better(self, org):
        ir = _ir(sort=Sort(metric=None, direction="asc"),
                 filters=[Filter(field="total_connects", operator=">", value=1000)])
        assert validate_ir(ir, org).ir.sort.direction == "desc"

    def test_an_unstated_direction_follows_the_metric_for_lower_is_better(self, org):
        """`overdue` is a measure where less is better, so ascending IS
        correct for it — the rule is the metric's polarity, not a constant."""
        ir = _ir(sort=Sort(metric=None, direction="desc"),
                 filters=[Filter(field="overdue", operator=">", value=5)])
        assert validate_ir(ir, org).ir.sort.direction == "asc"

    def test_an_explicit_ranking_is_never_touched(self, org):
        """"bottom 5 by connects" sets sort.metric and means ascending."""
        ir = _ir(operation="leaderboard", intent="leaderboard",
                 metric=MetricRef(key="total_connects"),
                 sort=Sort(metric="total_connects", direction="asc"))
        result = validate_ir(ir, org)
        assert result.ir.sort.direction == "asc"
        assert "sort.direction" not in _fields(result.ir)

    def test_the_correction_is_recorded(self, org):
        ir = _ir(sort=Sort(metric=None, direction="asc"),
                 filters=[Filter(field="total_connects", operator=">", value=1000)])
        repairs = validate_ir(ir, org).ir.repairs
        assert any(r["field"] == "sort.direction" and r["to"] == "desc"
                   for r in repairs)


# =====================================================================
# 5.5 — every rewrite is recorded, in one shape
# =====================================================================
class TestEveryRewriteIsRecorded:

    def test_intent_rewrite(self, org):
        ir = _ir(operation="leaderboard", intent="filtered_list",
                 metric=MetricRef(key="mtd_cleared"), sort=Sort(metric="mtd_cleared"))
        assert "intent" in _fields(validate_ir(ir, org).ir)

    def test_operation_rewrite(self, org):
        ir = _ir(operation="filtered_list",
                 filters=[Filter(field="team", value="AMD")])
        assert "operation" in _fields(validate_ir(ir, org).ir)

    def test_subject_level_rewrite(self, org):
        ir = _ir(operation="group_metric", subject_level="advisor",
                 metric=MetricRef(key="mtd_cleared"),
                 subjects=[Subject(type="team", value="AMD", match_confidence=1.0)])
        assert "subject_level" in _fields(validate_ir(ir, org).ir)

    def test_fuzzy_metric_correction(self, org):
        ir = _ir(operation="leaderboard", intent="leaderboard",
                 metric=MetricRef(key="achievement"),
                 sort=Sort(metric="achievement"))
        result = validate_ir(ir, org)
        assert "metric.key" in _fields(result.ir)
        assert result.ir.metric.key == "achievement_pct"

    def test_filter_pruning(self, org):
        ir = _ir(operation="leaderboard", intent="leaderboard",
                 metric=MetricRef(key="mtd_cleared"), sort=Sort(metric="mtd_cleared"),
                 filters=[Filter(field="not_a_field", value="x")])
        assert "filters" in _fields(validate_ir(ir, org).ir)

    def test_ungroundable_subject_removal(self, org):
        ir = _ir(operation="group_metric", subject_level="team",
                 metric=MetricRef(key="mtd_cleared"),
                 subjects=[Subject(type="team", value="Atlantis", match_confidence=1.0)])
        result = validate_ir(ir, org)
        assert "subjects" in _fields(result.ir)
        assert result.ir.subjects == []

    def test_subject_retyping(self, org):
        ir = _ir(operation="group_metric", subject_level="company",
                 metric=MetricRef(key="mtd_cleared"),
                 subjects=[Subject(type="company", value="AMD", match_confidence=1.0)])
        assert "subjects[].type" in _fields(validate_ir(ir, org).ir)

    def test_a_clean_parse_records_nothing(self, org):
        """The property that makes the list worth reading: it is empty
        when the validator changed nothing."""
        ir = _ir(operation="leaderboard", intent="leaderboard",
                 metric=MetricRef(key="mtd_cleared"),
                 sort=Sort(metric="mtd_cleared", direction="desc"))
        assert validate_ir(ir, org).ir.repairs == []


class TestTheRecordIsUsable:

    def test_every_entry_has_the_same_four_keys(self, org):
        ir = _ir(operation="group_metric", subject_level="company",
                 metric=MetricRef(key="achievement"),
                 subjects=[Subject(type="company", value="AMD", match_confidence=1.0)])
        for r in validate_ir(ir, org).ir.repairs:
            assert set(r) == {"field", "from", "to", "why"}
            assert r["why"], r

    def test_the_raw_parse_is_reconstructible(self, org):
        """The debugging requirement: raw LLM IR -> modification -> final
        IR. Replaying the repairs backwards from the final IR recovers
        what the model actually said."""
        ir = _ir(operation="group_metric", subject_level="company",
                 metric=MetricRef(key="mtd_cleared"),
                 subjects=[Subject(type="company", value="AMD", match_confidence=1.0)])
        result = validate_ir(ir, org)

        level = result.ir.subject_level
        for r in reversed(result.ir.repairs):
            if r["field"] == "subject_level":
                level = r["from"]
        assert level == "company"

    def test_it_reaches_the_request_trace_too(self, org):
        """One repair, both sinks — structured for replay, readable in the
        trace beside every other decision the request made."""
        routing.start_trace("test")
        ir = _ir(operation="group_metric", subject_level="company",
                 metric=MetricRef(key="mtd_cleared"),
                 subjects=[Subject(type="company", value="AMD", match_confidence=1.0)])
        validate_ir(ir, org)
        rendered = routing.current_trace().render()
        assert "Repair:" in rendered

    def test_it_is_not_offered_to_the_model(self, org):
        """Observability only: `repairs` must never appear in the output
        grammar, or the model could author its own repair history."""
        from app.llm.llm_client import QUERY_IR_JSON_SCHEMA

        assert "repairs" not in QUERY_IR_JSON_SCHEMA["properties"]
        assert "repairs" not in QUERY_IR_JSON_SCHEMA["required"]


# =====================================================================
# 5.4 — answering a clarification must not change the question
# =====================================================================
class TestFillingAPendingSlot:
    """`_fill_pending_slot` merges a short reply ("revenue", "Blue Area")
    into the partial IR we asked about. Two defects lived in it.

    IT PROMOTED EVERY FILLED IR TO A RANKING. A population that asked
    which group the user meant came back as a `leaderboard` once they
    said — attaching a measure the question never had. Every measure is
    read through its own table, so the join drops the people with no row
    in it and the list returns SHORTER than the truth: the exact
    regression `population` exists to prevent.

    AND IT COULD ONLY FILL TWO LEVELS. It read `teams` and `companies`,
    while extraction grounds seven. Asked "which unit head did you mean?",
    a reply naming one filled nothing, so the message fell through as a
    brand-new query and the same question was asked again next turn.
    """

    def _pending(self, ir, missing):
        from app.llm.conversation_memory import PendingClarification

        return PendingClarification(partial_ir=ir, missing=missing)

    def _fill(self, ir, missing, text, entities):
        from app.llm.nlu_pipeline import _fill_pending_slot

        return _fill_pending_slot(self._pending(ir, missing), text, entities)

    # ---- the shape survives ------------------------------------------
    def test_a_population_stays_a_population(self):
        partial = _ir(operation="population", intent="clarify", metric=None,
                      subject_level="advisor")
        filled = self._fill(partial, ["subject"], "Blue Area",
                            {"teams": ["Blue Area"]})

        assert filled is not None
        assert filled.resolved_operation() == "population"
        assert filled.primary_metric() is None

    def test_a_filled_measure_still_produces_a_ranking(self):
        """The other half: when the answer supplies a MEASURE, a ranking
        is exactly what was asked for."""
        partial = _ir(operation="clarify_metric", intent="clarify", metric=None)
        filled = self._fill(partial, ["metric"], "revenue", {})

        assert filled is not None
        assert filled.resolved_operation() == "leaderboard"
        assert filled.primary_metric() == "mtd_cleared"

    def test_two_subjects_and_a_measure_still_make_a_comparison(self):
        partial = _ir(operation="clarify_metric", intent="clarify",
                      metric=MetricRef(key="mtd_cleared"),
                      sort=Sort(metric="mtd_cleared"))
        filled = self._fill(partial, ["subject"], "Blue Area and Graana",
                            {"teams": ["Blue Area"], "companies": ["Graana"]})

        assert filled is not None
        assert filled.resolved_operation() == "comparison"

    # ---- every level the extractor can ground ------------------------
    @pytest.mark.parametrize("level,plural,value", [
        ("team", "teams", "Blue Area"),
        ("company", "companies", "Graana"),
        ("unit_head", "unit_heads", "Tariq Mehmood"),
        ("zonal_head", "zonal_heads", "Fawad Hafeez"),
        ("bcm", "bcms", "Usman Ghani"),
        ("office", "offices", "Beverly Center"),
        ("region", "regions", "North"),
    ])
    def test_a_reply_naming_any_groupable_level_fills_the_slot(
            self, level, plural, value):
        partial = _ir(operation="clarify_metric", intent="clarify",
                      metric=MetricRef(key="mtd_cleared"),
                      sort=Sort(metric="mtd_cleared"))
        filled = self._fill(partial, ["subject"], value, {plural: [value]})

        assert filled is not None, f"{level}: the reply filled nothing"
        assert [(s.type, s.value) for s in filled.subjects] == [(level, value)]

    def test_the_levels_are_derived_from_the_hierarchy(self):
        """The two-level list drifted out of step with the five the
        extractor already grounded. Reading the hierarchy's own map is
        what stops it happening again."""
        from app.llm.nlu_pipeline import _fill_pending_slot  # noqa: F401

        assert set(hierarchy.LEVEL_ENTITY_KEYS) >= {
            "team", "company", "unit_head", "zonal_head", "bcm"}

    def test_an_unrelated_reply_still_fills_nothing(self):
        """Widening which levels can be filled must not make every message
        look like an answer — that is what lets a genuinely new question
        be swallowed by the pending clarification."""
        partial = _ir(operation="clarify_metric", intent="clarify", metric=None)
        assert self._fill(partial, ["subject"], "what about next week", {}) is None

    def test_a_subject_already_present_is_not_duplicated(self):
        partial = _ir(operation="clarify_metric", intent="clarify",
                      metric=MetricRef(key="mtd_cleared"),
                      sort=Sort(metric="mtd_cleared"),
                      subjects=[Subject(type="team", value="Blue Area",
                                        match_confidence=1.0)])
        filled = self._fill(partial, ["subject"], "Blue Area",
                            {"teams": ["Blue Area"]})

        assert filled is None or [s.value for s in filled.subjects] == ["Blue Area"]
