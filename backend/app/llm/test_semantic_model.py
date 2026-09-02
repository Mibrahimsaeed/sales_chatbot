"""The semantic contract: what a user MEANT, before any database is read.

The model exists for one distinction the execution contract cannot make.
`QueryIR.subjects` means two different things depending on three other
fields, and nothing states which reading applies:

    "connects of Blue Area"              Blue Area IS the answer
    "connects of advisors in Blue Area"  Blue Area RESTRICTS the answer

Both arrive as `subjects=[team Blue Area]`. The pipeline got this wrong
in both directions — a team's own figure returned for a query about its
advisors, and 48 advisors returned for a query about the team. Here they
are different fields and cannot be confused.
"""

import pytest

from app.llm import operations
from app.llm.query_ir import (
    Filter, FilterGroup, MetricRef, QueryIR, Sort, Subject, TimeRange,
)
from app.llm.semantic_model import (
    Condition, ConditionGroup, ConversationReference, EntityRef,
    MetricRequest, Ordering, Relationship, SemanticModel, TimeRange as SemTimeRange,
    from_query_ir,
)


# =====================================================================
# The distinction the model exists for
# =====================================================================
class TestSubjectIsNotScope:

    def test_a_named_entity_is_the_subject_by_default(self):
        """Rule A. "connects of Blue Area" — Blue Area is the answer."""
        m = SemanticModel(
            operation="group_metric",
            metrics=[MetricRequest(name="total_connects")],
            subject=EntityRef(name="Blue Area", level="team"),
            subject_level="team",
        )
        assert m.subject.name == "Blue Area"
        assert m.scope == []
        assert m.requested_level is None
        assert m.relationship is None
        assert not m.is_hierarchy_query()

    def test_an_explicit_level_makes_the_entity_a_scope(self):
        """Rule B. "connects of advisors in Blue Area" — advisors are the
        answer, Blue Area restricts it."""
        m = SemanticModel(
            operation="filtered_list",
            metrics=[MetricRequest(name="total_connects")],
            scope=[EntityRef(name="Blue Area", level="team")],
            subject_level="advisor",
            requested_level="advisor",
            relationship=Relationship(kind="membership"),
        )
        assert m.subject is None
        assert [e.name for e in m.scope] == ["Blue Area"]
        assert m.requested_level == "advisor"
        assert m.is_hierarchy_query()

    def test_the_two_are_different_fields_not_one_field_read_two_ways(self):
        """The structural property. Nothing has to consult `target_level`
        or a level word to tell which reading applies."""
        own = SemanticModel(operation="group_metric",
                            subject=EntityRef(name="Blue Area", level="team"),
                            subject_level="team")
        members = SemanticModel(operation="filtered_list",
                                scope=[EntityRef(name="Blue Area", level="team")],
                                subject_level="advisor", requested_level="advisor")
        assert (own.subject is None) != (members.subject is None)
        assert bool(own.scope) != bool(members.scope)


class TestHierarchyIsNeverInvented:
    """Rule D. A subject that HAS members is not a traversal."""

    def test_having_members_underneath_is_not_a_hierarchy_query(self):
        m = SemanticModel(operation="group_metric",
                          metrics=[MetricRequest(name="total_connects")],
                          subject=EntityRef(name="Blue Area", level="team"),
                          subject_level="team")
        assert not m.is_hierarchy_query()

    def test_an_explicit_relationship_is_a_hierarchy_query(self):
        """"teams under Faisal" — Faisal is the subject, teams requested."""
        m = SemanticModel(operation="population",
                          subject=EntityRef(name="Faisal", level="unit_head"),
                          subject_level="team", requested_level="team",
                          relationship=Relationship(kind="membership", depth="subtree"))
        assert m.is_hierarchy_query()
        assert m.requested_level == "team"
        assert m.subject.name == "Faisal"

    def test_a_requested_level_matching_the_subject_is_not_traversal(self):
        """"teams in Graana" asks for teams; a query whose requested level
        IS the subject's own level is not a descent."""
        m = SemanticModel(operation="group_metric",
                          subject=EntityRef(name="Blue Area", level="team"),
                          subject_level="team", requested_level="team")
        assert not m.is_hierarchy_query()

    def test_direct_depth_is_distinct_from_subtree(self):
        """Rule C. "reports DIRECTLY to" is not "under"."""
        direct = Relationship(kind="reports_to", depth="direct")
        subtree = Relationship(kind="reports_to", depth="subtree")
        assert direct.depth != subtree.depth

    def test_the_reverse_direction_has_its_own_kind(self):
        """"the unit head OF Blue Area" reads UPWARD. Recorded, not
        resolved — which person that is belongs to grounding."""
        m = SemanticModel(operation="reverse_hierarchy",
                          subject=EntityRef(name="Blue Area", level="team"),
                          requested_level="unit_head",
                          relationship=Relationship(kind="manager_of"))
        assert m.relationship.kind == "manager_of"
        assert m.subject.name == "Blue Area"


# =====================================================================
# It represents meaning, not execution
# =====================================================================
class TestItCarriesNoExecutionDetail:

    @pytest.mark.parametrize("field", [
        "resolved_id", "resolved_wid",   # grounding's answers
        "flat", "nlu_mode", "repairs",   # rendering / observability
        "intent",                        # the legacy duplicate of `operation`
        "confidence_level",
    ])
    def test_execution_fields_are_absent(self, field):
        assert field not in SemanticModel.model_fields
        assert field not in EntityRef.model_fields

    def test_an_entity_is_the_users_words_not_a_record(self):
        """Grounding decides whether "Blue Area" exists and which record
        it is. Nothing here can express an answer to that."""
        e = EntityRef(name="Blue Area", level="team")
        assert e.name == "Blue Area"
        assert not hasattr(e, "resolved_wid")

    def test_a_level_the_user_stated_is_distinguishable_from_one_inferred(self):
        """A stated level is evidence; an inferred one is a guess. Later
        stages must be able to tell them apart before overruling either."""
        stated = EntityRef(name="Faisal", level="unit_head", level_was_stated=True)
        guessed = EntityRef(name="Faisal", level="unit_head")
        assert stated.level_was_stated and not guessed.level_was_stated


class TestMetricsAreEqual:
    """QueryIR splits `metric` from `metrics[]`, which invites treating
    the first as real and the rest as decoration."""

    def test_every_measure_is_kept_in_order(self):
        m = SemanticModel(operation="filtered_list", metrics=[
            MetricRequest(name="answered_calls"),
            MetricRequest(name="total_connects"),
            MetricRequest(name="meeting_rate")])
        assert m.metric_names() == ["answered_calls", "total_connects", "meeting_rate"]

    def test_there_is_no_primary_metric_field(self):
        assert "metric" not in SemanticModel.model_fields

    def test_ordering_is_a_separate_question_from_which_metrics(self):
        """Which measures were asked for, and which one ORDERS the answer,
        are different facts."""
        m = SemanticModel(operation="leaderboard",
                          metrics=[MetricRequest(name="total_connects"),
                                   MetricRequest(name="answered_calls")],
                          ordering=Ordering(metric="total_connects",
                                            direction="desc", stated=True))
        assert len(m.metric_names()) == 2
        assert m.ordering.metric == "total_connects"

    def test_a_measureless_question_is_representable(self):
        """A roster asks WHO and names no measure. Empty is meaningful."""
        m = SemanticModel(operation="roster",
                          scope=[EntityRef(name="Blue Area", level="team")],
                          requested_level="advisor")
        assert m.metric_names() == []


class TestStatedVersusDefaulted:
    """QueryIR's `time_range.period` defaults to MTD, so "the user said
    this month" and "the user said nothing" are the same value."""

    def test_an_unstated_period_is_distinguishable_from_a_stated_one(self):
        assert SemTimeRange().stated is False
        assert SemTimeRange(period="YTD", stated=True).stated is True

    def test_an_unstated_sort_direction_is_distinguishable(self):
        """Only a stated direction may override a measure's own polarity."""
        assert Ordering().stated is False
        assert Ordering(metric="overdue", direction="asc", stated=True).stated is True


class TestAmbiguityAndMissingInformation:

    def test_the_parser_can_say_a_name_was_ambiguous(self):
        m = SemanticModel(operation="group_metric",
                          subject=EntityRef(name="Yasir Ali"),
                          ambiguous=["Yasir Ali"])
        assert m.ambiguous == ["Yasir Ali"]

    def test_an_entity_with_no_level_is_a_real_state(self):
        """"connects of Faisal" names somebody without saying what they
        are. That is the question grounding exists to settle."""
        assert EntityRef(name="Faisal").level is None

    def test_missing_slots_are_recorded(self):
        m = SemanticModel(operation="leaderboard", missing=["metric"])
        assert m.missing == ["metric"]


class TestConversationReference:

    def test_a_first_turn_carries_nothing(self):
        assert SemanticModel(operation="group_metric").conversation.is_follow_up is False

    def test_a_follow_up_records_which_fields_it_inherited(self):
        """Which fields were carried is what makes a wrong follow-up
        diagnosable — otherwise a merged model is indistinguishable from
        one the user typed in full."""
        m = SemanticModel(
            operation="group_metric",
            subject=EntityRef(name="Blue Area", level="team"),
            time_range=SemTimeRange(period="YTD", stated=True),
            conversation=ConversationReference(
                is_follow_up=True, inherited=["subject", "metrics"],
                referring_phrase="what about"))
        assert m.conversation.is_follow_up
        assert "subject" in m.conversation.inherited


class TestComparisons:

    def test_comparison_subjects_are_not_scope(self):
        """Both sides are subjects; neither restricts the other."""
        m = SemanticModel(operation="comparison",
                          metrics=[MetricRequest(name="total_connects")],
                          comparison_subjects=[EntityRef(name="Blue Area", level="team"),
                                               EntityRef(name="DownTown", level="team")])
        assert len(m.comparison_subjects) == 2
        assert m.scope == []
        assert m.subject is None

    def test_a_period_comparison_is_representable(self):
        m = SemanticModel(operation="group_metric",
                          time_range=SemTimeRange(period="MTD", stated=True,
                                                  compare_to="YTD"))
        assert m.time_range.compare_to == "YTD"


class TestBooleanConditions:

    def test_a_disjunction_survives(self):
        m = SemanticModel(operation="population", condition_tree=ConditionGroup(
            op="or", children=[Condition(field="team", value="Blue Area"),
                               Condition(field="team", value="DownTown")]))
        assert m.condition_tree.op == "or"
        assert len(m.all_conditions()) == 2

    def test_an_exclusion_survives(self):
        m = SemanticModel(operation="population", condition_tree=ConditionGroup(
            op="not", children=[Condition(field="team", value="Blue Area")]))
        assert m.condition_tree.op == "not"

    def test_flat_and_tree_conditions_are_read_together(self):
        m = SemanticModel(
            operation="filtered_list",
            conditions=[Condition(field="total_connects", operator=">", value=1000)],
            condition_tree=ConditionGroup(op="or", children=[
                Condition(field="team", value="Blue Area")]))
        assert {c.field for c in m.all_conditions()} == {"total_connects", "team"}


class TestVocabularyIsSingleSourced:

    def test_the_operation_must_be_one_the_registry_declares(self):
        with pytest.raises(ValueError):
            SemanticModel(operation="not_a_real_operation")

    @pytest.mark.parametrize("name", sorted(operations.OPERATIONS))
    def test_every_registry_operation_is_expressible(self, name):
        assert SemanticModel(operation=name).operation == name

    def test_levels_come_from_the_hierarchy_registry(self):
        """A level added to hierarchy.py must be usable here with no edit."""
        from app.llm import hierarchy
        for level in hierarchy.HIERARCHY_LEVELS:
            assert EntityRef(name="x", level=level).level == level


# =====================================================================
# It can hold what the system already produces
# =====================================================================
class TestItHoldsExistingParses:
    """Phase 1's compatibility obligation: whatever the pipeline produces
    today must be expressible, or the model is not a contract."""

    def _ir(self, **kw):
        base = dict(intent="filtered_list", operation="filtered_list",
                    subject_level="advisor", sort=Sort(metric=None), limit=None)
        base.update(kw)
        return QueryIR(**base)

    def test_an_own_figure_query_becomes_a_subject(self):
        ir = self._ir(operation="group_metric", subject_level="team",
                      metric=MetricRef(key="total_connects"),
                      subjects=[Subject(type="team", value="Blue Area")])
        m = from_query_ir(ir)
        assert m.subject and m.subject.name == "Blue Area"
        assert m.scope == []
        assert not m.is_hierarchy_query()

    def test_a_member_query_becomes_a_scope(self):
        """The level word is the user's wording, so it is passed in — the
        IR does not carry it."""
        ir = self._ir(subject_level="advisor",
                      metric=MetricRef(key="total_connects"),
                      subjects=[Subject(type="team", value="Blue Area")])
        m = from_query_ir(ir, level_word="advisor")
        assert m.subject is None
        assert [e.name for e in m.scope] == ["Blue Area"]
        assert m.requested_level == "advisor"
        assert m.is_hierarchy_query()

    def test_a_hierarchy_read_becomes_a_relationship(self):
        ir = self._ir(operation="population", subject_level="advisor",
                      metric=None, target_level="advisor", subject_of="unit_head",
                      relation="direct",
                      subjects=[Subject(type="unit_head", value="Faisal")])
        m = from_query_ir(ir, level_word="advisor")
        assert m.relationship is not None
        assert m.relationship.depth == "direct"
        assert m.requested_level == "advisor"

    def test_a_comparison_keeps_both_sides(self):
        ir = self._ir(operation="comparison", intent="comparison",
                      subject_level="team", metric=MetricRef(key="total_connects"),
                      subjects=[Subject(type="team", value="Blue Area"),
                                Subject(type="team", value="DownTown")])
        m = from_query_ir(ir)
        assert [e.name for e in m.comparison_subjects] == ["Blue Area", "DownTown"]
        assert m.scope == [] and m.subject is None

    def test_every_measure_survives_the_conversion(self):
        ir = self._ir(metric=MetricRef(key="total_connects"),
                      metrics=[MetricRef(key="answered_calls")])
        assert from_query_ir(ir).metric_names() == ["total_connects", "answered_calls"]

    def test_a_boolean_tree_survives_the_conversion(self):
        ir = self._ir(operation="population", filter_tree=FilterGroup(
            op="or", children=[Filter(field="team", value="Blue Area"),
                               Filter(field="team", value="DownTown")]))
        m = from_query_ir(ir)
        assert m.condition_tree.op == "or"
        assert len(m.all_conditions()) == 2

    def test_period_and_limit_survive(self):
        ir = self._ir(operation="leaderboard", metric=MetricRef(key="mtd_cleared"),
                      time_range=TimeRange(period="YTD"), limit=5)
        m = from_query_ir(ir)
        assert m.time_range.period == "YTD"
        assert m.limit == 5

    def test_missing_and_ambiguity_survive(self):
        ir = self._ir(missing=["metric"], ambiguity_reasons=["which metric"])
        m = from_query_ir(ir)
        assert m.missing == ["metric"]
        assert m.ambiguous == ["which metric"]

    def test_the_two_readings_of_one_ir_shape_are_separated(self):
        """THE regression this file exists for, end to end: the same
        `subjects=[team Blue Area]` becomes a subject or a scope depending
        only on whether the user named a level."""
        shared = dict(metric=MetricRef(key="total_connects"),
                      subjects=[Subject(type="team", value="Blue Area")])
        own = from_query_ir(self._ir(operation="group_metric",
                                     subject_level="team", **shared))
        members = from_query_ir(self._ir(subject_level="advisor", **shared),
                                level_word="advisor")
        assert own.subject is not None and own.scope == []
        assert members.subject is None and members.scope != []
