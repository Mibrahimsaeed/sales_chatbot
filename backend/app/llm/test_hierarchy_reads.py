"""Hierarchy reads as a QueryIR shape, not a routing keyword.

THE DEFECT. "enumerate THIS level beneath THAT subject" had no IR
representation: `subject_level` conflates "who the query is about" with
"what to return", and nothing carried the difference between the
immediate reports and the whole subtree. So `roster`, `direct_reports`
and `scoped_reports` were marked plan-only, the LLM was never asked, and
routing fell to whether the sentence happened to contain the adverb
"directly" — drop it and the same question re-routed and answered
something else entirely.

THE SHAPE IS NOW THREE FIELDS: `target_level` (what to return),
`subject_of` (the level it sits beneath) and `relation` (direct vs
subtree). These tests pin the semantics of those fields and the
invariants that make them safe, structurally — never by asserting on a
particular sentence.
"""

import pytest

from app.database.models import Advisor
from app.llm import entity_extractor
from app.llm.ir_validator import validate_ir
from app.llm.query_ir import Filter, MetricRef, QueryIR, Sort, Subject
from app.llm.response_planner import plan_response


def _read(target, of, relation="subtree", subjects=None, **kw):
    base = dict(
        intent="filtered_list", operation="scoped_reports",
        subject_level="team", target_level=target, subject_of=of,
        relation=relation, sort=Sort(metric=None), limit=None,
        subjects=subjects if subjects is not None else [Subject(type="team", value="Alpha")],
    )
    base.update(kw)
    return QueryIR(**base)


@pytest.fixture()
def org(db_session):
    """A three-layer org: one unit head, two BCMs, four advisors.

    The unit head is ALSO the direct BCM of one advisor, which is the
    real shape that makes "direct" and "subtree" differ — and the case a
    self-exclusion rule has to get right.
    """
    rows = [
        # wid, name,     team,    rm(unit_head), portfolio_lead, management_lead(bcm)
        (1, "UH One",     "Alpha", "UH One", "UH One", "UH One"),   # the head himself
        (2, "BCM A",      "Alpha", "UH One", "UH One", "UH One"),   # reports DIRECTLY to UH
        (3, "BCM B",      "Alpha", "UH One", "UH One", "UH One"),   # reports DIRECTLY to UH
        (4, "Adv One",    "Alpha", "UH One", "UH One", "BCM A"),    # two levels down
        (5, "Adv Two",    "Alpha", "UH One", "UH One", "BCM A"),
        (6, "Adv Three",  "Alpha", "UH One", "UH One", "BCM B"),
        (7, "Outsider",   "Beta",  "UH Two", "UH Two", "BCM C"),    # different subtree
    ]
    for wid, name, team, rm, pl, ml in rows:
        db_session.add(Advisor(wid=wid, name=name, team=team, company="Acme",
                               rm=rm, portfolio_lead=pl, management_lead=ml,
                               in_master_sheet=True))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


# =====================================================================
# The IR can now carry the shape
# =====================================================================
class TestTheShapeIsRepresentable:
    def test_the_three_fields_default_to_a_non_hierarchy_query(self):
        """Additive: an IR that does not set them behaves as before."""
        ir = QueryIR(intent="leaderboard", subject_level="advisor")
        assert ir.target_level is None
        assert ir.subject_of is None
        assert ir.relation == "subtree"
        assert ir.is_hierarchy_read() is False
        assert ir.grouping_level() == "advisor"

    def test_a_target_level_alone_is_not_a_hierarchy_read(self):
        """Both halves are required — a target with nothing to scope it
        beneath is an ordinary population at that level."""
        ir = QueryIR(intent="filtered_list", subject_level="advisor",
                     target_level="advisor")
        assert ir.is_hierarchy_read() is False

    def test_the_target_level_decides_what_rows_come_back(self):
        assert _read("bcm", "unit_head").grouping_level() == "bcm"
        assert _read("advisor", "unit_head").grouping_level() == "advisor"

    def test_the_capability_exists_but_routing_is_still_gated(self):
        """THE REMAINING STEP, recorded rather than assumed.

        The IR can now carry a hierarchy read and the compiler executes
        it correctly (every test below runs through the real compiler).
        What is NOT yet flipped is the registry flag that routes these
        operations to the LLM.

        Flipping it is one line, and it fails 43 existing tests in
        test_direct_reports.py / test_scoped_reports.py — not because the
        answers are wrong, but because those tests assert
        `resolution.plan.action`, i.e. they pin the ROUTE. They cannot
        pass while the query is IR-served, however correct the IR is.

        Completing the migration therefore needs a decision this test
        deliberately does not make: when the question names no target
        level, does "who reports to a Unit Head" mean the level
        immediately below (what the plan does) or the leaf (what the
        model tends to choose)? Those are different answers to the same
        sentence, and rewriting the tests before settling it would just
        enshrine whichever the model happened to pick.
        """
        from app.llm.operations import OPERATIONS

        for name in ("roster", "direct_reports", "scoped_reports"):
            assert OPERATIONS[name].expressible_in_ir is False, (
                f"{name} routing was flipped — update the 43 route-asserting "
                "tests in test_direct_reports/test_scoped_reports first")

    def test_the_llm_can_actually_emit_the_fields(self):
        """A field absent from the grammar is unemittable however well
        the prompt describes it."""
        from app.llm.llm_client import QUERY_IR_JSON_SCHEMA as S

        for field in ("target_level", "subject_of", "relation"):
            assert field in S["properties"], field
            assert field in S["required"], field
        # strict mode stays satisfiable
        assert set(S["properties"]) == set(S["required"])

    def test_the_prompt_documents_the_fields(self):
        from app.llm.prompt_builder import _ir_schema

        text = _ir_schema()
        for field in ("target_level", "subject_of", "relation"):
            assert field in text, field


# =====================================================================
# direct vs subtree — the semantics "directly" used to carry as a keyword
# =====================================================================
class TestDirectVersusSubtree:
    def test_subtree_returns_everyone_beneath_at_any_depth(self, org):
        from app.llm.query_compiler import compile_and_run

        ir = _read("advisor", "unit_head", "subtree",
                   subjects=[Subject(type="unit_head", value="UH One")])
        rows = compile_and_run(org, ir)
        names = {r["name"] for r in rows}
        # everyone with rm=UH One, minus the head himself
        assert names == {"BCM A", "BCM B", "Adv One", "Adv Two", "Adv Three"}

    def test_direct_returns_only_the_immediate_reports(self, org):
        from app.llm.query_compiler import compile_and_run

        ir = _read("advisor", "unit_head", "direct",
                   subjects=[Subject(type="unit_head", value="UH One")])
        rows = compile_and_run(org, ir)
        # an advisor's immediate manager is their BCM, so this is the
        # people whose management_lead IS the head — self excluded
        assert {r["name"] for r in rows} == {"BCM A", "BCM B"}

    def test_direct_is_a_strict_subset_of_subtree(self, org):
        from app.llm.query_compiler import compile_and_run

        subj = [Subject(type="unit_head", value="UH One")]
        sub = {r["name"] for r in compile_and_run(org, _read("advisor", "unit_head", "subtree", subjects=subj))}
        dir_ = {r["name"] for r in compile_and_run(org, _read("advisor", "unit_head", "direct", subjects=subj))}
        assert dir_ < sub

    def test_the_subject_is_never_one_of_their_own_reports(self, org):
        from app.llm.query_compiler import compile_and_run

        subj = [Subject(type="unit_head", value="UH One")]
        for relation in ("direct", "subtree"):
            rows = compile_and_run(org, _read("advisor", "unit_head", relation, subjects=subj))
            assert "UH One" not in {r["name"] for r in rows}, relation

    def test_a_different_subtree_is_excluded(self, org):
        from app.llm.query_compiler import compile_and_run

        rows = compile_and_run(org, _read("advisor", "unit_head", "subtree",
                                          subjects=[Subject(type="unit_head", value="UH One")]))
        assert "Outsider" not in {r["name"] for r in rows}


# =====================================================================
# Different target levels, and the scope resolved out of a group
# =====================================================================
class TestTargetLevelsAndScopes:
    def test_a_different_target_level_returns_that_level(self, org):
        from app.llm.query_compiler import compile_and_run

        rows = compile_and_run(org, _read("bcm", "unit_head", "subtree",
                                          subjects=[Subject(type="unit_head", value="UH One")]))
        assert {r["name"] for r in rows} == {"BCM A", "BCM B"}

    def test_the_subject_may_be_the_SCOPE_rather_than_the_manager(self, org):
        """"the Unit Head in <team>" — the subject is the team and the
        role holder is read out of it, reusing get_manager_of_group."""
        from app.llm.query_compiler import compile_and_run

        ir = _read("advisor", "unit_head", "subtree",
                   subjects=[Subject(type="team", value="Alpha")])
        rows = compile_and_run(org, ir)
        assert {r["name"] for r in rows} == {"BCM A", "BCM B", "Adv One", "Adv Two", "Adv Three"}

    def test_the_count_matches_the_rows(self, org):
        from app.llm.query_compiler import compile_and_run, count_ir

        ir = _read("advisor", "unit_head", "subtree",
                   subjects=[Subject(type="unit_head", value="UH One")])
        assert count_ir(org, ir) == len(compile_and_run(org, ir))


# =====================================================================
# Validation
# =====================================================================
class TestValidation:
    def test_a_hierarchy_read_needs_no_metric(self, org):
        """It enumerates people; there is nothing to rank by. Requiring
        one would refuse the shape the IR was widened to express."""
        ir = _read("advisor", "unit_head",
                   subjects=[Subject(type="unit_head", value="UH One")])
        assert "metric" not in validate_ir(ir, org).missing

    def test_an_inverted_pair_is_refused(self, org):
        """A level cannot contain the level above it. Without this the
        scope filter silently returns nothing, which reads as "no data"
        rather than "that question is backwards".

        The SLOT was renamed in Phase 5.1; the refusal is unchanged. It
        used to be built as `f"subject:{subject_of}:{target_level} is not
        beneath it"`, and `_ask_for` splits a "subject:" entry on ":" into
        a level and a value — so the value became the literal string
        "advisor is not beneath it" and the user was asked which unit head
        they meant by it. Asserting on the prose was what let a structured
        slot carry a sentence.
        """
        ir = _read("unit_head", "advisor",
                   subjects=[Subject(type="advisor", value="Adv One")])
        missing = validate_ir(ir, org).missing
        assert any(m.startswith("inverted_hierarchy:") for m in missing), missing

    def test_a_hierarchy_read_requires_a_subject(self, org):
        ir = _read("advisor", "unit_head", subjects=[])
        assert "subjects" in validate_ir(ir, org).missing

    def test_a_scope_subject_is_regrounded_not_refused(self, org):
        """A parser may attach the SCOPE's value to the ROLE's level.
        Grounding it at the declared level fails, and the user would be
        asked which unit head they meant by a team name. The name is
        re-typed to the level that actually claims it."""
        ir = _read("advisor", "unit_head",
                   subjects=[Subject(type="unit_head", value="Alpha")])
        result = validate_ir(ir, org)
        assert not [m for m in result.missing if m.startswith("subject:")]
        assert result.ir.subjects[0].type == "team"


# =====================================================================
# Answer shape
# =====================================================================
class TestResponseShape:
    def test_a_hierarchy_read_answers_as_a_population(self):
        """Not a leaderboard: there is no measure, so a ranking would
        print "no data" beside every name."""
        rows = [{"wid": 1, "name": "A"}, {"wid": 2, "name": "B"}]
        plan = plan_response(_read("advisor", "unit_head"), rows)
        assert plan.mode == "population"
        assert plan.shape == "filtered_table"

    def test_ordinary_queries_keep_their_shapes(self):
        rows = [{"wid": 1, "name": "A", "value": 5}, {"wid": 2, "name": "B", "value": 4}]
        lb = QueryIR(intent="leaderboard", operation="leaderboard", subject_level="advisor",
                     metric=MetricRef(key="mtd_cleared"), sort=Sort(metric="mtd_cleared"))
        assert plan_response(lb, rows).mode == "leaderboard"

        fl = QueryIR(intent="filtered_list", operation="filtered_list", subject_level="advisor",
                     metrics=[MetricRef(key="total_connects")],
                     filters=[Filter(field="total_connects", operator=">", value=1)],
                     sort=Sort(metric=None))
        assert plan_response(fl, rows).mode == "filtered_list"


# =====================================================================
# Regression: unrelated shapes are untouched
# =====================================================================
class TestUnrelatedQueriesUnchanged:
    def test_a_leaderboard_still_compiles_with_its_metric(self, org):
        from app.llm.query_compiler import _effective_metric

        ir = QueryIR(intent="leaderboard", operation="leaderboard", subject_level="advisor",
                     metric=MetricRef(key="mtd_cleared"), sort=Sort(metric="mtd_cleared"))
        assert ir.is_hierarchy_read() is False
        assert _effective_metric(ir) == "mtd_cleared"

    def test_a_population_is_still_metric_free(self, org):
        ir = QueryIR(intent="filtered_list", operation="population", subject_level="advisor",
                     sort=Sort(metric=None))
        assert ir.primary_metric() is None
        assert "metric" not in validate_ir(ir, org).missing

    def test_a_filter_only_query_still_derives_its_metric(self, org):
        ir = QueryIR(intent="filtered_list", operation="filtered_list", subject_level="advisor",
                     filters=[Filter(field="total_connects", operator=">", value=100)],
                     sort=Sort(metric=None))
        assert ir.is_hierarchy_read() is False
        assert ir.primary_metric() == "total_connects"

    def test_a_comparison_still_needs_two_subjects(self, org):
        ir = QueryIR(intent="comparison", operation="comparison", subject_level="team",
                     subjects=[Subject(type="team", value="Alpha")],
                     metric=MetricRef(key="mtd_cleared"), sort=Sort(metric="mtd_cleared"))
        assert "subjects" in validate_ir(ir, org).missing
