""""subject_level" is the level of the thing ASKED ABOUT, not of the rows
the measure happens to live in.

THE DEFECT. `company`, `office` and `region` are ATTRIBUTE levels — they
do not sit in the containment chain — and the business model protected
that fact with a rule that also forbade them as `subject_level` except
for an explicit ranking. A question about a named company's own figure is
not a ranking, so the model had no permitted level for it and fell to the
chain leaf: it emitted `subjects=[{"type": "company", ...}]` and
`subject_level="advisor"` in the same object, then answered with one
arbitrary advisor's number where the company's was asked for.

Nothing downstream was wrong. Grounding typed the entity correctly and
passed it to the model, the validator left the level alone, and the
compiler answers company level correctly — the restriction was the whole
of it.

The protection it was carrying is real and is kept: an attribute is still
never a step in the chain, so it is never a traversal target and never
"the level below" anything. What is removed is the collateral damage —
naming an entity is not traversing to it.

Synthetic entities throughout: the point is the relationship between
`subjects[].type`, `subject_level` and the measure's storage level, not
any particular company.
"""

import pytest

from app.database.models import Advisor, Calls
from app.llm import entity_extractor, hierarchy
from app.llm.ir_validator import validate_ir
from app.llm.metric_ontology import METRICS
from app.llm.prompt_builder import BUSINESS_MODEL, _ir_schema
from app.llm.query_compiler import compile_and_run, is_answerable
from app.llm.query_ir import MetricRef, QueryIR, Sort, Subject

# Two companies, two teams, four advisors. Connect counts are chosen so
# every level's total is distinct, which is what lets a wrong level be
# detected from the number alone rather than from the query shape.
_ORG = [
    # wid, name,        team,         company
    (1, "Adviser One",   "Team Alpha", "Acme Holdings"),
    (2, "Adviser Two",   "Team Alpha", "Acme Holdings"),
    (3, "Adviser Three", "Team Beta",  "Acme Holdings"),
    (4, "Adviser Four",  "Team Gamma", "Borealis Group"),
]
_CONNECTS = {1: 10, 2: 20, 3: 30, 4: 100}
# Acme = 10+20+30 = 60;  Team Alpha = 30;  Borealis = 100


@pytest.fixture()
def org(db_session):
    for wid, name, team, company in _ORG:
        db_session.add(Advisor(wid=wid, name=name, team=team, company=company,
                               rm=name, portfolio_lead=name, management_lead=name,
                               in_master_sheet=True))
        db_session.add(Calls(wid=wid, connects_mtd=_CONNECTS[wid]))
    db_session.commit()
    # The gazetteers `validate_ir` grounds subjects against are cached at
    # module level with a TTL, so a cache warmed by another test's fixture
    # would leave these companies ungroundable — and this fixture's would
    # leak into the next test. Same reset the other fixture-backed suites
    # do, for the same reason.
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _ir(level, subject_type, subject_value, metric="total_connects"):
    return QueryIR(
        intent="filtered_list", operation="group_metric", subject_level=level,
        subjects=[Subject(type=subject_type, value=subject_value,
                          match_confidence=1.0)],
        metric=MetricRef(key=metric, confidence=0.95), sort=Sort(metric=None),
    )


def _value(db, ir):
    rows = compile_and_run(db, validate_ir(ir, db).ir)
    return rows[0]["value"] if rows else None


# =====================================================================
# The prompt states the rule, once, and keeps the protection
# =====================================================================
class TestThePromptDefinesTheField:

    def test_subject_level_is_actually_defined(self):
        """It was named in the field list and given a bare enum of valid
        values, and never once defined — so the only guidance the model
        had was the restriction that caused the bug."""
        text = _ir_schema()
        assert "SUBJECT LEVEL" in text
        assert "the level THE ANSWER IS ABOUT" in text

    def test_the_prompt_says_the_two_fields_must_agree(self):
        text = _ir_schema()
        assert "the two must" in text and "agree" in text

    def test_the_prompt_separates_the_measures_own_level(self):
        """The distinction the failure turned on: where a measure is
        STORED says nothing about what was ASKED FOR."""
        text = _ir_schema()
        assert "SEPARATE THING" in text

    def test_attributes_are_no_longer_restricted_to_rankings(self):
        """The exact instruction that caused it."""
        assert "for an explicit ranking over it" not in BUSINESS_MODEL

    def test_the_traversal_protection_is_kept(self):
        """Removing the restriction must not remove what it was
        protecting: an attribute is still not a step in the chain."""
        assert "never treat an attribute as a STEP IN THE CHAIN" in BUSINESS_MODEL
        assert "not a hierarchy traversal" in BUSINESS_MODEL

    def test_the_rule_is_stated_in_one_place(self):
        """A rule repeated in two sections is a rule that will one day
        disagree with itself."""
        assert _ir_schema().count("SUBJECT LEVEL") == 1


# =====================================================================
# The semantics, at every level, against distinct numbers
# =====================================================================
class TestTheAnswerIsAboutTheSubjectsOwnLevel:

    def test_a_company_subject_answers_at_company_level(self, org):
        assert _value(org, _ir("company", "company", "Acme Holdings")) == 60

    def test_a_team_subject_answers_at_team_level(self, org):
        assert _value(org, _ir("team", "team", "Team Alpha")) == 30

    def test_the_two_levels_give_different_answers(self, org):
        """Which is why choosing the wrong one is not a cosmetic error."""
        assert _value(org, _ir("company", "company", "Acme Holdings")) != \
               _value(org, _ir("team", "team", "Team Alpha"))

    def test_each_company_gets_its_own_figure(self, org):
        assert _value(org, _ir("company", "company", "Borealis Group")) == 100

    @pytest.mark.parametrize("level", ["company", "team"])
    def test_the_level_survives_validation_unchanged(self, org, level):
        value = "Acme Holdings" if level == "company" else "Team Alpha"
        result = validate_ir(_ir(level, level, value), org)
        assert result.missing == []
        assert result.ir.subject_level == level
        assert result.ir.grouping_level() == level


# =====================================================================
# The measure's storage level is a different thing
# =====================================================================
class TestTheMeasuresOwnLevelIsSeparate:

    def test_a_leaf_stored_measure_still_answers_at_company_level(self, org):
        """`total_connects` is stored per advisor and declares
        primary_level='advisor'. That is HOW it is computed, not WHAT was
        asked for — the company's total is still the company's total."""
        assert METRICS["total_connects"].primary_level == "advisor"
        assert _value(org, _ir("company", "company", "Acme Holdings")) == 60

    def test_the_compiler_answers_every_attribute_level(self):
        """The capability was never the blocker."""
        for level in ("company", "office", "region"):
            assert is_answerable("total_connects", level)

    def test_a_company_subject_at_advisor_level_is_normalized(self, org):
        """The contradiction the parser used to emit: a company subject
        reported at advisor level. It answered about one arbitrary member
        instead of the company. Validation now repairs the pairing, so
        the question that comes out is the one that was asked.

        The unrepaired reading is still a DIFFERENT question — that is
        why the repair matters — so it is compiled here without the
        validator to show the two do not coincide."""
        repaired = validate_ir(_ir("advisor", "company", "Acme Holdings"), org)
        assert repaired.ir.subject_level == "company"
        assert _value(org, _ir("advisor", "company", "Acme Holdings")) == 60

        unrepaired = _ir("advisor", "company", "Acme Holdings")
        assert compile_and_run(org, unrepaired)[0]["value"] != 60


# =====================================================================
# What must not have changed
# =====================================================================
class TestNothingElseMoved:

    def test_an_attribute_is_still_not_in_the_chain(self):
        """The traversal protection, asserted against the registry rather
        than against prompt wording."""
        for level in ("company", "office", "region"):
            assert level in hierarchy.ATTRIBUTE_LEVELS
            assert level not in hierarchy.CHAIN

    def test_a_ranking_over_an_attribute_still_works(self, org):
        """"top <attribute> by <measure>" names no subject and ranks OVER
        the level — the case the old rule permitted, which must keep
        working now that it is no longer the only one."""
        ir = QueryIR(intent="leaderboard", operation="leaderboard",
                     subject_level="company", subjects=[],
                     metric=MetricRef(key="total_connects", confidence=0.95),
                     sort=Sort(metric="total_connects", direction="desc"))
        rows = compile_and_run(org, validate_ir(ir, org).ir)
        assert [r["name"] for r in rows][:2] == ["Borealis Group", "Acme Holdings"]

    def test_a_ranking_over_the_leaf_still_works(self, org):
        ir = QueryIR(intent="leaderboard", operation="leaderboard",
                     subject_level="advisor", subjects=[],
                     metric=MetricRef(key="total_connects", confidence=0.95),
                     sort=Sort(metric="total_connects", direction="desc"), limit=2)
        rows = compile_and_run(org, validate_ir(ir, org).ir)
        assert [r["name"] for r in rows] == ["Adviser Four", "Adviser Three"]
