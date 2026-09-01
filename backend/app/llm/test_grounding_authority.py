"""Deterministic grounding outranks the model's guess about a level.

THE DEFECT. Entity extraction runs before the model and matches names
against the live gazetteers. Its findings reached the model only as prompt
text — "Entities already found by rule-based grounding (use these, don't
re-derive)" — and text is advice. Measured against gpt-4o-mini:

    "revenue of AMD year to date"
      extraction : AMD -> team, confidence 1.0   (exact gazetteer hit)
      prompt     : says exactly that
      model      : subjects=[{"type": "company", "value": "AMD"}]
      validator  : no company called AMD -> subject DROPPED
      user       : "Quick question - which company you meant by 'AMD'?"

The name was never in doubt. A deterministic, exact match against real
data was discarded in favour of a level the model guessed, and the user
was asked to disambiguate something the system already knew.

WHAT MAKES THIS SAFE. The correction is comparative, never blanket:

  - only a NEAR-EXACT extraction hit (>= 0.95) may outrank the parse;
  - only when the name grounds at exactly ONE level, so a genuinely
    ambiguous name is still left to the clarification that exists for it;
  - only when the model's declared level does not hold up at least as
    well, so a correct parse is never touched.

Below that bar, a second recovery still applies: when the declared level
does not claim the name AT ALL, the other gazetteers are asked what it is.
That was previously gated on `is_hierarchy_read()`, which described where
the defect had been observed rather than where it can occur.
"""

import pytest

from app.database.models import Advisor, Performance, PerformancePeriod
from app.llm import entity_extractor
from app.llm.entity_extractor import extract_entities
from app.llm.ir_validator import validate_ir
from app.llm.preprocessing import normalize
from app.llm.query_compiler import compile_and_run
from app.llm.query_ir import MetricRef, QueryIR, Sort, Subject, TimeRange


@pytest.fixture()
def org(db_session):
    """A team whose name looks nothing like a company, and companies that
    exist — so "which is AMD?" has a real, checkable answer."""
    people = [
        (1, "Ali Raza", "AMD", "Graana"),
        (2, "Sana Tariq", "AMD", "Graana"),
        (3, "Hina Malik", "Blue Area", "IMARAT"),
    ]
    for wid, name, team, company in people:
        db_session.add(Advisor(wid=wid, name=name, team=team, company=company,
                               in_master_sheet=True))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.YTD,
                                   target=1000, cleared=100 * wid, pct=10.0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=10 * wid, pct=10.0))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _ir(**overrides):
    base = dict(intent="filtered_list", operation="group_metric",
                subject_level="company", sort=Sort(metric=None), limit=None,
                metric=MetricRef(key="ytd_cleared", confidence=0.95),
                time_range=TimeRange(period="YTD"))
    base.update(overrides)
    return QueryIR(**base)


def _mistyped_as_company():
    """The IR gpt-4o-mini actually produced for "revenue of AMD year to
    date", reproduced field for field so the regression is the real one
    rather than an approximation of it."""
    return _ir(subjects=[Subject(type="company", value="AMD", match_confidence=1.0)])


# =====================================================================
# The required regression
# =====================================================================
class TestAMDResolvesToTheTeamItIs:

    def test_the_subject_is_re_typed_to_the_level_extraction_proved(self, org):
        entities = extract_entities(normalize("revenue of AMD year to date"), org)
        result = validate_ir(_mistyped_as_company(), org, entities=entities)

        assert [(s.type, s.value) for s in result.ir.subjects] == [("team", "AMD")]

    def test_it_is_not_refused(self, org):
        """The user-visible symptom: a clarifying question about a company
        that does not exist, for a name the system had already matched."""
        entities = extract_entities(normalize("revenue of AMD year to date"), org)
        result = validate_ir(_mistyped_as_company(), org, entities=entities)

        assert result.is_valid, result.missing
        assert not any("AMD" in m for m in result.missing)

    def test_the_reporting_level_agrees_with_the_corrected_subject(self, org):
        """`subject_level` is copied from `subjects[0].type`, so correcting
        the type inside the grounding loop — where this fix started — left
        subject_level="company" beside a `team` subject. The compiler then
        filters by team and GROUPS BY company: AMD's revenue, reported
        under a company's name. A wrong answer in place of a wrong
        question is not an improvement, which is why the correction runs
        as a pre-pass."""
        entities = extract_entities(normalize("revenue of AMD year to date"), org)
        result = validate_ir(_mistyped_as_company(), org, entities=entities)

        assert result.ir.subject_level == "team"

    def test_it_answers_with_the_teams_own_figure(self, org):
        """End to end: the corrected IR compiles and returns AMD's YTD
        revenue — 100 + 200 for its two advisors — not a company's."""
        entities = extract_entities(normalize("revenue of AMD year to date"), org)
        result = validate_ir(_mistyped_as_company(), org, entities=entities)
        rows = compile_and_run(org, result.ir)

        assert [(r["name"], r["value"]) for r in rows] == [("AMD", 300)]


# =====================================================================
# The recovery is no longer gated on hierarchy reads
# =====================================================================
class TestRegroundingAppliesToOrdinaryQueries:

    def test_a_mistyped_subject_recovers_without_the_entity_dict(self, org):
        """`_reground_scope_subject` asks the other gazetteers what a name
        IS when its declared level does not claim it. It fired only for
        hierarchy reads, so an ordinary query's mistyped subject was
        dropped instead. The guess is a property of the PARSER, not of the
        query's shape."""
        result = validate_ir(_mistyped_as_company(), org)

        assert [(s.type, s.value) for s in result.ir.subjects] == [("team", "AMD")]

    def test_a_hierarchy_read_still_recovers_its_scope(self, org):
        """The case the gate was written for must keep working: the value
        is the SCOPE, typed as the role that sits inside it."""
        ir = _ir(operation="population", subject_level="advisor",
                 metric=None, target_level="advisor", subject_of="unit_head",
                 subjects=[Subject(type="unit_head", value="AMD", match_confidence=1.0)])
        result = validate_ir(ir, org)

        assert [s.type for s in result.ir.subjects] == ["team"]

    def test_a_name_nothing_claims_is_still_refused(self, org):
        """Recovery must not become invention. A value no gazetteer knows
        keeps its declared type and is refused, exactly as before."""
        ir = _ir(subjects=[Subject(type="company", value="Atlantis",
                                   match_confidence=1.0)])
        result = validate_ir(ir, org)

        assert not result.is_valid
        assert any("Atlantis" in m for m in result.missing)


# =====================================================================
# The case regrounding alone cannot reach
# =====================================================================
class TestAWeakMatchAtTheDeclaredLevelDoesNotWin:
    """`_reground_scope_subject` fires only when the declared level claims
    the name NOT AT ALL. A level that claims it BADLY therefore slips
    through: the fuzzy matcher scores "AMD" against a company called "AM
    Developments" at 0.80, comfortably over the 0.55 floor, so grounding
    "succeeds" and the query is silently scoped to a company the user
    never named. Nothing downstream can tell — the subject has a
    resolved_id and a plausible confidence.

    This is the case extraction authority exists for, and the only one it
    is needed for: an exact hit at 1.00 outranks a 0.80 guess.
    """

    @pytest.fixture()
    def near_miss(self, db_session):
        db_session.add(Advisor(wid=1, name="Ali Raza", team="AMD",
                               company="AM Developments", in_master_sheet=True))
        db_session.add(Performance(wid=1, period=PerformancePeriod.YTD,
                                   target=1000, cleared=500, pct=50.0))
        db_session.commit()
        entity_extractor._cache["loaded_at"] = 0
        yield db_session
        entity_extractor._cache["loaded_at"] = 0

    def test_the_declared_level_really_does_ground_weakly(self, near_miss):
        """The precondition, asserted rather than assumed — if the matcher
        stops scoring this above the floor, the test below would pass for
        the wrong reason."""
        from app.llm.ir_validator import _grounds_here

        score = _grounds_here(
            Subject(type="company", value="AMD", match_confidence=1.0), near_miss)
        assert 0.55 <= score < 0.95, score

    def test_regrounding_alone_scopes_it_to_the_wrong_company(self, near_miss):
        """Without the extractor's findings the weak match stands. Recorded
        so the value of passing them is visible rather than asserted."""
        result = validate_ir(_mistyped_as_company(), near_miss)

        assert [(s.type, s.value) for s in result.ir.subjects] == [
            ("company", "AM Developments")]

    def test_the_exact_extraction_hit_wins(self, near_miss):
        entities = extract_entities(normalize("revenue of AMD year to date"), near_miss)
        result = validate_ir(_mistyped_as_company(), near_miss, entities=entities)

        assert [(s.type, s.value) for s in result.ir.subjects] == [("team", "AMD")]
        assert result.ir.subject_level == "team"


# =====================================================================
# It abstains wherever the model's reading might be right
# =====================================================================
class TestItDoesNotOverrideALegitimateReading:

    def test_a_correctly_typed_subject_is_untouched(self, org):
        entities = extract_entities(normalize("revenue of Graana"), org)
        ir = _ir(subjects=[Subject(type="company", value="Graana",
                                   match_confidence=1.0)])
        result = validate_ir(ir, org, entities=entities)

        assert [(s.type, s.value) for s in result.ir.subjects] == [("company", "Graana")]

    def test_a_name_that_grounds_at_two_levels_is_left_to_the_model(self, org):
        """The clarification the pipeline already owns. Forcing a level for
        an ambiguous name would answer a question the system is entitled
        to ask — production has names that are simultaneously a unit_head,
        a zonal_head, a bcm and an advisor."""
        entities = {
            "team_matches": [{"value": "Ali Raza", "score": 1.0}],
            "advisor_matches": [{"value": "Ali Raza", "score": 1.0, "wid": 1}],
        }
        ir = _ir(subjects=[Subject(type="advisor", value="Ali Raza",
                                   match_confidence=1.0)])
        result = validate_ir(ir, org, entities=entities)

        assert result.ir.subjects[0].type == "advisor"

    def test_a_merely_FUZZY_extraction_hit_does_not_outrank_the_parse(self, org):
        """Two guesses, not evidence against a guess. Only a near-exact
        match may overrule the parse that saw the whole sentence."""
        entities = {"team_matches": [{"value": "AMD", "score": 0.7}]}
        ir = _ir(subjects=[Subject(type="company", value="Graana",
                                   match_confidence=1.0)])
        result = validate_ir(ir, org, entities=entities)

        assert result.ir.subjects[0].type == "company"

    def test_no_entity_dict_is_not_an_error(self, org):
        """`entities` is optional: ir_patcher and the pending-slot fill
        build their own IRs and have none to pass."""
        ir = _ir(subjects=[Subject(type="company", value="Graana",
                                   match_confidence=1.0)])
        assert validate_ir(ir, org).is_valid


# =====================================================================
# Several subjects are each judged on their own
# =====================================================================
class TestEverySubjectIsJudgedIndependently:

    def test_a_comparison_corrects_only_the_side_that_is_wrong(self, org):
        entities = extract_entities(normalize("compare AMD and Graana"), org)
        ir = _ir(operation="comparison", intent="comparison", subject_level="company",
                 metric=MetricRef(key="ytd_cleared", confidence=0.95),
                 subjects=[Subject(type="company", value="AMD", match_confidence=1.0),
                           Subject(type="company", value="Graana", match_confidence=1.0)])
        result = validate_ir(ir, org, entities=entities)

        assert [(s.type, s.value) for s in result.ir.subjects] == [
            ("team", "AMD"), ("company", "Graana")]
