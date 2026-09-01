"""A single-subject query is reported at that subject's own level.

THE DEFECT. `subjects[].type` says WHAT the question is about;
`subject_level` says WHERE the answer is reported. Nothing made them
agree, so the parser emitted a group as the subject and the leaf as the
level — `subjects=[<a company>]` with `subject_level="advisor"` — and the
compiler faithfully answered about one arbitrary member of that company
instead of the company. Everything else was right: grounding typed the
entity, the measure was correct, the operation was correct, and every
(metric, level) pair the compiler was asked for is answerable.

Stating the rule in the prompt was not enough. It fixed the phrasings it
was tested on and left others failing, and the violation rate tracked
surface wording rather than meaning — 8 of 10 single-subject aggregates
still contradicted themselves on the wordings that happened to fail.

The normalization is therefore STRUCTURAL: it reads the shape of the IR
and nothing else — no metric, no wording, no entity — which is why it
holds for phrasings no one has tried.

Its three exclusions are the shapes where the answer is deliberately not
at the subject's own level, and each is a different question:

  group_by      breaks one level's figures out at another
  target_level  enumerates a level BENEATH the subject (a hierarchy read)
  2+ subjects   is a comparison, whose sides carry their own levels

Synthetic entities throughout.
"""

import pytest

from app.database.models import Advisor, Calls
from app.llm import entity_extractor
from app.llm.ir_validator import validate_ir
from app.llm.query_compiler import compile_and_run
from app.llm.query_ir import Filter, MetricRef, QueryIR, Sort, Subject

_ORG = [
    (1, "Adviser One",   "Team Alpha", "Acme Holdings"),
    (2, "Adviser Two",   "Team Alpha", "Acme Holdings"),
    (3, "Adviser Three", "Team Beta",  "Acme Holdings"),
    (4, "Adviser Four",  "Team Gamma", "Borealis Group"),
]


@pytest.fixture()
def org(db_session):
    for wid, name, team, company in _ORG:
        db_session.add(Advisor(wid=wid, name=name, team=team, company=company,
                               rm="UH One", portfolio_lead="ZH One",
                               management_lead="BCM One", in_master_sheet=True))
        db_session.add(Calls(wid=wid, connects_mtd=10 * wid))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    yield db_session
    entity_extractor._cache["loaded_at"] = 0


def _ir(**over):
    base = dict(intent="filtered_list", operation="group_metric",
                subject_level="advisor", sort=Sort(metric=None),
                metric=MetricRef(key="total_connects", confidence=0.95))
    base.update(over)
    return QueryIR(**base)


def _subject(type_, value):
    return Subject(type=type_, value=value, match_confidence=1.0)


def _level(db, ir):
    return validate_ir(ir, db).ir.subject_level


# =====================================================================
# 1-3. one subject, no grouping, no hierarchy target
# =====================================================================
@pytest.mark.parametrize("level,value", [
    ("company", "Acme Holdings"),
    ("team", "Team Alpha"),
    ("advisor", "Adviser One"),
    ("unit_head", "UH One"),
    ("zonal_head", "ZH One"),
    ("bcm", "BCM One"),
])
def test_a_single_subject_is_reported_at_its_own_level(org, level, value):
    """Every level, including the leaf — where the rule is a no-op and
    must stay one."""
    assert _level(org, _ir(subjects=[_subject(level, value)])) == level


def test_it_repairs_a_contradiction_whatever_the_parser_said(org):
    """The observed failure shape."""
    ir = _ir(subject_level="advisor", subjects=[_subject("company", "Acme Holdings")])
    assert _level(org, ir) == "company"


def test_it_leaves_an_already_consistent_ir_alone(org):
    ir = _ir(subject_level="team", subjects=[_subject("team", "Team Alpha")])
    assert _level(org, ir) == "team"


# =====================================================================
# 4-6. the three exclusions
# =====================================================================
def test_group_by_keeps_its_own_reporting_level(org):
    """"<group>'s advisors by connects" reports one level's figures
    broken out at another — the subject's level is not the answer's."""
    ir = _ir(subject_level="advisor", group_by="advisor",
             subjects=[_subject("company", "Acme Holdings")])
    assert _level(org, ir) == "advisor"


def test_a_hierarchy_read_keeps_its_target_level(org):
    """A hierarchy read enumerates a level BENEATH the subject, so the
    subject's own level is the one level it must not become."""
    ir = _ir(subject_level="advisor", target_level="advisor", subject_of="company",
             subjects=[_subject("company", "Acme Holdings")], metric=None)
    result = validate_ir(ir, org)
    assert result.ir.subject_level == "advisor"
    assert result.ir.target_level == "advisor"


def test_a_comparison_keeps_the_level_it_was_given(org):
    """Two subjects carry their own levels and may differ from each
    other; there is no single subject to take the level from."""
    ir = _ir(intent="comparison", operation="comparison", subject_level="team",
             subjects=[_subject("team", "Team Alpha"), _subject("team", "Team Beta")],
             sort=Sort(metric="total_connects"))
    assert _level(org, ir) == "team"


def test_a_mixed_level_comparison_is_untouched(org):
    ir = _ir(intent="comparison", operation="comparison", subject_level="company",
             subjects=[_subject("company", "Acme Holdings"), _subject("team", "Team Alpha")],
             sort=Sort(metric="total_connects"))
    assert _level(org, ir) == "company"


# =====================================================================
# THE SUBJECT IS SOMETIMES THE SCOPE, NOT THE ANSWER
# =====================================================================
class TestAScopeSubjectIsNotTheAnswersLevel:
    """"the <level>s IN <container>" carries the container as its subject
    and names the ENUMERATED level in `subject_level`. Structurally it is
    identical to an own-figure question — one subject, no `group_by`, no
    `target_level` — so the first version of this normalization collapsed
    it: "top teams in <a company> by revenue" answered with that
    company's own total instead of ranking its teams.

    The two families separate on the operation, which is what the guard
    now reads."""

    def test_a_leaderboard_keeps_the_level_it_ranks_over(self, org):
        ir = _ir(intent="leaderboard", operation="leaderboard",
                 subject_level="team",
                 subjects=[_subject("company", "Acme Holdings")],
                 sort=Sort(metric="total_connects", direction="desc"))
        assert _level(org, ir) == "team"

    def test_a_population_keeps_the_level_it_enumerates(self, org):
        ir = _ir(operation="population", subject_level="advisor", metric=None,
                 subjects=[_subject("company", "Acme Holdings")])
        assert _level(org, ir) == "advisor"

    def test_a_scoped_leaderboard_still_ranks_members_not_the_container(self, org):
        """The behaviour, not just the field: the answer must be a
        ranking of MEMBERS, never collapsed to the single container row.
        That collapse is exactly what the un-narrowed guard produced.

        Deliberately asserts the shape rather than the membership. The
        scope itself is NOT applied by the compiler — a subject whose
        level differs from `subject_level` is dropped rather than turned
        into a filter, so a team outside this company still appears here.
        That is a separate, pre-existing defect (it predates this guard
        and is why the collapse went unnoticed); pinning the corrected
        membership here would assert a fix that has not been made."""
        ir = _ir(intent="leaderboard", operation="leaderboard",
                 subject_level="team",
                 subjects=[_subject("company", "Acme Holdings")],
                 sort=Sort(metric="total_connects", direction="desc"))
        rows = compile_and_run(org, validate_ir(ir, org).ir)
        names = [r["name"] for r in rows]
        assert len(names) > 1, "collapsed to a single row"
        assert "Acme Holdings" not in names, "answered with the container"
        assert {"Team Alpha", "Team Beta"} <= set(names)

    def test_an_own_figure_query_over_the_same_subject_still_normalizes(self, org):
        """The other half of the pair, to show the exclusion is narrow:
        same subject, same measure, non-ranking operation."""
        ir = _ir(operation="group_metric", subject_level="advisor",
                 subjects=[_subject("company", "Acme Holdings")])
        result = validate_ir(ir, org)
        assert result.ir.subject_level == "company"
        rows = compile_and_run(org, result.ir)
        assert [r["name"] for r in rows] == ["Acme Holdings"]

    def test_the_exclusion_is_keyed_on_the_resolved_operation(self, org):
        """`operation` may be null and derived from `intent`, so the guard
        must read the RESOLVED value — otherwise the exclusion is bypassed
        by simply leaving the field unset.

        Only `leaderboard` is exercised: `population` is an operation with
        no corresponding intent, so an IR with `operation=None` can never
        resolve to it, and constructing one is impossible rather than
        merely untested."""
        ir = _ir(intent="leaderboard", operation=None, subject_level="team",
                 subjects=[_subject("company", "Acme Holdings")],
                 sort=Sort(metric="total_connects", direction="desc"))
        assert ir.resolved_operation() == "leaderboard"
        assert _level(org, ir) == "team"


# =====================================================================
# 7-8. what must not move
# =====================================================================
def test_a_ranking_that_names_no_subject_is_untouched(org):
    """"top <level> by <measure>" ranks OVER a level and names no
    subject — there is nothing to copy from, and the level it already
    carries is the answer."""
    ir = _ir(intent="leaderboard", operation="leaderboard", subject_level="team",
             subjects=[], sort=Sort(metric="total_connects", direction="desc"))
    assert _level(org, ir) == "team"


def test_a_population_filtered_without_a_subject_is_untouched(org):
    ir = _ir(operation="population", subject_level="advisor", subjects=[], metric=None,
             filters=[Filter(field="company", operator="=", value="Acme Holdings")])
    assert _level(org, ir) == "advisor"


def test_every_subject_type_is_a_level_that_can_be_copied_in():
    """THE GUARD'S REASON. The normalization copies `subjects[0].type`
    into `subject_level`, so every value the first field can hold must be
    a value the second can. That holds today — which makes the
    `is_valid_level` check defensive rather than dead — and this is what
    would catch the two vocabularies drifting apart.

    `QueryIR.subject_level` is not validated on assignment, so a
    divergence would be written silently and surface much later as an
    unanswerable level rather than as an error here."""
    import typing
    from app.llm import hierarchy
    from app.llm.query_ir import Level

    subject_types = typing.get_args(Subject.model_fields["type"].annotation)
    levels = set(typing.get_args(Level))
    for t in subject_types:
        assert t in levels, f"{t} is a subject type but not a level"
        assert hierarchy.is_valid_level(t), f"{t} is a subject type the guard rejects"


def test_a_type_the_guard_rejects_leaves_the_level_alone(org):
    """The guard's behaviour, exercised directly. Pydantic makes a bad
    `Subject.type` unconstructible, so the condition is reached by
    stubbing the predicate — the point is that a rejected type is skipped
    rather than copied in."""
    import app.llm.ir_validator as validator

    ir = _ir(subject_level="advisor", subjects=[_subject("company", "Acme Holdings")])
    original = validator.hierarchy.is_valid_level
    validator.hierarchy.is_valid_level = lambda level: False
    try:
        assert validate_ir(ir, org).ir.subject_level == "advisor"
    finally:
        validator.hierarchy.is_valid_level = original


def test_the_metric_degrade_still_has_the_last_word(org):
    """Ordering guard. The degrade exists to rescue a level the compiler
    cannot serve, so it must run AFTER this normalization, not before.
    Every real (metric, level) pair is answerable today, so this asserts
    the ordering rather than a live degrade: the normalized level
    survives because nothing needed rescuing."""
    ir = _ir(subject_level="advisor", subjects=[_subject("company", "Acme Holdings")])
    result = validate_ir(ir, org)
    assert result.missing == []
    assert result.ir.subject_level == "company"
