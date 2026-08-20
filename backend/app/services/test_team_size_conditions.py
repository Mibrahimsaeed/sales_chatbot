"""Team size is a metric, so it can be ranked and compared.

The COUNT was never missing. aggregation.headcount() has answered "team
size of X" throughout, and every working-day rate divides by the same
row count (metric_ontology, above cr_rate: "the row count IS the team
size"). What was missing is that it was never DECLARED, so it could not
reach an IR as a sort metric or a filter field:

    "BCMs with team size > 1"        -> "I'm not tracking that one"
    "Unit Heads with team size > 5"  -> fallback_reasoning widened onto
                                        `one_unit_ratio` from the stray
                                        word "Unit" and filtered on it

The second is the worse half — a neighbouring measure, applied
confidently, rather than a refusal.

The fix declares `team_size` with a `literal(1)` advisor binding rolled
up by SUM, which is COUNT(rows in scope) over the same master-sheet
population headcount() walks. ONE definition reached two ways, and these
tests assert that equality directly rather than trusting it: every
displayed figure is compared against headcount() for that same manager.

The comparator, the HAVING placement, the condition column and pagination
are all pre-existing machinery — nothing here changes them, and several
tests below exist to prove that.
"""

import pytest

from app.database.models import Advisor
from app.llm import aggregation, entity_extractor, nlu_pipeline
from app.llm.query_compiler import compile_and_run, count_ir
from app.services import chat_service

# BCM `i` has exactly `i` advisors, so every threshold below has a
# predictable, hand-checkable answer and the boundary cases are real
# rows rather than contrivances.
_MAX_BCM = 20
_PAGE = chat_service.PAGE_SIZE


@pytest.fixture()
def org(db_session):
    wid = 0
    for i in range(1, _MAX_BCM + 1):
        bcm = f"BCM{i:02d}"
        zonal = "ZH1" if i <= 10 else "ZH2"
        for j in range(i):
            wid += 1
            db_session.add(Advisor(
                wid=wid, name=f"Adv{i:02d}_{j:02d}", team=f"T{i % 3}",
                company="Graana", rm="UH1", portfolio_lead=zonal,
                management_lead=bcm, in_master_sheet=True))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    return db_session


def _rows(db, text, all_rows=True):
    """Every matching row for `text`, through the real pipeline."""
    resolution = nlu_pipeline.resolve(text, db, session_id=None)
    assert resolution.ir is not None, f"no IR for {text!r}: kind={resolution.kind}"
    ir = resolution.ir
    if all_rows:
        ir = ir.model_copy(update={"limit": None})
    return resolution.ir, compile_and_run(db, ir) or []


def _sizes(rows):
    return {r["name"]: r["value"] for r in rows}


def _expected(pool, op, n):
    import operator
    ops = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}
    return {name for name, size in pool.items() if ops[op](size, n)}


_BCM_POOL = {f"BCM{i:02d}": i for i in range(1, _MAX_BCM + 1)}


# ------------------------------------------------------------ 1-4, 7
@pytest.mark.parametrize("op,n", [(">", 5), ("<", 5), (">=", 5), ("<=", 5)])
def test_bcm_team_size_comparisons(op, n, org):
    ir, rows = _rows(org, f"BCMs with team size {op} {n}")

    assert ir.subject_level == "bcm"
    assert [(f.field, f.operator, f.value) for f in ir.filters] == [("team_size", op, float(n))]
    assert _sizes(rows).keys() == _expected(_BCM_POOL, op, n)


# --------------------------------------------------------------- 5
def test_boundary_values_are_exact(org):
    """>= and <= include the boundary; > and < exclude it. Asserted on the
    row that IS the boundary, not on counts that could coincide."""
    boundary = "BCM05"

    assert boundary in _sizes(_rows(org, "BCMs with team size >= 5")[1])
    assert boundary in _sizes(_rows(org, "BCMs with team size <= 5")[1])
    assert boundary not in _sizes(_rows(org, "BCMs with team size > 5")[1])
    assert boundary not in _sizes(_rows(org, "BCMs with team size < 5")[1])


def test_the_two_halves_partition_the_pool(org):
    """`> n` and `<= n` are complements, so together they must be every
    BCM exactly once — a filter that is off by one at the boundary breaks
    this even when both counts look plausible alone."""
    above = _sizes(_rows(org, "BCMs with team size > 5")[1]).keys()
    at_or_below = _sizes(_rows(org, "BCMs with team size <= 5")[1]).keys()

    assert set(above) | set(at_or_below) == set(_BCM_POOL)
    assert not set(above) & set(at_or_below)


# --------------------------------------------------------------- 6
def test_zonal_head_comparisons(org):
    """ZH1 has BCMs 1-10 (55 advisors), ZH2 has 11-20 (155)."""
    _, rows = _rows(org, "Zonal Heads with team size >= 100")
    assert _sizes(rows) == {"ZH2": 155}

    _, rows = _rows(org, "Zonal Heads with team size < 100")
    assert _sizes(rows) == {"ZH1": 55}


# --------------------------------------------------------------- 7
def test_unit_head_comparisons(org):
    """UH1 holds everyone: 1+2+...+20 = 210."""
    _, rows = _rows(org, "Unit Heads with team size > 5")
    assert _sizes(rows) == {"UH1": 210}

    _, rows = _rows(org, "Unit Heads with team size < 5")
    assert _sizes(rows) == {}


def test_unit_heads_no_longer_resolve_the_one_unit_ratio(org):
    """The silent misresolution: "Unit Heads with team size > 5" widened
    onto `one_unit_ratio` because of the word "Unit"."""
    ir, _ = _rows(org, "Unit Heads with team size > 5")
    assert ir.sort.metric == "team_size"
    assert all(f.field == "team_size" for f in ir.filters)


# --------------------------------------------------------------- 8
def test_people_under_them_phrasing(org):
    ir, rows = _rows(org, "BCMs with more than 5 people under them")
    assert [(f.field, f.operator, f.value) for f in ir.filters] == [("team_size", ">", 5.0)]
    assert _sizes(rows).keys() == _expected(_BCM_POOL, ">", 5)


def test_the_reported_phrasings_all_resolve(org):
    for text in ("BCM list with team size greater than 1",
                 "BCMs with team size > 1",
                 "BCMs with team size greater than 1"):
        ir, rows = _rows(org, text)
        assert ir.sort.metric == "team_size", text
        assert _sizes(rows).keys() == _expected(_BCM_POOL, ">", 1), text


# ------------------------------------------------------------- 9, 10
def test_every_returned_bcm_satisfies_the_condition(org):
    _, rows = _rows(org, "BCMs with team size > 5")
    assert rows
    for row in rows:
        assert row["value"] > 5, row


def test_no_qualifying_bcm_is_missing(org):
    """Stated as the full set, so an over-restrictive filter fails here
    even though every row it DID return was valid."""
    _, rows = _rows(org, "BCMs with team size > 5")
    assert _sizes(rows).keys() == {f"BCM{i:02d}" for i in range(6, _MAX_BCM + 1)}


# ------------------------------------------------------------- 11
def test_displayed_team_size_equals_headcount(org):
    """THE equality this fix rests on: the number shown and the number
    filtered on are one calculation, and it is the one headcount()
    already performed for "team size of X"."""
    _, rows = _rows(org, "BCMs with team size > 1")
    assert rows
    for row in rows:
        assert row["value"] == aggregation.headcount(org, "bcm", row["name"]), row["name"]


def test_the_condition_metric_gets_its_own_column(org):
    """Team size is a filtered metric, so the conditional-column work
    displays it without needing to know what team size is."""
    ir, rows = _rows(org, "BCMs with team size > 5", all_rows=False)
    keys = chat_service._attach_bundle_columns(org, ir, rows)

    assert keys == ["team_size"]
    assert rows[0]["columns"]["team_size"]["label"] == "Team Size"
    assert rows[0]["columns"]["team_size"]["value"] == rows[0]["value"]


# --------------------------------------------------------- 12, 13, 14
def test_more_than_one_page_of_matches(org):
    ir, _ = _rows(org, "BCMs with team size >= 1")
    total = count_ir(org, ir)
    assert total == _MAX_BCM
    assert total > _PAGE, "fixture must exceed one page for this to mean anything"

    page1 = compile_and_run(org, chat_service._page_ir(ir, 0, total), offset=0)
    assert len(page1) == _PAGE


def test_show_more_preserves_the_condition_and_the_column(org):
    ir, _ = _rows(org, "BCMs with team size >= 1")
    total = count_ir(org, ir)

    page1 = compile_and_run(org, chat_service._page_ir(ir, 0, total), offset=0)
    keys1 = chat_service._attach_bundle_columns(org, ir, page1)
    offset = len(page1)
    page2 = compile_and_run(org, chat_service._page_ir(ir, offset, total), offset=offset)
    keys2 = chat_service._attach_bundle_columns(org, ir, page2)

    assert keys1 == keys2 == ["team_size"]
    assert page2
    for row in page1 + page2:
        assert row["value"] >= 1


def test_pages_carry_every_bcm_exactly_once(org):
    ir, _ = _rows(org, "BCMs with team size >= 1")
    total = count_ir(org, ir)
    page1 = compile_and_run(org, chat_service._page_ir(ir, 0, total), offset=0)
    offset = len(page1)
    page2 = compile_and_run(org, chat_service._page_ir(ir, offset, total), offset=offset)

    names = [r["name"] for r in page1 + page2]
    assert len(names) == _MAX_BCM
    assert len(set(names)) == _MAX_BCM


# ------------------------------------------------------------- 15-17
def test_an_unconditional_bcm_leaderboard_is_unchanged(org):
    """No condition, a different measure: team_size must not have become
    the default for anything."""
    resolution = nlu_pipeline.resolve("top 5 BCMs by connects", org, session_id=None)
    assert resolution.ir is not None
    assert resolution.ir.sort.metric != "team_size"
    assert resolution.ir.filters == []


def test_team_of_x_is_unchanged(org):
    """"team of X" keeps the nested breakdown — requirement 16."""
    resolution = nlu_pipeline.resolve("team of ZH1", org, session_id=None)
    assert resolution.kind == "plan"
    assert resolution.plan.action == "breakdown"


def test_team_size_of_x_is_unchanged(org):
    """Requirement 17. A NAMED group keeps the breakdown that Phase 37
    settled — declaring the metric must not take the query over.

    This is what _answers_as_group_shape exists for, and it is the one
    thing this change had to be careful about: the metric resolves for
    "team size of ZH1" just as it does for "BCMs with team size > 5", and
    only the presence of a named group tells the two apart.
    """
    resolution = nlu_pipeline.resolve("team size of ZH1", org, session_id=None)

    assert resolution.kind == "plan"
    assert resolution.plan.action == "breakdown"
    assert resolution.ir is None, "a named group must not take the metric path"

    reply = str(chat_service._dispatch(org, resolution).get("reply") or "")
    assert f"{aggregation.headcount(org, 'zonal_head', 'ZH1')} advisors" in reply
    assert "55 advisors" in reply


def test_the_named_group_is_what_separates_the_two_readings(org):
    """The same words, with and without a named group."""
    named = nlu_pipeline.resolve("team size of ZH1", org, session_id=None)
    ranked = nlu_pipeline.resolve("BCMs with team size > 5", org, session_id=None)

    assert named.ir is None
    assert ranked.ir is not None and ranked.ir.sort.metric == "team_size"


def test_a_roster_is_unchanged(org):
    resolution = nlu_pipeline.resolve("all advisors in T1", org, session_id=None)
    assert resolution.kind == "plan"
    assert resolution.plan.action == "roster"


# --------------------------------------------------------------- 18
def test_multi_role_managers_keep_the_highest_role_rule(db_session):
    """A person who is both a BCM and a Zonal Head belongs at the senior
    level (_exclude_more_senior_roles), and declaring a new metric must
    not change which rows a BCM listing contains."""
    for wid, name, zonal, bcm in [
        (1, "A1", "Dual", "Dual"),      # Dual is his own BCM
        (2, "A2", "Dual", "Dual"),
        (3, "A3", "Dual", "PlainBCM"),
        (4, "A4", "Dual", "PlainBCM"),
        (5, "A5", "Dual", "PlainBCM"),
    ]:
        db_session.add(Advisor(wid=wid, name=name, team="T", company="Graana",
                               rm="UH", portfolio_lead=zonal, management_lead=bcm,
                               in_master_sheet=True))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0

    _, rows = _rows(db_session, "BCMs with team size >= 1")
    names = _sizes(rows)

    # "Dual" is named at zonal_head, so he is not listed as a BCM.
    assert "Dual" not in names
    assert names == {"PlainBCM": 3}
    # And he is still a Zonal Head, with the whole group.
    _, zonal_rows = _rows(db_session, "Zonal Heads with team size >= 1")
    assert _sizes(zonal_rows) == {"Dual": 5}
