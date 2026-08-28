"""
Generic summary + nested-breakdown service for any hierarchy level
(app/llm/hierarchy.py) — powers the three NEW levels this feature adds
(unit_head, zonal_head, business_center), driven entirely by
hierarchy.LEVEL_COLUMNS instead of one hand-written function per level.

PHASE 4 UPDATE. This module used to say team_service.py and
company_service.py were "intentionally left untouched" because
duplicating their behaviour here risked a regression. That reasoning
kept three near-identical summaries alive, and they drifted. All three
now delegate to app/llm/aggregation.py, which owns the roll-up rules;
team_service still reads its team-named TeamTarget figures explicitly,
because those are a separate published source rather than a roll-up.

get_level_breakdown is the new capability itself (decision: "the default
response should always be nested by team"): for a specific unit_head/
zonal_head/business_center/company, list its advisors GROUPED BY TEAM
rather than collapsed into one flat aggregate — a Unit Head or Zonal Head
can oversee multiple teams/business centers, and the nested shape makes
that structure visible instead of hiding it behind a single number.

get_level_flat_list is the explicit opt-in for the alternative (originally
deferred — "if a user explicitly asks to list all advisors, we can support
a flat list later" — now requested): same top line, an ungrouped advisor
list instead of nested-by-team.
"""

from __future__ import annotations

from sqlalchemy import and_, distinct, func
from sqlalchemy.orm import Session

from app.core.exception import NotFoundError
from app.database.models import Advisor, Performance, PerformancePeriod, Pipeline, SalesFunnel
from app.llm import aggregation, hierarchy


def get_level_roster(db: Session, level: str, value: str) -> dict:
    """WHO IS IN a group — a plain list of advisors, no aggregate metrics.

    Distinct from get_level_summary ("how is this group doing") and
    get_level_breakdown ("the group, nested by team, with totals"). A
    roster question — "all advisors in Blue Area" — asks to ENUMERATE
    people, and answering it with connects/pipeline/overdue totals is
    answering a different question.

    Works for any hierarchy level via hierarchy.column_for(), so team,
    company, business_center, unit_head and zonal_head all share one
    implementation."""
    column = hierarchy.column_for(level)
    if column is None:
        raise ValueError(f"'{level}' is not a known hierarchy level")

    rows = (
        db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company)
        .filter(column.ilike(value), Advisor.in_master_sheet.is_(True))
        .order_by(Advisor.name)
        .all()
    )
    if not rows:
        raise NotFoundError(f"No {hierarchy.label_for(level)} matching '{value}'")

    return {
        "level": level,
        "level_label": hierarchy.label_for(level),
        "value": value,
        "count": len(rows),
        "advisors": [
            {"wid": r.wid, "name": r.name, "team": r.team, "company": r.company}
            for r in rows
        ],
    }


def _direct_members(db: Session, level: str, value: str, target: str) -> list[dict]:
    """`value`'s immediate reports at exactly `target`, as member dicts.

    NAMES at a manager target and advisor ROWS at the leaf: a manager
    level is a column of names with no row of its own, while an advisor
    is a person with a wid. Deliberately no aggregate — the count is
    len() of this, so the number and the list can never come from two
    different scopes.
    """
    return _members_for(db, hierarchy.direct_scope_filter(level, value, target), target)


def _members_for(db: Session, predicate, target: str) -> list[dict]:
    """The member dicts for an already-built scope predicate.

    Shared by the direct and transitive readings so they cannot drift on
    what a member IS — names at a manager target, advisor rows at the
    leaf, master-sheet filtered, in one order.
    """
    if predicate is None:
        return []

    if target == "advisor":
        rows = (
            db.query(Advisor.wid, Advisor.name, Advisor.team, Advisor.company)
            .filter(predicate, Advisor.in_master_sheet.is_(True))
            .order_by(Advisor.name)
            .all()
        )
        return [{"wid": r.wid, "name": r.name, "team": r.team,
                 "company": r.company} for r in rows]

    column = hierarchy.column_for(target)
    rows = (
        db.query(distinct(column))
        .filter(predicate, Advisor.in_master_sheet.is_(True), column.isnot(None))
        .all()
    )
    return [{"name": name} for name in sorted({r[0] for r in rows if r[0]})]


def get_direct_reports(db: Session, level: str, value: str,
                       target_level: str | None = None) -> dict | None:
    """Who reports to `value` IMMEDIATELY — not the whole subtree.

    The mirror of get_manager_of_group: that reads the level above a
    group, this reads the level below a person. Both derive the pair from
    the chain rather than naming it, so neither needs touching when CHAIN
    is rebound.

    ONE POPULATION FOR THE COUNT AND THE LIST. `count` is len(members),
    not a second query — "how many advisors report directly to X" and
    "who reports directly to X" are the same question asked two ways, and
    answering them from two scopes is how they come to disagree.
    """
    level = hierarchy.canonical_level(level)
    named_target = hierarchy.canonical_level(target_level)
    target = named_target or hierarchy.child_of(level)
    if target is None:
        return None

    members = _direct_members(db, level, value, target)

    # WHERE THE DEFAULT TARGET IS EMPTY, KEEP DESCENDING. A person is
    # routinely their own sub-level here — Ch Muhammad Usman is the only
    # BCM beneath himself as Zonal Head — so once self is excluded the
    # level immediately below holds nobody, while four advisors DO name
    # him as their immediate manager. "Nobody reports to him" would be
    # false in the only sense the question means.
    #
    # Skipped when the caller NAMED a target: "how many advisors report
    # directly to X" asks about one level and must answer about that
    # level, empty or not.
    if named_target is None and not members:
        for candidate in hierarchy.descendants(target):
            found = _direct_members(db, level, value, candidate)
            if found:
                target, members = candidate, found
                break

    return {
        "level": level,
        "level_label": hierarchy.label_for(level),
        "value": value,
        "target_level": target,
        "target_level_label": hierarchy.label_for(target),
        "count": len(members),
        "members": members,
    }


def get_scoped_reports(db: Session, level: str, value: str,
                       target_level: str) -> dict | None:
    """Everyone at `target_level` ANYWHERE beneath `value`.

    The transitive twin of get_direct_reports: "which BCMs work under
    Unit Head X" rather than "who reports directly to X". Deliberately
    the same shape, the same keys and the same one-population rule — the
    count is len(members), so "how many BCMs work under X" and "which
    BCMs work under X" are one question answered from one scope.

    `target_level` is REQUIRED here, unlike the direct version. The
    direct reading has a sensible default (the rung immediately below);
    the transitive one does not — "everyone under X" without a named
    level is the existing roster/breakdown question and must keep going
    there, so a caller with no target is a caller that should not have
    reached this function.

    Returns None when the target is not a level strictly beneath `level`,
    which the caller renders as "not a question about this level".
    """
    level = hierarchy.canonical_level(level)
    target = hierarchy.canonical_level(target_level)
    predicate = hierarchy.subtree_scope_filter(level, value, target)
    if predicate is None:
        return None

    members = _members_for(db, predicate, target)
    return {
        "level": level,
        "level_label": hierarchy.label_for(level),
        "value": value,
        "target_level": target,
        "target_level_label": hierarchy.label_for(target),
        "count": len(members),
        "members": members,
    }


def get_manager_of(db: Session, wid: int, level: str) -> dict | None:
    """Reverse hierarchy lookup: who is this advisor's unit head / zonal
    head / business center? (Phase 1 identity refactor — keyed by WID, the
    only identifier that addresses one specific person.)

    Returns None when the advisor exists but has no manager recorded at
    that level, and None when the wid doesn't exist — the caller
    distinguishes those by looking the advisor up first, so that "no BM on
    file" and "no such advisor" produce different replies rather than one
    vague miss."""
    # manager_column_for, not column_for: reverse lookup covers fields
    # that are deliberately NOT grouping levels (regional_manager/rm,
    # portfolio_lead, management_lead). "Who is Yasir's RM?" is a valid
    # question about one person's manager even though RM is not something
    # you can rank or roll up by.
    column = hierarchy.manager_column_for(level)
    if column is None:
        raise ValueError(f"'{level}' is not a reverse-lookupable hierarchy field")

    row = (
        db.query(Advisor.wid, Advisor.name, column.label("manager"))
        .filter(Advisor.wid == wid)
        .first()
    )
    if row is None or not row.manager:
        return None
    return {
        "wid": row.wid,
        "advisor": row.name,
        "level": level,
        "level_label": hierarchy.label_for(level),
        "manager": row.manager,
    }


def get_manager_of_group(db: Session, level: str, value: str,
                        target_level: str) -> dict | None:
    """Who sits ABOVE a group — a BCM's zonal head, a zonal head's unit
    head, a team's... nothing, since team is the root.

    Phase 5.4. Reverse lookup previously required an ADVISOR subject, so
    "which zonal head oversees BCM Usman Ghani" fell through to a
    breakdown and answered with that BCM's advisors — a confident list
    that is not what was asked. The chain has always supported the
    question (hierarchy.parent_of); nothing exposed it.

    A group has no row of its own, so the manager is read off the
    advisors IN it: everyone under a BCM shares that BCM's zonal head.
    Returning EVERY distinct value rather than the first is deliberate —
    if a group spans two managers the data contradicts the chain, and the
    caller should say so instead of picking one.
    """
    target_column = hierarchy.column_for(target_level)
    if target_column is None:
        raise ValueError(f"'{target_level}' is not a known hierarchy level")

    rows = (
        db.query(distinct(target_column))
        .filter(hierarchy.scope_filter(level, value),
                Advisor.in_master_sheet.is_(True),
                target_column.isnot(None))
        .all()
    )
    managers = sorted({r[0] for r in rows if r[0]})
    if not managers:
        return None

    return {
        "level": level,
        "level_label": hierarchy.label_for(level),
        "value": value,
        "target_level": target_level,
        "target_level_label": hierarchy.label_for(target_level),
        "managers": managers,
    }


def get_ancestry(db: Session, level: str, value: str) -> dict | None:
    """Every level ABOVE this one, innermost first.

    Generic over the chain: `hierarchy.ancestors(level)` gives the levels
    and get_manager_of_group reads the value at each. An advisor's
    ancestry is BCM -> Zonal Head -> Unit Head -> Team; a BCM's is Zonal
    Head -> Unit Head -> Team. No level is named here, so the day the
    chain changes this follows it.
    """
    if not hierarchy.is_chain_level(hierarchy.canonical_level(level)):
        return None

    canonical = hierarchy.canonical_level(level)
    chain: list[dict] = []
    for ancestor in hierarchy.ancestors(canonical):
        found = get_manager_of_group(db, canonical, value, ancestor)
        if found is None:
            continue
        chain.append({
            "level": ancestor,
            "level_label": hierarchy.label_for(ancestor),
            # A well-formed group has exactly one manager per ancestor
            # level; `managers` carries the rest when it does not.
            "value": found["managers"][0],
            "managers": found["managers"],
        })

    if not chain:
        return None
    return {
        "level": canonical,
        "level_label": hierarchy.label_for(canonical),
        "value": value,
        "ancestry": chain,
    }


def get_level_summary(db: Session, level: str, value: str) -> dict:
    """Flat aggregate across every advisor in scope at `level`.

    PHASE 4: this was the third near-copy of the same three hand-written
    `func.sum` queries (team_service and company_service had the other
    two, differing only in which Advisor column they filtered on). All
    three now call `aggregation.summary`, which is also what comparisons
    and leaderboards aggregate through — so a unit head's connects cannot
    depend on which endpoint asked.
    """
    if hierarchy.column_for(level) is None:
        raise ValueError(f"'{level}' is not a known hierarchy level")

    summary = aggregation.summary(db, level, value)
    if not summary["advisors"]:
        raise NotFoundError(f"No {hierarchy.label_for(level)} matching '{value}'")
    return summary


def nesting_level(level: str) -> str:
    """The level a breakdown of `level` should GROUP BY.

    Phase 3: the child of `level` in the declared chain, not a hardcoded
    "team". A breakdown of a Unit Head now nests by Zonal Head, and of a
    Zonal Head by BCM — the structure the org actually has. The previous
    fixed 2-hop (any level -> team -> advisor) skipped every intermediate
    level, so a Unit Head breakdown showed teams and hid the two layers
    between.

    Falls back to `advisor` for an ATTRIBUTE level (company, office,
    region), which has no child in the chain — grouping those by the leaf
    is the only meaningful nesting available.
    """
    return hierarchy.child_of(level) or "advisor"


def _fetch_advisor_rows(db: Session, level: str, value: str):
    """Shared by get_level_breakdown (nested) and get_level_flat_list
    (flat) — same per-advisor query, just consumed two different ways, so
    the join/filter logic lives in one place.

    Selects the NESTING column alongside the advisor so the caller can
    group by whatever the chain says sits below `level`.
    """
    nest_column = hierarchy.column_for(nesting_level(level))
    return (
        db.query(
            Advisor.wid,
            Advisor.name,
            nest_column.label("nest_key"),
            (SalesFunnel.mtd_new_connect + SalesFunnel.mtd_followup_connect).label("connects"),
            Performance.cleared.label("mtd_cleared"),
            Performance.target.label("mtd_target"),
        )
        .outerjoin(SalesFunnel, SalesFunnel.wid == Advisor.wid)
        .outerjoin(Performance, and_(Performance.wid == Advisor.wid, Performance.period == PerformancePeriod.MTD))
        .filter(hierarchy.scope_filter(level, value), Advisor.in_master_sheet.is_(True))
        .order_by(nest_column, Advisor.name)
        .all()
    )


def get_level_breakdown(db: Session, level: str, value: str) -> dict:
    """get_level_summary's totals PLUS the per-advisor detail, nested by
    team (decision: always nested by team, never a flat advisor list by
    default — see get_level_flat_list for the explicit opt-in) — raises
    NotFoundError (via get_level_summary) under the same condition
    team_service/company_service already do."""
    summary = get_level_summary(db, level, value)
    rows = _fetch_advisor_rows(db, level, value)

    groups: dict[str, list[dict]] = {}
    for r in rows:
        group_name = r.nest_key or "Unassigned"
        groups.setdefault(group_name, []).append({
            "wid": r.wid,
            "name": r.name,
            "connects": r.connects or 0,
            "mtd_cleared": r.mtd_cleared or 0,
            "mtd_target": r.mtd_target or 0,
        })

    child = nesting_level(level)
    return {
        **summary,
        # `nested_by` names the level these groups are — a consumer can no
        # longer assume "teams" the way the fixed 2-hop implied.
        "nested_by": child,
        "nested_by_label": hierarchy.label_for(child),
        "teams": [
            {"team": group_name, "advisor_count": len(advisors), "advisors": advisors}
            for group_name, advisors in groups.items()
        ],
    }


def get_level_flat_list(db: Session, level: str, value: str) -> dict:
    """The explicit flat opt-in (deferred in the original decision, now
    requested): same top-line summary as get_level_breakdown, but
    "advisors" is a single flat list — no team grouping — for a user who
    explicitly asked not to have the result nested."""
    summary = get_level_summary(db, level, value)
    rows = _fetch_advisor_rows(db, level, value)

    # "advisor_list", not "advisors" — get_level_summary already uses
    # "advisors" for the advisor COUNT (kept intact here via **summary, same
    # as get_level_breakdown does); reusing that key for the row list would
    # silently clobber the count instead of merging alongside it.
    return {
        **summary,
        "advisor_list": [
            {
                "wid": r.wid,
                "name": r.name,
                # The flat list keeps the "team" key for consumers, but it
                # now carries whatever the chain nests below `level` —
                # named by `nested_by` on the breakdown shape.
                "team": r.nest_key,
                "connects": r.connects or 0,
                "mtd_cleared": r.mtd_cleared or 0,
                "mtd_target": r.mtd_target or 0,
            }
            for r in rows
        ],
    }
