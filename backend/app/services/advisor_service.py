"""
Advisor reads against the advisor_profile VIEW — one query across all
star-schema tables instead of six separate joins, since this is the
chatbot's most common lookup pattern.

Phase 2: person RESOLUTION (name -> which human) now lives entirely in
app/llm/advisor_resolver.py. This module only FETCHES profile rows, and
only ever by wid. The split matters because the two were previously
conflated in one query:

    SELECT * FROM advisor_profile WHERE name ILIKE '%q%' ORDER BY wid LIMIT 1

which returned a different person in two independent ways — substring
containment ("Ahmed Ali" -> "Ahmed Ali Pirzada") and silent lowest-wid
selection among the 238 duplicate-name groups in production.

Neither `ILIKE '%…%'` nor `LIMIT 1` appears anywhere in person resolution
now. `find_advisor_by_name` is gone: callers resolve first (getting a
ResolvedAdvisor, which may be ambiguous and require asking the user), then
fetch by wid.
"""

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from app.database.models import Advisor
from app.llm import advisor_resolver
from app.llm.advisor_resolver import ResolvedAdvisor
from app.llm.metric_ontology import METRICS


def find_advisor_by_wid(db: Session, wid: int) -> dict | None:
    """The only identity-bearing read. Exact by construction — it cannot
    return a different person."""
    row = db.execute(
        text("SELECT * FROM advisor_profile WHERE wid = :wid"),
        {"wid": wid},
    ).mappings().first()
    return dict(row) if row else None


def get_advisor_metric(db: Session, wid: int, metric_key: str) -> float | None:
    """ONE metric for ONE person, or None when the metric has no
    advisor-level binding or the person has no row in its fact table.

    Read through the ontology binding rather than off the advisor_profile
    view: the binding is already the single answer to "where does this
    metric's value live", and several metrics are computed expressions
    (total_connects is new + follow-up) rather than columns. A view-column
    lookup table would be a second, drifting copy of that knowledge, and
    would need extending every time a metric is added — whereas this
    works for any metric the ontology can already express.

    Keyed by WID, never by name: a name is not an identifier here (238
    duplicate-name groups in production), and this function exists to
    answer a question about one specific person.
    """
    metric = METRICS.get(metric_key)
    if metric is None:
        return None
    binding = metric.bindings.get("advisor")
    if binding is None or binding.team_named:
        return None

    query = db.query(binding.expr).select_from(Advisor)
    # Advisor is the query root, so "joining" it is a no-op — a metric
    # whose column lives on the advisor row (1-Unit ownership) bound here
    # produced `FROM advisors, advisors` and an ambiguous-column error.
    if binding.model is not Advisor:
        query = query.join(binding.model, binding.model.wid == Advisor.wid)
    # A ratio may span two fact tables (Connect->CR divides client
    # registrations by answered calls). Referencing the second without
    # joining it does not fail — SQLAlchemy cross-joins it, which here
    # returned several rows and made .scalar() raise MultipleResultsFound.
    # Same declaration the compiler and the aggregation engine honour.
    for extra in getattr(binding, "join_models", ()) or ():
        query = query.join(extra, extra.wid == Advisor.wid)
    query = query.filter(Advisor.wid == wid)
    if binding.period is not None:
        query = query.filter(binding.model.period == binding.period)

    result = query.scalar()
    return float(result) if result is not None else None


def find_advisors_by_wids(db: Session, wids: list[int]) -> list[dict]:
    """Profile rows for several wids at once — used to render a
    disambiguation list without N round trips."""
    if not wids:
        return []
    # expanding bindparam — renders a proper parameterized IN list on both
    # Postgres and SQLite, so no dialect branching and no string building
    stmt = text("SELECT * FROM advisor_profile WHERE wid IN :wids").bindparams(
        bindparam("wids", expanding=True)
    )
    rows = db.execute(stmt, {"wids": list(wids)}).mappings().all()
    return [dict(r) for r in rows]


def resolve_advisor(db: Session, query: str) -> ResolvedAdvisor:
    """Name/text -> ResolvedAdvisor {wid, name, confidence, candidates}.
    Thin pass-through so service-layer callers don't need to reach into
    the NLU package directly; all policy lives in advisor_resolver."""
    return advisor_resolver.resolve_advisor(query, db)


def find_advisor_candidates(db: Session, query: str, limit: int = 10) -> list[dict]:
    """Every advisor profile a name could refer to.

    Resolution is delegated to advisor_resolver (exact -> high-confidence
    fuzzy, no substring tier), so this can no longer return a person whose
    name merely CONTAINS the query. Returns [] rather than a loose guess
    when nothing clears the person floor."""
    resolution = advisor_resolver.resolve_advisor(query, db)
    if not resolution.candidates:
        return []
    wids = [c.wid for c in resolution.candidates][:limit]
    profiles = {p["wid"]: p for p in find_advisors_by_wids(db, wids)}
    # preserve the resolver's ordering (best score first)
    return [profiles[w] for w in wids if w in profiles]
