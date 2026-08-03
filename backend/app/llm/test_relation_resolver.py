"""Relationship resolution (app/llm/relation_resolver.py) — M1.

The two rules in the module docstring are the tests that matter: never
infer from an ambiguous source (risk R2 — the wrong-person failure class),
and never resolve an uncached relation (that is milestone M3, and doing
it here would issue database reads M1 never budgeted for).
"""

import pytest

from app.llm import advisor_resolver, relation_resolver
from app.llm.advisor_resolver import AdvisorIdentity, ResolvedAdvisor


WAQAR = AdvisorIdentity(wid=1, name="Waqar Haider", team="Blue Area", company="Graana")


def _resolved(*identities):
    return ResolvedAdvisor(status=advisor_resolver.RESOLVED, candidates=list(identities))


def _ambiguous(*identities):
    return ResolvedAdvisor(status=advisor_resolver.AMBIGUOUS, candidates=list(identities))


def test_resolves_a_cached_relation_from_an_identity():
    related = relation_resolver.resolve_from_identity(WAQAR, "team")
    assert related.value == "Blue Area"
    assert related.target_level == "team"
    assert related.source_id == 1
    assert related.provenance == "inferred:advisor:1"


def test_resolves_company_too():
    assert relation_resolver.resolve_from_identity(WAQAR, "company").value == "Graana"


def test_uncached_relations_return_none_in_m1():
    """unit_head/zonal_head/... are declared but not in the identity
    cache projection; resolving them is M3."""
    for level in ("unit_head", "zonal_head", "business_center",
                  "regional_manager", "portfolio_lead", "management_lead"):
        assert relation_resolver.resolve_from_identity(WAQAR, level) is None


def test_undeclared_relation_returns_none():
    assert relation_resolver.resolve_from_identity(WAQAR, "region") is None
    assert relation_resolver.resolve_from_identity(WAQAR, "nonsense") is None


def test_missing_value_on_the_source_row_returns_none():
    """An advisor with no team on file. "I don't have that" is a real
    outcome; fabricating one is not."""
    teamless = AdvisorIdentity(wid=2, name="Nadia Rehman", team=None, company="Graana")
    assert relation_resolver.resolve_from_identity(teamless, "team") is None


def test_none_identity_is_handled():
    assert relation_resolver.resolve_from_identity(None, "team") is None
    assert relation_resolver.resolve_from_identity(WAQAR, "") is None


# ---------------------------------------------------------------------
# Risk R2 — the wrong-person guard
# ---------------------------------------------------------------------

def test_ambiguous_resolution_never_infers():
    """Eight people share a name in production. Picking the first
    candidate's team would answer confidently about the wrong person."""
    other = AdvisorIdentity(wid=5, name="Yasir Ali", team="Downtown", company="Agency21")
    first = AdvisorIdentity(wid=4, name="Yasir Ali", team="Blue Area", company="Graana")
    assert relation_resolver.resolve_from_resolution(_ambiguous(first, other), "team") is None


def test_not_found_resolution_never_infers():
    empty = ResolvedAdvisor(status=advisor_resolver.NOT_FOUND)
    assert relation_resolver.resolve_from_resolution(empty, "team") is None
    assert relation_resolver.resolve_from_resolution(None, "team") is None


def test_resolved_resolution_infers():
    related = relation_resolver.resolve_from_resolution(_resolved(WAQAR), "team")
    assert related is not None and related.value == "Blue Area"


# ---------------------------------------------------------------------
# The naming assumption the resolver relies on
# ---------------------------------------------------------------------

def test_cached_relations_are_reachable_as_identity_attributes():
    """resolve_from_identity() reads getattr(identity, target_level).
    If a future cached relation's level name diverges from its
    AdvisorIdentity attribute, it would silently resolve to None — this
    fails loudly instead."""
    from app.llm import relations

    for spec in relations.registry.specs_for("advisor"):
        if spec.cached:
            assert hasattr(WAQAR, spec.target_level), (
                f"cached relation {spec.target_level!r} has no AdvisorIdentity attribute"
            )
