"""Characterisation of the identity cache projection (written BEFORE M3).

Records what `AdvisorIdentity` carries and what relationship inference
can therefore resolve, captured before M3 decides between widening the
cache and reading on demand.

The field-preservation tests are the important ones. Identities are
rebuilt field-by-field in two places — `_with_score()` and the semantic
widening path in entity_extractor — so ANY new field is silently dropped
on those paths unless they are updated with it. That failure mode is
particularly nasty: inference would work for an exact name match and
return None for a fuzzy or semantic one, which reads as flakiness rather
than as a bug. These tests are written to catch it structurally rather
than to enumerate fields.
"""

import dataclasses

import pytest

from app.database.models import Advisor
from app.llm import advisor_resolver, entity_extractor, relations
from app.llm.advisor_resolver import AdvisorIdentity


@pytest.fixture()
def db(db_session):
    db_session.add(Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana",
                           bm="Kaleem Ullah", zm="Adeel Dogar", office="Gulberg BC",
                           rm="Tariq Mehmood", portfolio_lead="Sana Malik",
                           management_lead="Imran Shah"))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


# ---------------------------------------------------------------------
# What the cache carries
# ---------------------------------------------------------------------

def test_identity_carries_wid_and_name(db):
    advisor_resolver.refresh_cache(db, force=True)
    identity = advisor_resolver._cache["identities"][0]
    assert identity.wid == 1
    assert identity.name == "Waqar Haider"


def test_identity_carries_team_and_company(db):
    advisor_resolver.refresh_cache(db, force=True)
    identity = advisor_resolver._cache["identities"][0]
    assert identity.team == "Blue Area"
    assert identity.company == "Graana"


def test_cached_flags_match_what_the_projection_actually_loads(db):
    """The registry's `cached` flag is a claim about this projection. If
    the two disagree, the resolver either misses a free value or returns
    None for one it believes is present."""
    advisor_resolver.refresh_cache(db, force=True)
    identity = advisor_resolver._cache["identities"][0]

    for spec in relations.registry.specs_for("advisor"):
        if spec.cached:
            assert getattr(identity, spec.target_level, None) is not None, (
                f"{spec.target_level} is flagged cached but is not on AdvisorIdentity"
            )


# ---------------------------------------------------------------------
# Field preservation — the silent-drop hazard
# ---------------------------------------------------------------------

def test_with_score_preserves_every_field(db):
    """`_with_score` rebuilds an identity. Every field must survive, or
    inference works on exact matches and mysteriously fails on fuzzy
    ones."""
    original = AdvisorIdentity(wid=1, name="Waqar Haider", team="Blue Area", company="Graana")
    rescored = advisor_resolver._with_score(original, 0.93)

    assert rescored.score == 0.93
    for field in dataclasses.fields(AdvisorIdentity):
        if field.name == "score":
            continue
        assert getattr(rescored, field.name) == getattr(original, field.name), (
            f"_with_score dropped {field.name!r}"
        )


def test_semantic_widening_preserves_every_field(db, monkeypatch):
    """entity_extractor rebuilds identities when semantic search finds a
    name. Same hazard, different code path."""
    from app.llm import entity_linker

    monkeypatch.setattr(
        entity_linker, "semantic_candidates",
        lambda text, kind, db, **kw: [{"value": "Waqar Haider", "score": 0.78}]
        if kind == "advisor" else [],
    )
    entities = entity_extractor.extract_entities("the guy called wakar hyder", db)
    resolution = entities.get("advisor_resolution")
    assert resolution is not None and resolution.candidates

    identity = resolution.candidates[0]
    for field in dataclasses.fields(AdvisorIdentity):
        if field.name == "score":
            continue
        expected = getattr(advisor_resolver._cache["identities"][0], field.name)
        assert getattr(identity, field.name) == expected, (
            f"semantic widening dropped {field.name!r}"
        )


# ---------------------------------------------------------------------
# Inference reach, pre-M3
# ---------------------------------------------------------------------

def test_cached_relation_set(db):
    """UPDATED BY M3, as this test's pre-M3 version said it should be:
    the projection widened from (team, company) to include the three
    group levels. The invariant that matters — flags agreeing with what
    is actually loaded — is asserted by
    test_cached_flags_match_what_the_projection_actually_loads above,
    which needed no change because it derives both sides."""
    cached = {s.target_level for s in relations.registry.specs_for("advisor") if s.cached}
    assert cached == {"team", "company", "unit_head", "zonal_head", "bcm", "office"}


def test_resolver_takes_no_database_session():
    """M1's interface. An on-demand implementation would have to change
    it — recorded so that choice is visible rather than incidental."""
    import inspect

    from app.llm import relation_resolver

    params = inspect.signature(relation_resolver.resolve_from_identity).parameters
    assert list(params) == ["identity", "target_level"]
