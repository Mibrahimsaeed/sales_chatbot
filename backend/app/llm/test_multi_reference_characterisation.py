"""Characterisation of multi-reference behaviour (written BEFORE M6).

Records what happens today when one query contains TWO relationship
references. Three distinct defects are captured, because M6 fixes three
different things and each needs its own before/after:

1. SPAN BLEED. The parser finds both references, but the second one's
   source span reaches back across the first ("Haider's team with Sana
   Tariq"), so the two references do not describe two independent people.

2. ONE WINNER. Identity resolution runs once over the whole message and
   returns a single advisor — the LAST name span wins — so both
   references resolve to the same person, or to the wrong one.

3. EXPLICIT BLOCKS INFERRED. M1's rule "an entity named outright wins"
   was written for single-reference queries, where it is right. In a
   comparison it silently drops one side: "compare X's team with
   Downtown" keeps Downtown and never binds X's team, leaving one target
   where two were asked for.

The one case that already works ("compare X's team with Graana", where
the two sides land on DIFFERENT levels) is recorded too — M6 must not
break it.
"""

import pytest

from app.core.config import settings
from app.database.models import Advisor
from app.llm import advisor_resolver, entity_extractor, reference_parser


@pytest.fixture()
def db(db_session):
    db_session.add(Advisor(wid=1, name="Waqar Haider", team="Blue Area", company="Graana"))
    db_session.add(Advisor(wid=2, name="Sana Tariq", team="Downtown", company="Agency21"))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


@pytest.fixture(autouse=True)
def inference_on(monkeypatch):
    monkeypatch.setattr(settings, "relation_inference_enabled", True)
    monkeypatch.setattr(settings, "relation_inference_levels", "team,company")


# ---------------------------------------------------------------------
# What already works and must keep working
# ---------------------------------------------------------------------

def test_a_single_reference_resolves(db):
    entities = entity_extractor.extract_entities("How is Waqar Haider's team doing", db)
    assert entities["team"] == "Blue Area"


def test_the_parser_finds_both_references():
    """Detection was never the problem."""
    references = reference_parser.parse("Compare Waqar Haider's team with Sana Tariq's team")
    assert [r.target_level for r in references] == ["team", "team"]


def test_cross_level_comparison_already_works(db):
    """The two sides land on different levels, so neither overwrites the
    other and both survive. M6 must not regress this."""
    entities = entity_extractor.extract_entities("Compare Waqar Haider's team with Graana", db)
    assert entities["teams"] == ["Blue Area"]
    assert entities["companies"] == ["Graana"]


# ---------------------------------------------------------------------
# RETIRED (M6 landed).
#
# This file also recorded four defects from the failing side: the second
# source span bled across the first, whole-message identity resolution
# kept only the last name, an explicitly named group suppressed the
# inferred one, and an in-message pronoun bound nothing. All four were
# written to fail once M6 fixed them, and all four did.
#
# They now live in test_m6_multi_reference.py, asserted from the other
# direction. Keeping both would mean asserting the defects still exist.
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# The comparison machinery that M6 will feed
# ---------------------------------------------------------------------

def test_two_values_at_one_level_already_drive_a_comparison(db):
    """The planner needs no teaching about pairs — it reads the plural
    entity keys. M6's job is to PUT two values there."""
    from app.llm.query_planner import score_intents

    entities = {"teams": ["Blue Area", "Downtown"], "team": "Blue Area"}
    _ctx, candidates = score_intents("compare blue area with downtown", entities)
    assert candidates[0].intent == "comparison"


def test_comparison_service_accepts_an_advisor_as_a_target():
    """`advisor` maps to Advisor.name in LEVEL_COLUMNS, so an advisor is
    already a valid comparison target — no IR or service change is needed
    for "how does X compare to his team". Only the planner never builds
    such a target."""
    from app.llm import hierarchy

    assert hierarchy.column_for("advisor") is not None
