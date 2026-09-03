"""A person named with no level is asked about at their HIGHEST role.

"connects of Naina" names a person, not a job. If Naina is a Unit Head
who also has an advisor row — which most managers do, because the
hierarchy table is advisor-centric — answering from that row is a true
statement about one row and a false one about her: a scope of one where
the question meant her whole organisation.

WHY THIS IS NOT THE REWRITE PHASE 11 REMOVED. That phase switched off the
rewrites that overruled a level the query STATED. This fires only when
the query stated NONE, which `entities["level_word"]` decides from the
user's own words. So:

    "connects of Naina"                 -> her highest role
    "connects of Naina as an advisor"   -> advisor, untouched
    "connects of unit head Naina"       -> unit_head, untouched

Nothing the user said is contradicted; what was left unsaid is settled
from the hierarchy, which is what EntityRef's own docstring says grounding
exists to do.

The ranking is not new: highest_level_of reads the levels a name grounds
at and picks the senior-most by hierarchy.CHAIN — the same answer
nlu_pipeline._authoritative_role gives on the rule-based path.
"""

import pytest

from app.database.models import Advisor, SalesFunnel
from app.llm import conversation_memory, entity_extractor, semantic_parser

# One person per shape. Each manager ALSO has an advisor row, which is the
# whole difficulty: the junior reading is always available and always
# looks like a real answer.
# NAMES DELIBERATELY CONTAIN NO LEVEL WORD. "Naina Shah" and "Bilal Anwar" read
# as a stated level to detect_level — "unit" and "bcm" are its keywords —
# so the guard declined and the promotion never ran. A real hazard for
# anyone actually named that, and the wrong thing to be testing here.
PEOPLE = [
    # name,          rm (unit head),  portfolio_lead,   management_lead
    ("Sana Iqbal",   "Naina Shah",    "Zara Qureshi",   "Bilal Anwar"),
    ("Bilal Anwar",  "Naina Shah",    "Zara Qureshi",   "Bilal Anwar"),
    ("Zara Qureshi", "Naina Shah",    "Zara Qureshi",   "Zara Qureshi"),
    ("Naina Shah",   "Naina Shah",    "Naina Shah",     "Naina Shah"),
]


@pytest.fixture()
def org(db_session, monkeypatch):
    for wid, (name, rm, pl, ml) in enumerate(PEOPLE, start=1):
        db_session.add(Advisor(wid=wid, name=name, team="Alpha", company="Graana",
                               rm=rm, portfolio_lead=pl, management_lead=ml,
                               in_master_sheet=True))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=wid, mtd_followup_connect=0))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "llm_first")
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    conversation_memory._store.clear()


def _model_says(monkeypatch, name, level):
    """The parse the model returns. Every case below hands it the JUNIOR
    reading — an advisor — because that is what it returns for a bare name
    and what the resolution has to correct."""
    payload = {
        "intent": "filtered_list", "operation": "group_metric",
        "subject_level": level,
        "subjects": [{"type": level, "value": name, "match_confidence": 1.0}],
        "metric": {"key": "total_connects", "confidence": 0.9},
        "metrics": [], "filters": [], "filter_tree": None,
        "time_range": {"mode": "snapshot", "period": "MTD",
                       "compare_to": None, "confidence": 0.9},
        "sort": {"metric": "total_connects", "direction": "desc"},
        "limit": None, "group_by": None,
        "target_level": None, "subject_of": None, "relation": "subtree",
        "overall_confidence": 0.95, "intent_confidence": 0.95,
    }
    monkeypatch.setattr(semantic_parser, "call_llm_structured",
                        lambda p, s, schema_name=None: payload)


def _resolve(db, text, name, level="advisor"):
    entities = entity_extractor.extract_entities(text.lower(), db)
    interpretation = semantic_parser.interpret(text.lower(), entities, db,
                                               session_id=None)
    ir = interpretation.ir
    return ir.subjects[0].type, ir.subject_level


# ---------------------------------------------------------------------
# 1-4: an unstated level resolves to the person's highest role
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("Sana Iqbal", "advisor"),
    ("Bilal Anwar", "bcm"),
    ("Zara Qureshi", "zonal_head"),
    ("Naina Shah", "unit_head"),
])
def test_a_bare_name_resolves_to_its_highest_role(name, expected, org, monkeypatch):
    """The model returns `advisor` for all four — the junior reading is
    always available. Only the hierarchy separates them."""
    _model_says(monkeypatch, name, "advisor")

    subject_type, subject_level = _resolve(org, f"connects of {name}", name)

    assert subject_type == expected
    assert subject_level == expected, "both fields move together, or the "\
        "query filters at one level and groups at another"


def test_a_person_with_only_an_advisor_role_is_untouched(org, monkeypatch):
    """Nothing to promote to, so nothing happens — and no trace of a
    decision that was not made."""
    _model_says(monkeypatch, "Sana Iqbal", "advisor")
    assert _resolve(org, "connects of Only Advisor", "Sana Iqbal") == \
        ("advisor", "advisor")


# ---------------------------------------------------------------------
# 5-6: an explicitly stated level is respected
# ---------------------------------------------------------------------

def test_as_an_advisor_stays_an_advisor(org, monkeypatch):
    """THE PHASE 11 BOUNDARY. The user named the level, so the senior
    reading must not be substituted — a manager's own record has to stay
    reachable."""
    _model_says(monkeypatch, "Naina Shah", "advisor")

    subject_type, subject_level = _resolve(
        org, "connects of Naina Shah as an advisor", "Naina Shah")

    assert subject_type == "advisor"
    assert subject_level == "advisor"


def test_a_stated_unit_head_stays_a_unit_head(org, monkeypatch):
    _model_says(monkeypatch, "Naina Shah", "unit_head")

    assert _resolve(org, "connects of unit head Uma Unit", "Naina Shah") == \
        ("unit_head", "unit_head")


def test_a_stated_junior_level_is_not_promoted(org, monkeypatch):
    """"bcm Zara Zonal" names a level. Even though she is a Zonal Head,
    the query asked about the BCM scope and gets it."""
    _model_says(monkeypatch, "Zara Qureshi", "bcm")

    assert _resolve(org, "connects of bcm Zara Zonal", "Zara Qureshi")[0] == "bcm"


# ---------------------------------------------------------------------
# 7-8: nothing else moves
# ---------------------------------------------------------------------

def test_a_team_subject_is_never_promoted(org, monkeypatch):
    """A team is not a person and has no role to be senior to."""
    _model_says(monkeypatch, "Alpha", "team")

    assert _resolve(org, "connects of Alpha", "Alpha") == ("team", "team")


def test_only_ever_promotes_upward(org, monkeypatch):
    """A Unit Head read AS a unit head stays there; the rule can move a
    subject up the chain and never down."""
    _model_says(monkeypatch, "Naina Shah", "unit_head")

    assert _resolve(org, "connects of Uma Unit", "Naina Shah") == \
        ("unit_head", "unit_head")


def test_several_subjects_are_left_alone(org, monkeypatch):
    """A comparison names two people and is not this rule's business."""
    from app.llm.query_ir import MetricRef, QueryIR, Sort, Subject

    ir = QueryIR(intent="comparison", operation="comparison",
                 subject_level="advisor",
                 subjects=[Subject(type="advisor", value="Naina Shah"),
                           Subject(type="advisor", value="Zara Qureshi")],
                 metric=MetricRef(key="total_connects"),
                 sort=Sort(metric="total_connects"))
    entities = entity_extractor.extract_entities("compare uma unit and zara zonal", org)

    semantic_parser._resolve_unstated_person_level(ir, entities, org)

    assert [s.type for s in ir.subjects] == ["advisor", "advisor"]


def test_the_promotion_matches_the_rule_based_path(org):
    """One answer about who somebody is, whichever path asked. If these
    diverged, a name would mean different things depending on whether the
    model was reachable."""
    from app.llm.hierarchy_grounding import highest_level_of
    from app.llm.nlu_pipeline import _authoritative_role

    for name, expected in (("Bilal Anwar", "bcm"), ("Zara Qureshi", "zonal_head"),
                           ("Naina Shah", "unit_head")):
        highest = highest_level_of(name, org)
        assert highest == expected, name
        # the rule-based path, given the same levels, promotes to the same
        assert _authoritative_role("advisor", [highest]) == "advisor", \
            "a STATED advisor is never promoted, on either path"
        assert _authoritative_role("bcm", [highest]) == (
            highest if highest in ("unit_head", "zonal_head") else "bcm"), name
