from app.database.models import Advisor
from app.llm import ir_validator
from app.llm.ir_validator import clarification_options, confidence_breakdown, validate_ir
from app.llm.query_ir import Filter, MetricRef, QueryIR, Sort, Subject, TimeRange


def _ir(**overrides) -> QueryIR:
    base = dict(
        intent="leaderboard",
        subject_level="advisor",
        metric=MetricRef(key="mtd_cleared", confidence=0.9),
        sort=Sort(metric="mtd_cleared", direction="desc"),
        limit=10,
        overall_confidence=0.9,
    )
    base.update(overrides)
    return QueryIR(**base)


# ---- confidence_breakdown ----

def test_breakdown_full_confidence_leaderboard():
    breakdown = confidence_breakdown(_ir())
    assert breakdown["intent"] == 0.9
    assert breakdown["metric"] == 0.9
    assert breakdown["entities"] == 1.0   # no subjects to be unsure about
    assert breakdown["filters"] == 1.0    # no filters to be unsure about
    assert breakdown["time"] == 0.6       # default MTD is the ambiguous case


def test_breakdown_explicit_period_is_more_confident():
    ir = _ir(time_range=TimeRange(mode="snapshot", period="YTD"))
    breakdown = confidence_breakdown(ir)
    assert breakdown["time"] == 0.9


def test_breakdown_clarify_intent_has_zero_intent_confidence():
    ir = _ir(intent="clarify", overall_confidence=0.4)
    breakdown = confidence_breakdown(ir)
    assert breakdown["intent"] == 0.0


def test_breakdown_averages_subject_and_filter_confidence():
    ir = _ir(
        intent="comparison",
        subjects=[
            Subject(type="team", value="Blue Area", match_confidence=1.0),
            Subject(type="team", value="Downtown", match_confidence=0.8),
        ],
        filters=[Filter(field="attendance_rate", operator=">", value=90, confidence=0.7)],
    )
    breakdown = confidence_breakdown(ir)
    assert breakdown["entities"] == 0.9
    assert breakdown["filters"] == 0.7


def test_breakdown_missing_metric_floors_to_zero():
    ir = _ir(missing=["metric"])
    breakdown = confidence_breakdown(ir)
    assert breakdown["metric"] == 0.0


# ---- clarification_options ----

def test_metric_slot_returns_metric_labels():
    options = clarification_options("metric", db=None)
    assert "MTD Revenue Cleared" in options
    assert options == sorted(options)


def test_metric_low_confidence_slot_returns_metric_labels():
    options = clarification_options("metric_low_confidence:mtd_cleared", db=None)
    assert "MTD Revenue Cleared" in options


def test_team_subject_slot_returns_known_teams(db_session):
    db_session.add_all([
        Advisor(wid=1, name="A", team="Blue Area", company="Graana"),
        Advisor(wid=2, name="B", team="Downtown", company="IMARAT"),
    ])
    db_session.commit()
    options = clarification_options("subject:team:blu area", db_session)
    assert set(options) == {"Blue Area", "Downtown"}


def test_unsupported_intent_slot_has_no_options():
    assert clarification_options("unsupported_intent:trend:no history data", db=None) == []


def test_no_item_has_no_options():
    assert clarification_options(None, db=None) == []


def test_subjects_slot_has_no_enumerable_options():
    assert clarification_options("subjects", db=None) == []


# ---- semantic subject grounding fallback (Part 9) ----
# entity_linking is forced off globally by conftest's autouse fixture;
# these re-enable it locally and mock entity_linker.semantic_candidates()
# directly (the embeddings layer underneath it is covered by
# test_entity_linker.py).

def test_unresolved_team_subject_grounds_via_semantic_fallback(db_session, monkeypatch):
    db_session.add_all([
        Advisor(wid=1, name="A", team="Blue Area", company="Graana"),
        Advisor(wid=2, name="B", team="Downtown", company="IMARAT"),
    ])
    db_session.commit()
    monkeypatch.setattr(ir_validator.entity_linker.settings, "entity_linking_enabled", True)
    monkeypatch.setattr(
        ir_validator.entity_linker,
        "semantic_candidates",
        lambda text, entity_type, db, **kw: [{"value": "Blue Area", "score": 0.81}],
    )
    ir = _ir(
        intent="comparison",
        subjects=[Subject(type="team", value="the CBD zone", match_confidence=0.9)],
    )
    result = validate_ir(ir, db_session)
    assert result.ir.subjects[0].value == "Blue Area"
    assert result.ir.subjects[0].match_confidence == 0.81
    assert not any(m.startswith("subject:team:") for m in result.missing)


def test_subject_below_every_floor_still_asks_for_clarification(db_session, monkeypatch):
    db_session.add(Advisor(wid=1, name="A", team="Blue Area", company="Graana"))
    db_session.commit()
    monkeypatch.setattr(ir_validator.entity_linker.settings, "entity_linking_enabled", True)
    monkeypatch.setattr(ir_validator.entity_linker, "semantic_candidates", lambda text, entity_type, db, **kw: [])
    ir = _ir(
        intent="comparison",
        subjects=[Subject(type="team", value="total gibberish", match_confidence=0.9)],
    )
    result = validate_ir(ir, db_session)
    assert "subject:team:total gibberish" in result.missing


# ---- confidence-aware execution gate (Part 10) ----

def test_fully_grounded_ir_is_high_confidence(db_session):
    ir = _ir(overall_confidence=0.9)  # nothing missing, clears the default 0.8 high floor
    result = validate_ir(ir, db_session)
    assert result.is_valid
    assert result.confidence_level == "high"
    assert result.ir.confidence_level == "high"
    assert result.ir.ambiguity_reasons == []


def test_grounded_but_mediocre_overall_confidence_is_medium_not_high(db_session):
    # every individual field passes its own floor, but overall_confidence
    # itself falls short of the high threshold — Part 10 catches this as
    # its own slot instead of silently executing a shaky-but-technically-
    # complete IR as if it were fully confident
    ir = _ir(overall_confidence=0.6)
    result = validate_ir(ir, db_session)
    assert not result.is_valid
    assert result.confidence_level == "medium"
    assert "overall_low_confidence" in result.missing
    assert result.ir.ambiguity_reasons  # human-readable reason populated


def test_missing_metric_at_low_overall_confidence_is_low_not_medium(db_session):
    ir = _ir(metric=None, sort=Sort(metric=None, direction="desc"), overall_confidence=0.3)
    result = validate_ir(ir, db_session)
    assert not result.is_valid
    assert result.confidence_level == "low"


def test_missing_metric_at_medium_overall_confidence_is_medium(db_session):
    ir = _ir(metric=None, sort=Sort(metric=None, direction="desc"), overall_confidence=0.6)
    result = validate_ir(ir, db_session)
    assert not result.is_valid
    assert result.confidence_level == "medium"


def test_thresholds_are_configurable(db_session, monkeypatch):
    # dropping the high floor to 0.5 should let a previously-"medium"
    # 0.6-overall IR through as "high" instead
    monkeypatch.setattr(ir_validator.settings, "confidence_high_threshold", 0.5)
    ir = _ir(overall_confidence=0.6)
    result = validate_ir(ir, db_session)
    assert result.is_valid
    assert result.confidence_level == "high"


def test_ambiguity_reasons_are_human_readable_and_deduped(db_session):
    ir = _ir(metric=None, sort=Sort(metric=None, direction="desc"), overall_confidence=0.3)
    result = validate_ir(ir, db_session)
    assert result.ir.ambiguity_reasons == [ir_validator._ask_for("metric")]


def test_unsupported_intent_still_populates_confidence_level(db_session):
    ir = _ir(intent="trend")
    result = validate_ir(ir, db_session)
    assert not result.is_valid
    assert result.ir.confidence_level in ("medium", "low")
    assert result.ir.ambiguity_reasons
