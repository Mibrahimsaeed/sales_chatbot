from app.database.models import Advisor
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
