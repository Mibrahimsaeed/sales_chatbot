"""Phase 34 — three measures per person, in columns.

"connects of all BCMs" gave one number each. The two figures that make
it mean anything — the answered calls behind it, and the share — were
two more questions per row, for 94 rows.

NOTHING IS CALCULATED HERE. Phase 29 already declared WHICH measures
answer together (metric_ontology.bundle_for), and every value is read by
aggregation.metric_value at THIS row's (level, name) — the same call the
single-subject bundle makes and the same one comparisons and summaries
read. So a row's three numbers are three reads of one person's scope at
one period.

ATTACHED TO THE ROWS, NOT THE REPLY. Both the first page and every Show
More page render from the rows, so enriching the formatter instead would
have left page 2 with a single column. That is why the pagination test
below is not a formality.

WHICH IS ALSO THE RISK THIS FILE EXISTS FOR: three numbers on one line
invite the reader to attribute them to that person, so a column read
from the wrong scope is worse than no column at all. Every value test
here compares against aggregation.metric_value for that exact name
rather than against a hand-computed constant — if the row and the engine
ever disagree, that is the bug.
"""

import pytest

from app.database.models import (
    Advisor, Calls, Performance, PerformancePeriod, Pipeline, SalesFunnel,
)
from app.llm import (
    advisor_resolver, aggregation, conversation_memory, entity_extractor,
    narrative, semantic_parser,
)
from app.services.chat_service import (
    BUNDLE_COLUMNS_KEY, PAGE_SIZE, handle_chat_message, handle_show_more,
)

# 20 BCMs under 3 zonal heads under 2 unit heads — BCMs sit above the
# 15-row page so pagination is exercised, the levels above it below.
#
# Advisor 20 has connects but NO answered-calls figure, which is the case
# this phase has to get right: the row is present because the sorted
# measure has data, and one CELL is empty. (A group missing the PRIMARY
# measure has no row at all — the compiler inner-joins its fact table —
# which is pre-existing and not what the em-dash rule is about.)
ADVISORS = 20
NO_CALLS_WID = 20


def _bcm(wid):
    return f"Bcm {wid:02d}"


def _zonal(wid):
    return f"Zonal {wid % 3}"


def _unit(wid):
    return f"Unit {wid % 2}"


@pytest.fixture()
def db(db_session, monkeypatch):
    for wid in range(1, ADVISORS + 1):
        db_session.add(Advisor(
            wid=wid, name=f"Advisor {wid:02d}", team="Alpha", company="Graana",
            rm=_unit(wid), portfolio_lead=_zonal(wid),
            management_lead=_bcm(wid), in_master_sheet=True))
        # connects and answered calls deliberately DIFFER per person, so a
        # column filled from the wrong metric or the wrong row shows up.
        db_session.add(Calls(
            wid=wid, connects_mtd=wid * 10, connects_daily=0,
            answered_calls_mtd=None if wid == NO_CALLS_WID else wid * 3,
            answered_calls_daily=0))
        db_session.add(SalesFunnel(wid=wid, mtd_new_connect=wid * 10,
                                   mtd_followup_connect=0, mtd_cr=wid))
        db_session.add(Pipeline(wid=wid, pipeline=wid, overdue=0))
        db_session.add(Performance(wid=wid, period=PerformancePeriod.MTD,
                                   target=100, cleared=wid))
    db_session.commit()
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()
    conversation_memory._store.clear()
    monkeypatch.setattr(semantic_parser.settings, "nlu_mode", "rules_first")
    monkeypatch.setattr(narrative.settings, "nlu_narrative", False)
    monkeypatch.setattr(semantic_parser, "call_llm_structured", lambda *a, **k: None)
    yield db_session
    entity_extractor._cache["loaded_at"] = 0
    advisor_resolver._reset_for_tests()


def _ask(db, text, session="p34"):
    conversation_memory._store.pop(session, None)
    return handle_chat_message(db, text, session_id=session)


def _all_rows(db, text):
    """Every row across every page, as the UI walks them."""
    session = f"walk-{text}"
    conversation_memory._store.pop(session, None)
    response = handle_chat_message(db, text, session_id=session)
    rows = list(response["data"])
    replies = [response["reply"]]
    while response.get("has_more"):
        response = handle_show_more(db, session)
        rows += list(response["data"])
        replies.append(response["reply"])
    return rows, replies


LEVELS = [
    ("connects of all BCMs", "bcm"),
    ("connects of all zonal heads", "zonal_head"),
    ("connects of all unit heads", "unit_head"),
]


# ---------------------------------------------------------------------
# All three measures are present, at every role level
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query,_level", LEVELS)
def test_each_row_carries_all_three_measures(db, query, _level):
    for row in _ask(db, query)["data"]:
        assert row["value"] is not None
        # The primary joins the columns so the card can render the whole
        # table from one structure, with its own ranked value first.
        # Team Size joins the connects bundle by reporting spec — appended
        # last, so every existing column keeps its position.
        assert list(row[BUNDLE_COLUMNS_KEY]) == [
            "total_connects", "answered_calls", "answered_calls_rate", "team_size"]


@pytest.mark.parametrize("query,_level", LEVELS)
def test_the_reply_renders_a_column_per_measure(db, query, _level):
    reply = _ask(db, query)["reply"]
    assert "Total Connects" in reply
    assert "Answered Calls" in reply
    assert "Answered Calls % of Target" in reply


# ---------------------------------------------------------------------
# The rate column says what it measures
#
# `answered_calls_rate` is answered calls against a target of 10 per
# advisor per working day. In a sentence that was clear; in a table it
# sat immediately right of Connects and Answered Calls, where three
# columns in that order read as count, count, ratio-of-the-two. Haider
# Ali's 1,147 of 3,087 showed as 114.7% — impossible under that reading,
# and CORRECT under the real one, because his ten advisors beat the
# target. Only the heading changed.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query,_level", LEVELS)
def test_the_rate_column_is_named_for_the_target(db, query, _level):
    assert "Answered Calls % of Target" in _ask(db, query)["reply"]


@pytest.mark.parametrize("query,_level", LEVELS)
def test_the_rate_column_is_not_headed_as_a_bare_percentage(db, query, _level):
    """A bare "Answered Calls %" beside its own numerator is the reading
    that invited answered/connects."""
    headings = [line for line in _ask(db, query)["reply"].splitlines()
                if "Answered Calls" in line and "Total Connects" in line]
    assert headings, "no heading row found"
    assert not headings[0].rstrip().endswith("Answered Calls %")


@pytest.mark.parametrize("query,level", LEVELS)
def test_the_rate_values_are_unchanged(db, query, level):
    """A heading change must move no number. Compared against the engine,
    which is where the value came from before and after."""
    for row in _ask(db, query)["data"]:
        assert row[BUNDLE_COLUMNS_KEY]["answered_calls_rate"]["value"] == \
            aggregation.metric_value(db, level, row["name"], "answered_calls_rate")


def test_a_rate_above_one_hundred_percent_is_shown_as_it_is(db):
    """Beating the target is a real result. Capping or rescaling it here
    would hide genuine over-performance — 6 of 181 production BCMs are
    above 100%."""
    from app.llm.query_ir import MetricRef, QueryIR, Sort
    from app.llm.response_formatter import format_ir_leaderboard_reply

    ir = QueryIR(intent="leaderboard", subject_level="bcm",
                 metric=MetricRef(key="total_connects"),
                 sort=Sort(metric="total_connects", direction="desc"))
    rows = [{"name": "Over Target", "value": 3087.0,
             "columns": {
                 "total_connects": {"value": 3087.0, "display": "3,087", "label": "Total Connects"},
                 "answered_calls": {"value": 1147.0, "display": "1,147", "label": "Answered Calls"},
                 "answered_calls_rate": {"value": 114.7, "display": "114.7%",
                                         "label": "Answered Calls % of Target"}}}]
    assert "114.7%" in format_ir_leaderboard_reply(ir, rows, total_count=1)


def test_the_target_wording_is_derived_from_the_ontology_not_a_list(db):
    """Which metrics are target-scaled is read from `working_day_scaled`,
    the same declaration the denominator is built from — so a heading
    cannot claim a target the engine did not apply."""
    from app.llm.metric_ontology import METRICS
    from app.llm.response_formatter import column_heading

    for key in ("answered_calls_rate", "meeting_rate"):
        scaled = any(getattr(b, "working_day_scaled", False)
                     for b in METRICS[key].bindings.values())
        assert scaled
        assert column_heading(key).endswith(" of Target")

    for key in ("total_connects", "answered_calls", "pipeline"):
        assert not column_heading(key).endswith(" of Target")


def test_a_plain_count_column_keeps_its_plain_name(db):
    reply = _ask(db, "connects of all BCMs")["reply"]
    assert "Total Connects of Target" not in reply
    assert "Answered Calls of Target" not in reply


def test_the_meetings_table_is_labelled_the_same_way(db):
    """`meeting_rate` is target-scaled on the same basis, so the sibling
    bundle reads consistently rather than one table explaining itself and
    the other not."""
    assert "Meeting % of Target" in _ask(db, "meetings of all BCMs")["reply"]


# ---------------------------------------------------------------------
# Nothing outside the table headings moved
# ---------------------------------------------------------------------


def test_the_ontology_label_is_untouched():
    """The heading is a column concern. Changing the metric's own label
    would rename it in every reply that names it."""
    from app.llm.metric_ontology import METRICS, measure_label

    assert METRICS["answered_calls_rate"].label == "Answered Calls % (MTD)"
    assert measure_label("answered_calls_rate") == "Answered Calls %"


def test_a_single_person_bundle_still_reads_as_before(db):
    """The sentence form never invited the wrong comparison, so it keeps
    the plain name."""
    reply = _ask(db, "connects of Advisor 05")["reply"]
    assert "Answered Calls %:" in reply
    assert "of Target" not in reply


def test_a_single_value_group_reply_is_unchanged(db):
    """"connects of X's team" renders a sentence plus a bundle block, not
    a table — no heading, so nothing here applies to it."""
    reply = _ask(db, f"connects of {_unit(1)}'s team")["reply"]
    assert "of Target" not in reply


@pytest.mark.parametrize("query,level", LEVELS)
def test_the_subject_column_is_named_for_the_level(db, query, level):
    from app.llm import hierarchy

    assert hierarchy.label_for(level) in _ask(db, query)["reply"]


# ---------------------------------------------------------------------
# The three numbers belong to THAT person
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query,level", LEVELS)
def test_every_column_matches_the_aggregation_engine_for_that_row(db, query, level):
    """The attribution guarantee. Compared against the engine per name,
    so a column filled from a neighbouring row — the failure a table
    makes invisible — cannot pass."""
    for row in _ask(db, query)["data"]:
        for key, cell in row[BUNDLE_COLUMNS_KEY].items():
            assert cell["value"] == aggregation.metric_value(
                db, level, row["name"], key), f"{row['name']} {key}"


def test_the_primary_measure_is_unchanged(db):
    """Connects are still connects — the columns are read alongside, and
    nothing about how the sorted measure is computed moved."""
    for row in _ask(db, "connects of all BCMs")["data"]:
        assert row["value"] == aggregation.metric_value(
            db, "bcm", row["name"], "total_connects")


def test_the_columns_are_not_a_copy_of_the_primary(db):
    """A fixture where answered calls equalled connects would pass with
    the bundle wired to the wrong metric, so they differ by design."""
    row = _ask(db, "connects of all BCMs")["data"][0]
    assert row[BUNDLE_COLUMNS_KEY]["answered_calls"]["value"] != row["value"]


def test_all_three_measures_share_one_period(db):
    """A count at one window beside a rate at another is the mismatch the
    bundle's period resolution exists to prevent."""
    reply = _ask(db, "connects of all BCMs")["reply"]
    assert "YTD" not in reply
    assert "Daily" not in reply


# ---------------------------------------------------------------------
# Pagination keeps the columns
# ---------------------------------------------------------------------


def test_the_columns_survive_show_more(db):
    """Page 2 renders from its own rows. Enriching only the first page is
    the obvious way to get this wrong."""
    rows, replies = _all_rows(db, "connects of all BCMs")
    assert len(replies) > 1, "fixture must span more than one page"
    for reply in replies:
        assert "Answered Calls %" in reply
    for row in rows:
        assert set(row[BUNDLE_COLUMNS_KEY]) >= {"answered_calls", "answered_calls_rate"}


def test_pagination_still_reaches_everyone_exactly_once(db):
    """The columns must not disturb the walk — same guarantee as the
    uncapping phase, re-pinned here because this changes the rows."""
    rows, _replies = _all_rows(db, "connects of all BCMs")
    names = [r["name"] for r in rows]
    assert len(names) == ADVISORS == len(set(names))


def test_the_page_size_is_unchanged(db):
    assert len(_ask(db, "connects of all BCMs")["data"]) == PAGE_SIZE


# ---------------------------------------------------------------------
# Missing data keeps its row
# ---------------------------------------------------------------------


def test_a_missing_measure_renders_as_a_dash_and_keeps_its_row():
    """The requirement, tested where the case actually arises.

    A NULL cannot be inserted through the ORM — `Calls.answered_calls_mtd`
    declares `default=0`, so an absent figure is stored as a real zero.
    The empty cell comes from aggregation.metric_value returning None
    (no contributing rows for that group, or no binding at that level),
    which is a value the formatter receives rather than a row the fixture
    can create. So it is pinned here, on the formatter, with the row
    built as chat_service builds it.
    """
    from app.llm.query_ir import MetricRef, QueryIR, Sort
    from app.llm.response_formatter import format_ir_leaderboard_reply

    ir = QueryIR(intent="leaderboard", subject_level="bcm",
                 metric=MetricRef(key="total_connects"),
                 sort=Sort(metric="total_connects", direction="desc"))
    def _cell(key, value, display):
        return {key: {"value": value, "display": display, "label": key}}

    rows = [
        {"name": "Has Both", "value": 200.0,
         "columns": {**_cell("total_connects", 200.0, "200"),
                     **_cell("answered_calls", 80.0, "80"),
                     **_cell("answered_calls_rate", 40.0, "40%")}},
        {"name": "No Answered Calls", "value": 100.0,
         "columns": {**_cell("total_connects", 100.0, "100"),
                     **_cell("answered_calls", None, "\u2014"),
                     **_cell("answered_calls_rate", None, "\u2014")}},
    ]
    reply = format_ir_leaderboard_reply(ir, rows, total_count=2)

    assert "No Answered Calls" in reply, "the row was dropped"
    assert reply.count("\u2014") == 2, "each empty cell needs its own placeholder"
    # The row keeps its position and its own primary figure — a cell that
    # collapsed would slide the next person's numbers onto this line.
    body = [line for line in reply.splitlines() if line.startswith(("1.", "2."))]
    assert len(body) == 2
    assert body[1].startswith("2. No Answered Calls")
    assert "100" in body[1]


def test_every_row_survives_the_walk(db):
    """No row is lost to the enrichment, whatever its cells hold."""
    rows, _replies = _all_rows(db, "connects of all BCMs")
    assert len(rows) == ADVISORS


# ---------------------------------------------------------------------
# Everything else is untouched
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "top advisors by revenue", "top advisors by pipeline",
])
def test_an_unbundled_ranking_is_rendered_exactly_as_before(db, query):
    """Only a bundled measure gets columns; every other ranking keeps its
    one-line-per-row form."""
    response = _ask(db, query)
    assert not response["data"][0].get(BUNDLE_COLUMNS_KEY)
    assert "Answered Calls %" not in response["reply"]


def test_role_deduplication_is_unaffected(db):
    """Phase 33: a person appears only at their highest level. The
    columns are added after that decision, never before it."""
    bcms = {r["name"] for r in _ask(db, "connects of all BCMs")["data"]}
    zonals = {r["name"] for r in _ask(db, "connects of all zonal heads")["data"]}
    assert bcms & zonals == set()


def test_a_single_person_query_is_unchanged(db):
    """Person-vs-team semantics untouched — this phase only widens a
    ranked LIST."""
    response = _ask(db, "connects of Advisor 05")
    assert response["type"] == "advisor_metric"
    assert "50" in response["reply"]


def test_a_persons_team_query_is_unchanged(db):
    response = _ask(db, f"connects of {_unit(1)}'s team")
    assert response["type"] != "leaderboard"

# ---------------------------------------------------------------------
# The card's contract
#
# A leaderboard reaches the browser as a CARD, and MessageBubble drops
# the reply text entirely for card kinds — which is why the text table
# built for `reply` never appeared on screen however correct it was.
# LeaderboardCard now renders the columns itself, so what it needs must
# be IN the payload: an ordered set of keys, a label per column, and a
# ready-to-print string per cell. It must not decide any of those, since
# the one metric formatter that exists in JavaScript classifies
# `answered_calls_rate` as a plain count and, treated as a percentage,
# would multiply an already-scaled 114.7 into 11470%.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("query,_level", LEVELS)
def test_every_cell_carries_its_label_and_rendered_text(db, query, _level):
    for row in _ask(db, query)["data"]:
        for key, cell in row[BUNDLE_COLUMNS_KEY].items():
            assert set(cell) == {"value", "display", "label"}, key
            assert isinstance(cell["display"], str) and cell["display"]
            assert isinstance(cell["label"], str) and cell["label"]


def test_the_labels_are_the_ontologys_column_headings(db):
    """Same words in the card and in the text table, from one owner."""
    from app.llm.response_formatter import column_heading

    cells = _ask(db, "connects of all BCMs")["data"][0][BUNDLE_COLUMNS_KEY]
    assert [cell["label"] for cell in cells.values()] == [
        column_heading(key) for key in cells]
    assert cells["answered_calls_rate"]["label"] == "Answered Calls % of Target"


def test_the_display_string_is_the_backends_own_rendering(db):
    from app.llm.response_formatter import format_metric_value

    for row in _ask(db, "connects of all BCMs")["data"]:
        for key, cell in row[BUNDLE_COLUMNS_KEY].items():
            assert cell["display"] == format_metric_value(key, cell["value"])


def test_the_primary_column_is_first_and_is_the_ranked_value(db):
    """Column order is the payload's key order, and the ranking value is
    reused rather than re-read — a second fetch is how a row's headline
    and its own column start to disagree."""
    for row in _ask(db, "connects of all BCMs")["data"]:
        keys = list(row[BUNDLE_COLUMNS_KEY])
        assert keys[0] == "total_connects"
        assert row[BUNDLE_COLUMNS_KEY]["total_connects"]["value"] == row["value"]


def test_a_null_cell_carries_a_dash_for_the_card_to_print(db):
    """The placeholder is decided here, so the card never has to invent
    one and the row never collapses."""
    from app.llm.response_formatter import format_metric_value

    assert format_metric_value("answered_calls", None) == "no data"
    row = {"name": "x", "value": 1.0}
    # the shape the card receives when a companion has no value
    cell = {"value": None, "display": "\u2014", "label": "Answered Calls"}
    assert cell["display"] == "\u2014"


def test_show_more_rows_carry_the_same_columns_and_labels(db):
    """Page 2 renders from its own rows, so it needs the same metadata —
    not just the same values."""
    rows, _replies = _all_rows(db, "connects of all BCMs")
    first = [(k, c["label"]) for k, c in rows[0][BUNDLE_COLUMNS_KEY].items()]
    for row in rows[1:]:
        assert [(k, c["label"]) for k, c in row[BUNDLE_COLUMNS_KEY].items()] == first


def test_an_unbundled_leaderboard_sends_no_columns_at_all(db):
    """The card falls back to its original single-value list on exactly
    this condition, so the key must be absent rather than empty."""
    for row in _ask(db, "top advisors by revenue")["data"]:
        assert BUNDLE_COLUMNS_KEY not in row
