"""
Star schema: `advisors` is the dimension table. Every other table below is a
sibling fact table joined to it by `wid` — none of them depend on each other.

One table per BUSINESS ENTITY, not one table per Google Sheet tab. Several
tabs (MasterSheet, Data, PC Meeting, CCMC DATA MTD) report the same connect/
meeting numbers from different views — they all map into `sales_funnel`.
"""

import enum
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Enum, ForeignKey,
    UniqueConstraint, Boolean, func
)
from sqlalchemy.orm import relationship
from app.database.session import Base


class Advisor(Base):
    """Dimension table. Sourced from MasterSheet + CCMC DATA MTD (org columns)."""
    __tablename__ = "advisors"

    wid = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, index=True)
    company = Column(String, index=True)
    region = Column(String)
    team = Column(String, index=True)
    office = Column(String)
    unit = Column(String)                      # from "1 Unit" tab
    portfolio_lead = Column(String, index=True)
    management_lead = Column(String)
    bm = Column(String)
    zm = Column(String)
    rm = Column(String)
    # True only for a WID actually present in the MasterSheet tab — every
    # other sheet (CCMC DATA MTD, biometric, login report, etc.) can
    # introduce a WID that was never onboarded, and those rows used to
    # leak into every leaderboard/summary/lookup right alongside real
    # advisors. Defaults True so ORM-constructed rows that don't set this
    # explicitly (tests, fixtures) stay queryable; etl/transform.py always
    # sets it explicitly (True for MasterSheet rows, False otherwise) for
    # real synced data, so the default only matters off the ETL path.
    in_master_sheet = Column(Boolean, nullable=False, default=True, server_default="true")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    sales_funnel = relationship("SalesFunnel", back_populates="advisor", uselist=False, cascade="all, delete-orphan")
    pipeline = relationship("Pipeline", back_populates="advisor", uselist=False, cascade="all, delete-orphan")
    attendance = relationship("Attendance", back_populates="advisor", uselist=False, cascade="all, delete-orphan")
    performance = relationship("Performance", back_populates="advisor", cascade="all, delete-orphan")
    portfolio = relationship("Portfolio", back_populates="advisor", uselist=False, cascade="all, delete-orphan")
    bookings = relationship("Bookings", back_populates="advisor", uselist=False, cascade="all, delete-orphan")
    calls = relationship("Calls", back_populates="advisor", uselist=False, cascade="all, delete-orphan")


class SalesFunnel(Base):
    """MTD connects/meetings/conversion. Consolidates MasterSheet, Data,
    Connect Session, PC Meeting, and CCMC DATA MTD — they're the same metrics."""
    __tablename__ = "sales_funnel"

    wid = Column(Integer, ForeignKey("advisors.wid"), primary_key=True)
    mtd_new_connect = Column(Float, default=0)
    mtd_followup_connect = Column(Float, default=0)
    system_connect = Column(Float, default=0)       # "Total Connect Through System" — data-quality cross-check
    mtd_cr = Column(Float, default=0)
    mtd_new_meeting = Column(Float, default=0)
    mtd_followup_meeting = Column(Float, default=0)
    mtd_todo = Column(Float, default=0)
    mtd_booking_stored = Column(Float, default=0)
    mtd_conversion = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    advisor = relationship("Advisor", back_populates="sales_funnel")


class Pipeline(Base):
    """From P1 & Overdue. `period` lets YTD P1 & Overdue share this table later
    instead of spawning a parallel one."""
    __tablename__ = "pipeline"

    wid = Column(Integer, ForeignKey("advisors.wid"), primary_key=True)
    pipeline = Column(Float, default=0)
    overdue = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    advisor = relationship("Advisor", back_populates="pipeline")


class Attendance(Base):
    """Biometric + login combined — chat questions about "did X show up" need both."""
    __tablename__ = "attendance"

    wid = Column(Integer, ForeignKey("advisors.wid"), primary_key=True)
    biometric_time = Column(String)
    biometric_status = Column(String)                # 'On Time' | 'Late' | 'Not Marked'
    biometric_mtd_ontime = Column(Float, default=0)
    biometric_mtd_late = Column(Float, default=0)
    biometric_mtd_not_marked = Column(Float, default=0)
    login_time = Column(String)
    login_status = Column(String)
    login_mtd_ontime = Column(Float, default=0)
    login_mtd_late = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    advisor = relationship("Advisor", back_populates="attendance")


class PerformancePeriod(str, enum.Enum):
    MTD = "MTD"
    YTD = "YTD"
    THREE_M = "3M"


class Performance(Base):
    """One row per advisor per period, instead of mtd_target/ytd_target/three_m_target
    as parallel columns. Adding a new period later is a data change, not a migration."""
    __tablename__ = "performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    wid = Column(Integer, ForeignKey("advisors.wid"), nullable=False, index=True)
    period = Column(
    Enum(
        PerformancePeriod,
        values_callable=lambda enum_cls: [e.value for e in enum_cls]
    ),
    nullable=False
)
    target = Column(Float, default=0)
    cleared = Column(Float, default=0)
    pct = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    advisor = relationship("Advisor", back_populates="performance")

    __table_args__ = (UniqueConstraint("wid", "period", name="uq_performance_wid_period"),)


class Portfolio(Base):
    __tablename__ = "portfolio"

    wid = Column(Integer, ForeignKey("advisors.wid"), primary_key=True)
    value = Column(Float, default=0)
    returned = Column(Float, default=0)
    retention_pct = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    advisor = relationship("Advisor", back_populates="portfolio")


class Bookings(Base):
    """From NPR tab."""
    __tablename__ = "bookings"

    wid = Column(Integer, ForeignKey("advisors.wid"), primary_key=True)
    confirmed = Column(Float, default=0)
    expected = Column(Float, default=0)
    token = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    advisor = relationship("Advisor", back_populates="bookings")


class Calls(Base):
    """From Answered Calls tab."""
    __tablename__ = "calls"

    wid = Column(Integer, ForeignKey("advisors.wid"), primary_key=True)
    answered_calls_mtd = Column(Float, default=0)
    answered_calls_daily = Column(Float, default=0)
    connects_mtd = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    advisor = relationship("Advisor", back_populates="calls")


class TeamTarget(Base):
    """Sourced DIRECTLY from the Target Achievement tab — genuine team-level
    source data, not an aggregate rolled up from `advisors`. No FK to advisors:
    joined by team name at query time."""
    __tablename__ = "team_targets"

    team = Column(String, primary_key=True)
    target = Column(Float, default=0)
    achieved = Column(Float, default=0)
    achievement_pct = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class MorningMeetingCompliance(Base):
    """Optional — only add if the chatbot needs to answer 'who missed
    morning meeting'. Sourced from Morning Meeting / MM Data tabs."""
    __tablename__ = "morning_meeting_compliance"

    wid = Column(Integer, ForeignKey("advisors.wid"), primary_key=True)
    team = Column(String)
    zonal_head = Column(String)
    status = Column(String)                # 'On Time' | 'Late' | 'Not Submitted'
    mtd_ontime = Column(Float, default=0)
    mtd_late = Column(Float, default=0)
    mtd_not_submitted = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class AdvisorHistory(Base):
    """Append-only daily snapshots for trend questions later
    ("how has X done this week"). Not part of the star's current-state core."""
    __tablename__ = "advisor_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    wid = Column(Integer, index=True, nullable=False)
    snapshot_at = Column(DateTime, server_default=func.now(), index=True)
    mtd_cleared = Column(Float)
    mtd_target = Column(Float)
    connects = Column(Float)
    meetings = Column(Float)
    overdue = Column(Float)


class SyncLog(Base):
    __tablename__ = "sync_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(DateTime, nullable=False)
    finished_at = Column(DateTime)
    status = Column(String)              # 'running' | 'success' | 'failed'
    rows_synced = Column(Integer)
    error = Column(String)


class ChatLog(Base):
    """One row per chat message. This is how you'd measure classification
    quality in production: what fraction of queries hit low confidence,
    how often the LLM fallback fires, which intents get 'unknown' most —
    all directly queryable instead of guessed at.

    `resolved_ir` stores the full QueryIR (as JSON text) that was actually
    compiled and run for this message, when the query went through the
    new IR pipeline (nlu_pipeline.py Resolution.kind == "ir"). This is
    what makes a production query failure debuggable after the fact —
    Part 6 of the redesign brief — instead of only having the final
    intent label and a confidence float with no way to see which filters,
    subjects, or metric the parser actually produced.

    `confidence_metadata` (Part 10) stores the JSON-serialized per-field
    confidence breakdown (intent/metric/entities/filters/time) plus
    confidence_level and ambiguity_reasons for that same IR — its own
    column, not just nested inside resolved_ir, so "how often do we
    reject for low confidence" / "which dimension is the weak one most
    often" are queryable without parsing resolved_ir's JSON on every row.
    Null for shortcut/plan-kind resolutions, same as resolved_ir."""
    __tablename__ = "chat_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    user_message = Column(String, nullable=False)
    detected_intent = Column(String)
    confidence = Column(Float)
    used_llm_fallback = Column(Boolean, default=False)
    response_type = Column(String)
    resolved_ir = Column(String)          # QueryIR.model_dump_json(), null for shortcut/plan-kind resolutions
    confidence_metadata = Column(String)  # JSON: {intent, metric, entities, filters, time, level, ambiguity_reasons}
    created_at = Column(DateTime, server_default=func.now())




    


