"""
SQLAlchemy models for the Sales Chatbot database.

This schema follows a star schema design where `advisors` is the central
dimension table, and all other tables store specific business metrics
(performance, sales funnel, attendance, pipeline, portfolio, bookings,
calls, etc.) linked through the advisor's WID.

The schema is designed to consolidate data from multiple Google Sheets into
a normalized PostgreSQL database, enabling efficient ETL, fast querying,
and scalable chatbot responses.
"""



import enum
from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Enum, ForeignKey,
    UniqueConstraint, func
)
from sqlalchemy.orm import relationship
from app.database.session import Base


class Advisor(Base):
    __tablename__ = "advisors"

    wid = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, index=True)
    company = Column(String, index=True)
    region = Column(String)
    team = Column(String, index=True)
    office = Column(String)
    unit = Column(String)
    portfolio_lead = Column(String, index=True)
    management_lead = Column(String)
    bm = Column(String)
    zm = Column(String)
    rm = Column(String)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    sales_funnel = relationship("SalesFunnel", back_populates="advisor", uselist=False, cascade="all, delete-orphan")
    pipeline = relationship("Pipeline", back_populates="advisor", uselist=False, cascade="all, delete-orphan")
    attendance = relationship("Attendance", back_populates="advisor", uselist=False, cascade="all, delete-orphan")
    performance = relationship("Performance", back_populates="advisor", cascade="all, delete-orphan")
    portfolio = relationship("Portfolio", back_populates="advisor", uselist=False, cascade="all, delete-orphan")
    bookings = relationship("Bookings", back_populates="advisor", uselist=False, cascade="all, delete-orphan")
    calls = relationship("Calls", back_populates="advisor", uselist=False, cascade="all, delete-orphan")


class SalesFunnel(Base):
    __tablename__ = "sales_funnel"

    wid = Column(Integer, ForeignKey("advisors.wid"), primary_key=True)
    mtd_new_connect = Column(Float, default=0)
    mtd_followup_connect = Column(Float, default=0)
    system_connect = Column(Float, default=0)
    mtd_cr = Column(Float, default=0)
    mtd_new_meeting = Column(Float, default=0)
    mtd_followup_meeting = Column(Float, default=0)
    mtd_todo = Column(Float, default=0)
    mtd_booking_stored = Column(Float, default=0)
    mtd_conversion = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    advisor = relationship("Advisor", back_populates="sales_funnel")


class Pipeline(Base):
    __tablename__ = "pipeline"

    wid = Column(Integer, ForeignKey("advisors.wid"), primary_key=True)
    pipeline = Column(Float, default=0)
    overdue = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    advisor = relationship("Advisor", back_populates="pipeline")


class Attendance(Base):
    __tablename__ = "attendance"

    wid = Column(Integer, ForeignKey("advisors.wid"), primary_key=True)
    biometric_time = Column(String)
    biometric_status = Column(String)
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
    __tablename__ = "performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    wid = Column(Integer, ForeignKey("advisors.wid"), nullable=False, index=True)
    period = Column(
    Enum(
        PerformancePeriod,
        values_callable=lambda enum_cls: [e.value for e in enum_cls],
    ),
    nullable=False,
)
    target = Column(Float, default=0)
    cleared = Column(Float, default=0)
    pct = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    advisor = relationship("Advisor", back_populates="performance")

    __table_args__ = (
        UniqueConstraint("wid", "period", name="uq_performance_wid_period"),
    )


class Portfolio(Base):
    __tablename__ = "portfolio"

    wid = Column(Integer, ForeignKey("advisors.wid"), primary_key=True)
    value = Column(Float, default=0)
    returned = Column(Float, default=0)
    retention_pct = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    advisor = relationship("Advisor", back_populates="portfolio")


class Bookings(Base):
    __tablename__ = "bookings"

    wid = Column(Integer, ForeignKey("advisors.wid"), primary_key=True)
    confirmed = Column(Float, default=0)
    expected = Column(Float, default=0)
    token = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    advisor = relationship("Advisor", back_populates="bookings")


class Calls(Base):
    __tablename__ = "calls"

    wid = Column(Integer, ForeignKey("advisors.wid"), primary_key=True)
    answered_calls_mtd = Column(Float, default=0)
    answered_calls_daily = Column(Float, default=0)
    connects_mtd = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    advisor = relationship("Advisor", back_populates="calls")


class TeamTarget(Base):
    __tablename__ = "team_targets"

    team = Column(String, primary_key=True)
    target = Column(Float, default=0)
    achieved = Column(Float, default=0)
    achievement_pct = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class MorningMeetingCompliance(Base):
    __tablename__ = "morning_meeting_compliance"

    wid = Column(Integer, ForeignKey("advisors.wid"), primary_key=True)
    team = Column(String)
    zonal_head = Column(String)
    status = Column(String)
    mtd_ontime = Column(Float, default=0)
    mtd_late = Column(Float, default=0)
    mtd_not_submitted = Column(Float, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class AdvisorHistory(Base):
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
    status = Column(String)
    rows_synced = Column(Integer)
    error = Column(String)