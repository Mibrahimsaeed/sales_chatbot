from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

engine = create_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

# Phase 7: capture every statement executed during a traced chat request.
# Registered here (the one place the engine is created) rather than at a
# call site, so SQL from ANY layer — compiler, advisor lookup, hierarchy
# service — lands in the trace. No-ops entirely outside a chat request.
from app.core.tracing import install_sql_capture  # noqa: E402  (circular-import guard: needs `engine`)

install_sql_capture(engine)