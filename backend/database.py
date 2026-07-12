"""
Database Layer for the Multi-Tenant Chatbot Framework

SQLAlchemy engine/session over the configured DATABASE_URL (SQLite by default).
The DB is the source of truth for CLIENT METADATA (persona, domain, branding,
WhatsApp creds). Vector data continues to live in per-client FAISS collections.
"""

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

from config import get_settings
from logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# SQLite needs check_same_thread=False when used across FastAPI's threadpool.
_connect_args = (
    {"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {}
)

engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _add_missing_columns() -> None:
    """Idempotent lightweight migration: add columns introduced after a table was
    first created (SQLAlchemy create_all adds tables but never new columns).

    Keeps existing SQLite data (clients/interactions/etc.) instead of forcing a wipe.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    # (table, column, DDL type) additions to backfill on old databases.
    additions = [
        ("clients", "owner_id", "INTEGER"),
        ("users", "role", "VARCHAR"),
        ("users", "client_slug", "VARCHAR"),
    ]
    with engine.begin() as conn:
        for table, column, coltype in additions:
            if table not in existing_tables:
                continue  # create_all will make it fresh with all columns
            cols = {c["name"] for c in inspector.get_columns(table)}
            if column not in cols:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}"))
                logger.info(f"Migrated: added {table}.{column}")


def init_db() -> None:
    """Create all tables + backfill new columns. Safe to call on every startup."""
    # Import models so they register on Base.metadata before create_all.
    import db_models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
    logger.info("Database initialized (tables ensured)")
