"""Database session bootstrap.

Reads DB URL from environment variable LJS_DB_URL or defaults to a local
Postgres placeholder. In real usage ensure the DB/pgvector extensions exist.
"""
from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Resolve database URL with multiple fallbacks:
# 1. LJS_DB_URL (project-specific override)
# 2. DATABASE_URL (12-factor style used in docker-compose)
# 3. Sensible in-cluster default pointing at the 'postgres' service
DB_URL = (
    os.getenv('LJS_DB_URL')
    or os.getenv('DATABASE_URL')
    or 'postgresql+psycopg2://ljs_user:ljs_password@postgres:5432/lazyjobsearch'
)

# Future: pool sizing & echo controlled via config.
engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)

@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
