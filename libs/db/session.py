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

DB_URL = os.getenv('LJS_DB_URL', 'postgresql+psycopg2://postgres:postgres@localhost:5432/lazyjobsearch')

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
