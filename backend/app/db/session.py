"""Engine and session factory. One ``Database`` instance per application (held on app.state)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings


def _sqlite_pragmas(dbapi_conn: Any, _: Any) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")  # needed for ON DELETE CASCADE
    cur.execute("PRAGMA journal_mode=WAL")  # readers don't block the ingestion writer
    cur.execute("PRAGMA busy_timeout=10000")
    cur.close()


def build_engine(settings: Settings) -> Engine:
    url = settings.database_url
    if settings.is_sqlite:
        if url.startswith("sqlite:///") and ":memory:" not in url:
            from pathlib import Path

            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 10})
        event.listen(engine, "connect", _sqlite_pragmas)
        return engine
    return create_engine(
        url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
        pool_recycle=1800,
    )


class Database:
    def __init__(self, settings: Settings):
        self.engine = build_engine(settings)
        self._factory = sessionmaker(bind=self.engine, expire_on_commit=False, autoflush=False)

    def session(self) -> Session:
        return self._factory()

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        """Session that commits on success and rolls back on any exception."""
        session = self._factory()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def ping(self) -> bool:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True

    def dispose(self) -> None:
        self.engine.dispose()
