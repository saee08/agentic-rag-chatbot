"""Keyword (full-text) index for hybrid retrieval; machine-readable error codes on messages.

* PostgreSQL: GIN index on to_tsvector('english', text).
* SQLite: FTS5 external-content table kept in sync by triggers (fires for FK cascades too).

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-09
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

_SQLITE_UP = [
    "CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='rowid', "
    "tokenize='porter unicode61')",
    "CREATE TRIGGER chunks_fts_ai AFTER INSERT ON chunks BEGIN "
    "INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text); END",
    "CREATE TRIGGER chunks_fts_ad AFTER DELETE ON chunks BEGIN "
    "INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text); END",
    "CREATE TRIGGER chunks_fts_au AFTER UPDATE OF text ON chunks BEGIN "
    "INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text); "
    "INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text); END",
    "INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')",  # index chunks that already exist
]
_SQLITE_DOWN = [
    "DROP TRIGGER IF EXISTS chunks_fts_au",
    "DROP TRIGGER IF EXISTS chunks_fts_ad",
    "DROP TRIGGER IF EXISTS chunks_fts_ai",
    "DROP TABLE IF EXISTS chunks_fts",
]


def upgrade() -> None:
    dialect = op.get_bind().dialect.name
    with op.batch_alter_table("messages") as batch:
        batch.add_column(sa.Column("error_code", sa.String(50)))
    if dialect == "postgresql":
        op.execute("CREATE INDEX ix_chunks_text_fts ON chunks USING gin (to_tsvector('english', text))")
    elif dialect == "sqlite":
        for statement in _SQLITE_UP:
            op.execute(statement)


def downgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_chunks_text_fts")
    elif dialect == "sqlite":
        for statement in _SQLITE_DOWN:
            op.execute(statement)
    with op.batch_alter_table("messages") as batch:
        batch.drop_column("error_code")
