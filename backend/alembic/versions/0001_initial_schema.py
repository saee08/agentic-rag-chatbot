"""Initial schema: users, tokens, documents, chunks (+vectors), jobs, conversations, messages.

The vector column dimension is taken from EMBEDDING_DIM at migration time. On PostgreSQL the
pgvector extension is enabled and an HNSW cosine index is created for chunk search.

Revision ID: 0001
Revises:
Create Date: 2026-11-03
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.core.config import get_settings
from app.db.types import GUID, EmbeddingVector

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _ts(name: str, nullable: bool = False) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=nullable)


def upgrade() -> None:
    is_pg = op.get_bind().dialect.name == "postgresql"
    dim = get_settings().embedding_dim
    if is_pg:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        _ts("created_at"),
    )
    op.create_table(
        "auth_tokens",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("user_id", GUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
        _ts("expires_at"),
        _ts("revoked_at", nullable=True),
        _ts("created_at"),
    )
    op.create_index("ix_auth_tokens_user_id", "auth_tokens", ["user_id"])

    op.create_table(
        "documents",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("owner_id", GUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("file_type", sa.String(10), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("storage_key", sa.String(255), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("page_count", sa.Integer()),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("embedding_model", sa.String(200)),
        _ts("created_at"),
        _ts("updated_at"),
        sa.UniqueConstraint("owner_id", "content_sha256", name="uq_documents_owner_hash"),
    )
    op.create_index("ix_documents_owner_created", "documents", ["owner_id", "created_at"])

    op.create_table(
        "chunks",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("document_id", GUID(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("owner_id", GUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer()),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("word_count", sa.Integer(), nullable=False),
        sa.Column("embedding", EmbeddingVector(dim), nullable=False),
        sa.UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_owner", "chunks", ["owner_id"])
    if is_pg:
        op.execute(
            "CREATE INDEX ix_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
        )

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("document_id", GUID(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text()),
        _ts("started_at", nullable=True),
        _ts("finished_at", nullable=True),
        sa.Column("timings_ms", sa.JSON()),
        _ts("created_at"),
    )
    op.create_index("ix_ingestion_jobs_document_id", "ingestion_jobs", ["document_id"])
    op.create_index("ix_ingestion_jobs_status", "ingestion_jobs", ["status"])

    op.create_table(
        "conversations",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("owner_id", GUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        _ts("created_at"),
        _ts("updated_at"),
    )
    op.create_index("ix_conversations_owner_updated", "conversations", ["owner_id", "updated_at"])

    op.create_table(
        "messages",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column(
            "conversation_id", GUID(), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(30)),
        sa.Column("citations", sa.JSON()),
        sa.Column("agent_trace", sa.JSON()),
        sa.Column("usage", sa.JSON()),
        _ts("created_at"),
    )
    op.create_index("ix_messages_conversation_created", "messages", ["conversation_id", "created_at"])

    op.create_table(
        "app_metadata",
        sa.Column("key", sa.String(100), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    for table in (
        "app_metadata",
        "messages",
        "conversations",
        "ingestion_jobs",
        "chunks",
        "documents",
        "auth_tokens",
        "users",
    ):
        op.drop_table(table)
