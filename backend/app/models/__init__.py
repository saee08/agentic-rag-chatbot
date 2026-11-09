"""SQLAlchemy ORM models.

Ownership: every user-owned row carries ``owner_id`` (chunks denormalise it so vector search can
filter by owner without a join). Deleting a user, document or conversation cascades in the
database (ON DELETE CASCADE) so vectors can never outlive their document.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.db.types import GUID, EmbeddingVector


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class DocumentStatus(enum.StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class JobStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class MessageRole(enum.StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


def _enum(e: type[enum.Enum], name: str) -> Enum:
    # Stored as plain strings (portable, no native PG enum to migrate).
    return Enum(e, name=name, native_enum=False, length=20, values_callable=lambda x: [m.value for m in x])


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(GUID(), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)


class AuthToken(TimestampMixin, Base):
    __tablename__ = "auth_tokens"

    id: Mapped[str] = mapped_column(GUID(), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(lazy="joined")


class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("owner_id", "content_sha256", name="uq_documents_owner_hash"),
        Index("ix_documents_owner_created", "owner_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(GUID(), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)  # display name only
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(255), nullable=False)  # server-generated
    status: Mapped[DocumentStatus] = mapped_column(
        _enum(DocumentStatus, "document_status"), default=DocumentStatus.PENDING, nullable=False
    )
    error: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    embedding_model: Mapped[str | None] = mapped_column(String(200))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
        Index("ix_chunks_owner", "owner_id"),
    )

    # Stable id: uuid5(document_id, ordinal) -> re-ingesting the same document yields the same ids.
    id: Mapped[str] = mapped_column(GUID(), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        GUID(), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    owner_id: Mapped[str] = mapped_column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[np.ndarray] = mapped_column(EmbeddingVector(get_settings().embedding_dim), nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")


class IngestionJob(TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(GUID(), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        GUID(), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[JobStatus] = mapped_column(
        _enum(JobStatus, "job_status"), default=JobStatus.PENDING, nullable=False, index=True
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    timings_ms: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class Conversation(TimestampMixin, Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_owner_updated", "owner_id", "updated_at"),)

    id: Mapped[str] = mapped_column(GUID(), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="New conversation")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )


class Message(TimestampMixin, Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)

    id: Mapped[str] = mapped_column(GUID(), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(
        GUID(), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(_enum(MessageRole, "message_role"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str | None] = mapped_column(String(30))
    error_code: Mapped[str | None] = mapped_column(String(50))
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    agent_trace: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class AppMetadata(Base):
    """Key/value facts about the deployment, e.g. which embedding model built the vectors."""

    __tablename__ = "app_metadata"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
