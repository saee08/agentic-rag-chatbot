"""Request/response schemas (the public API contract)."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

T = TypeVar("T")

_EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")


class ListResponse(BaseModel, Generic[T]):
    items: list[T]


# --- auth --------------------------------------------------------------------------------------


class Credentials(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(min_length=1, max_length=256)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("invalid e-mail address")
        return v


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 - OAuth token type, not a secret
    expires_at: datetime


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    created_at: datetime


# --- documents ---------------------------------------------------------------------------------


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    size_bytes: int
    status: str
    error: str | None
    page_count: int | None
    chunk_count: int
    created_at: datetime
    updated_at: datetime

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v: Any) -> str:
        return getattr(v, "value", v)


class UploadResult(BaseModel):
    document: DocumentOut
    duplicate: bool = False


class UploadResponse(BaseModel):
    items: list[UploadResult]


# --- conversations -----------------------------------------------------------------------------


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


MessageStatus = Literal["sent", "answered", "no_answer", "no_documents", "direct_reply", "error"]


class SourceOut(BaseModel):
    """One cited passage. ``number`` matches the [n] markers in the answer text."""

    number: int
    document_id: str | None
    document_name: str
    page_number: int | None
    chunk_id: str | None
    excerpt: str  # the most relevant sentence(s), verbatim from the passage
    content: str  # the full retrieved passage
    score: float | None
    match: str | None  # "semantic", "keyword" or "semantic+keyword"


class RetrievalStats(BaseModel):
    passages_retrieved: int | None  # distinct passages the agent gathered (None for legacy messages)
    passages_cited: int
    documents: int  # distinct documents among cited passages
    pages: int  # distinct (document, page) pairs among cited passages


class TokenUsage(BaseModel):
    prompt: int
    completion: int
    total: int


class MessageMetadata(BaseModel):
    model: str | None
    latency_ms: float | None
    llm_calls: int | None
    tokens: TokenUsage | None
    grounded: bool | None
    retrieval: RetrievalStats


class StepOut(BaseModel):
    """A user-safe description of one processing stage (no prompts or model reasoning)."""

    kind: Literal["search", "library", "assess", "generate", "stop"]
    label: str
    detail: str | None = None


class ErrorInfo(BaseModel):
    code: str
    message: str
    retryable: bool


class MessageOut(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    status: MessageStatus
    content: str  # Markdown answer, the user's text, or a user-safe error message; never null
    sources: list[SourceOut] = Field(default_factory=list)
    metadata: MessageMetadata | None = None  # assistant replies only
    steps: list[StepOut] = Field(default_factory=list)
    error: ErrorInfo | None = None  # set iff status == "error"
    created_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


class ChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()


class ChatResponse(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut


# --- health ------------------------------------------------------------------------------------


class HealthOut(BaseModel):
    status: str


class ReadinessOut(BaseModel):
    status: str
    checks: dict[str, Any]
