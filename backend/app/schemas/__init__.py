"""Request/response schemas (the public API contract)."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Generic, TypeVar

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


class Citation(BaseModel):
    number: int
    document_id: str | None
    document_name: str
    page: int | None
    chunk_id: str | None
    score: float | None
    snippet: str


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    status: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] | None = None
    created_at: datetime

    @field_validator("role", mode="before")
    @classmethod
    def _role(cls, v: Any) -> str:
        return getattr(v, "value", v)

    @field_validator("citations", "agent_trace", mode="before")
    @classmethod
    def _none_to_list(cls, v: Any) -> Any:
        return v or []


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
