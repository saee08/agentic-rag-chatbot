"""Internal tool interface for the agent (this is NOT the Model Context Protocol).

Tools take validated Pydantic inputs and return Pydantic outputs. Security properties:
* The requesting user's id comes from ``ToolContext`` (set by the server from the auth token),
  never from model-generated arguments, so the model cannot widen the search scope.
* Each tool declares a permission; the registry refuses tools whose permission was not granted.
* All tools are read-only.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document
from app.rag.retriever import Retriever
from app.rag.vector_store import ScoredChunk

READ_DOCUMENTS = "documents:read"


@dataclass
class ToolContext:
    session: Session
    owner_id: str


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    permission: ClassVar[str]
    input_model: ClassVar[type[ToolInput]]

    @abstractmethod
    def run(self, ctx: ToolContext, args: Any) -> BaseModel: ...


# --- search_documents --------------------------------------------------------------------------


class SearchDocumentsInput(ToolInput):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)
    document_ids: list[str] | None = Field(default=None, max_length=20)


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None
    text: str
    score: float
    match: str


class SearchDocumentsOutput(BaseModel):
    query: str
    hits: list[SearchHit]
    timings_ms: dict[str, float]

    @property
    def best_score(self) -> float | None:
        return max((h.score for h in self.hits), default=None)


class SearchDocumentsTool(Tool):
    name = "search_documents"
    description = "Semantic search over the user's processed documents."
    permission = READ_DOCUMENTS
    input_model = SearchDocumentsInput

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def run(self, ctx: ToolContext, args: SearchDocumentsInput) -> SearchDocumentsOutput:
        result = self.retriever.retrieve(ctx.session, ctx.owner_id, args.query, args.top_k, args.document_ids)
        return SearchDocumentsOutput(query=result.query, hits=[_hit(c) for c in result.chunks],
                                     timings_ms=result.timings_ms)


def _hit(c: ScoredChunk) -> SearchHit:
    return SearchHit(chunk_id=c.chunk_id, document_id=c.document_id, document_name=c.document_name,
                     page=c.page, text=c.text, score=c.score, match=c.match)


# --- list_documents ----------------------------------------------------------------------------


class ListDocumentsInput(ToolInput):
    pass


class DocumentInfo(BaseModel):
    document_id: str
    name: str
    file_type: str
    status: str
    pages: int | None
    chunks: int


class ListDocumentsOutput(BaseModel):
    documents: list[DocumentInfo]


class ListDocumentsTool(Tool):
    name = "list_documents"
    description = "List the user's documents with type, page count and processing status."
    permission = READ_DOCUMENTS
    input_model = ListDocumentsInput
    max_documents = 100

    def run(self, ctx: ToolContext, args: ListDocumentsInput) -> ListDocumentsOutput:
        rows = ctx.session.scalars(
            select(Document).where(Document.owner_id == ctx.owner_id)
            .order_by(Document.created_at.desc()).limit(self.max_documents)
        )
        return ListDocumentsOutput(documents=[
            DocumentInfo(document_id=d.id, name=d.filename, file_type=d.file_type, status=d.status.value,
                         pages=d.page_count, chunks=d.chunk_count)
            for d in rows
        ])


# --- registry ----------------------------------------------------------------------------------


@dataclass
class ToolResult:
    name: str
    args: dict[str, Any]
    output: BaseModel | None
    error: str | None
    duration_ms: float

    @property
    def ok(self) -> bool:
        return self.error is None


class ToolRegistry:
    def __init__(self, tools: list[Tool], granted_permissions: set[str]):
        self._tools = {t.name: t for t in tools}
        self.granted = set(granted_permissions)

    @property
    def names(self) -> list[str]:
        return [n for n, t in self._tools.items() if t.permission in self.granted]

    def execute(self, name: str, raw_args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        start = time.perf_counter()
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, raw_args, None, f"unknown tool '{name}'", 0.0)
        if tool.permission not in self.granted:
            return ToolResult(name, raw_args, None, f"permission '{tool.permission}' not granted", 0.0)
        try:
            args = tool.input_model.model_validate(raw_args)
        except ValidationError as exc:
            msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return ToolResult(name, raw_args, None, f"invalid arguments: {msg}", 0.0)
        output = tool.run(ctx, args)
        return ToolResult(name, args.model_dump(), output, None, round((time.perf_counter() - start) * 1000, 2))
