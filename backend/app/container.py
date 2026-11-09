"""Builds the long-lived objects of one application instance (no module-level singletons)."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from sqlalchemy import select

from app.agents.orchestrator import AgentLimits, AgentRunner
from app.agents.tools import READ_DOCUMENTS, ListDocumentsTool, SearchDocumentsTool, ToolRegistry
from app.core.config import Settings
from app.core.security import RateLimiter
from app.db.session import Database
from app.models import AppMetadata
from app.rag.embeddings import Embedder, SentenceTransformerEmbedder
from app.rag.llm import LLMClient
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService
from app.workers.ingestion_worker import IngestionWorker


@dataclass
class RateLimiters:
    auth: RateLimiter
    chat: RateLimiter
    upload: RateLimiter


class Container:
    def __init__(self, settings: Settings, *, embedder: Embedder | None = None,
                 llm_transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.db = Database(settings)
        self.embedder: Embedder = embedder or SentenceTransformerEmbedder(
            settings.embedding_model, settings.embedding_dim, settings.embedding_batch_size, settings.embedding_device
        )
        if self.embedder.dim != settings.embedding_dim:
            raise RuntimeError("embedder dimension does not match EMBEDDING_DIM")
        self.retriever = Retriever(self.embedder, VectorStore(settings.embedding_dim),
                                   settings.retrieval_top_k, settings.retrieval_min_score)
        self.llm = LLMClient(settings, transport=llm_transport)
        self.auth = AuthService(settings)
        self.documents = DocumentService(settings)
        self.ingestion = IngestionService(settings, self.db, self.embedder, self.documents)
        self.worker = IngestionWorker(self.ingestion, self.db, settings.ingestion_workers)
        self.tools = ToolRegistry([SearchDocumentsTool(self.retriever), ListDocumentsTool()], {READ_DOCUMENTS})
        self.limits = AgentLimits(
            max_iterations=settings.agent_max_iterations,
            max_tool_calls=settings.agent_max_tool_calls,
            timeout_seconds=settings.agent_timeout_seconds,
            token_budget=settings.agent_token_budget,
            answer_max_tokens=settings.llm_max_tokens,
            strong_score=settings.retrieval_strong_score,
        )
        self.chat = ChatService(self.db, self.agent_runner, settings.agent_history_messages)
        self.rate_limits = RateLimiters(
            auth=RateLimiter(settings.rate_limit_auth_per_minute),
            chat=RateLimiter(settings.rate_limit_chat_per_minute),
            upload=RateLimiter(settings.rate_limit_upload_per_minute),
        )

    def agent_runner(self) -> AgentRunner:
        return AgentRunner(self.llm, self.tools, self.limits)

    def check_embedding_metadata(self) -> None:
        """Refuse to start if the database was indexed with a different embedding model:
        mixing vector spaces would silently return garbage results."""
        expected = {"embedding_model": self.embedder.model_name, "embedding_dim": str(self.embedder.dim)}
        with self.db.transaction() as session:
            stored = {m.key: m.value for m in session.scalars(
                select(AppMetadata).where(AppMetadata.key.in_(list(expected)))
            )}
            for key, value in expected.items():
                if key not in stored:
                    session.add(AppMetadata(key=key, value=value))
                elif stored[key] != value:
                    raise RuntimeError(
                        f"Database was indexed with {key}={stored[key]!r} but the app is configured with "
                        f"{value!r}. Re-index into a fresh database or restore the original setting."
                    )

    def close(self) -> None:
        self.worker.shutdown(wait=True)
        self.llm.close()
        self.db.dispose()
