"""Application factory.

Run from the ``backend`` directory with ``uvicorn app.main:create_app --factory``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.api.routes import auth, conversations, documents, health
from app.container import Container
from app.core.config import Settings, get_settings
from app.core.exceptions import install_exception_handlers
from app.core.logging import configure_logging, log_event
from app.core.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from app.db.migrate import upgrade
from app.rag.embeddings import Embedder

API_PREFIX = "/api/v1"
logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, *, embedder: Embedder | None = None,
               llm_transport: httpx.BaseTransport | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.auto_migrate:
            await run_in_threadpool(upgrade, settings.database_url)
        container = Container(settings, embedder=embedder, llm_transport=llm_transport)
        await run_in_threadpool(container.check_embedding_metadata)
        if settings.embedding_preload and hasattr(container.embedder, "load"):
            await run_in_threadpool(container.embedder.load)
        app.state.container = container
        container.worker.recover()
        log_event(logger, "application started", env=settings.app_env, llm_configured=settings.llm_configured,
                  database="sqlite" if settings.is_sqlite else "postgresql", model=settings.llm_model)
        try:
            yield
        finally:
            await run_in_threadpool(container.close)

    app = FastAPI(
        title="Agentic RAG Assistant",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs" if settings.app_env != "production" else None,
        redoc_url=None,
        openapi_url="/api/openapi.json" if settings.app_env != "production" else None,
    )
    install_exception_handlers(app)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,  # bearer tokens, no cookies
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )
    app.add_middleware(
        BodySizeLimitMiddleware,
        default_limit=1024 * 1024,
        upload_limit=settings.max_upload_bytes * settings.max_files_per_upload + 1024 * 1024,
        upload_path_prefix=f"{API_PREFIX}/documents",
    )
    app.add_middleware(RequestContextMiddleware)

    app.include_router(health.router)
    for router in (auth.router, documents.router, conversations.router):
        app.include_router(router, prefix=API_PREFIX)

    if settings.frontend_dir and settings.frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    return app

