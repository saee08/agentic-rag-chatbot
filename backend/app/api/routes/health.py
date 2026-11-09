from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.api.deps import ContainerDep
from app.schemas import HealthOut, ReadinessOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    """Liveness: the process is up and serving requests."""
    return HealthOut(status="ok")


@router.get("/ready", response_model=ReadinessOut)
def ready(container: ContainerDep) -> JSONResponse:
    """Readiness: dependencies needed to serve traffic are available."""
    checks: dict[str, object] = {}
    try:
        container.db.ping()
        checks["database"] = "ok"
    except SQLAlchemyError:
        checks["database"] = "unavailable"
    checks["embedding_model"] = "loaded" if container.embedder.loaded else "not_loaded"
    checks["llm"] = "configured" if container.llm.configured else "not_configured"
    ok = checks["database"] == "ok" and checks["embedding_model"] == "loaded"
    body = ReadinessOut(status="ready" if ok else "not_ready", checks=checks)
    return JSONResponse(body.model_dump(), status_code=200 if ok else 503)
