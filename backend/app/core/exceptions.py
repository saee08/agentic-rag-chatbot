"""Application error hierarchy and the FastAPI handlers that map it to a uniform JSON envelope:

    {"error": {"code": "...", "message": "...", "request_id": "..."}}
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import request_id_var

logger = logging.getLogger(__name__)


class AppError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, code: str | None = None, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        self.headers = headers or {}


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class AuthenticationError(AppError):
    status_code = 401
    code = "unauthorized"

    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, headers={"WWW-Authenticate": "Bearer"})


class PermissionDeniedError(AppError):
    status_code = 403
    code = "forbidden"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class ValidationFailedError(AppError):
    status_code = 422
    code = "validation_error"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"


class UnsupportedMediaTypeError(AppError):
    status_code = 415
    code = "unsupported_media_type"


class RateLimitedError(AppError):
    status_code = 429
    code = "rate_limited"

    def __init__(self, retry_after_seconds: int):
        super().__init__(
            "Too many requests, please slow down.",
            headers={"Retry-After": str(max(1, retry_after_seconds))},
        )


class ServiceUnavailableError(AppError):
    status_code = 503
    code = "service_unavailable"


class DocumentParsingError(Exception):
    """Raised by parsers. The message is safe to show to the document's owner."""


def error_body(code: str, message: str, details: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message, "request_id": request_id_var.get()}
    if details is not None:
        body["details"] = details
    return {"error": body}


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(error_body(exc.code, exc.message), exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Strip the submitted input from the details so passwords are never echoed back.
        details = [{"loc": list(e.get("loc", ())), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        return JSONResponse(error_body("validation_error", "Request validation failed", details), 422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return JSONResponse(
            error_body(code, str(exc.detail)), exc.status_code, headers=getattr(exc, "headers", None)
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error", extra={"fields": {"error_type": type(exc).__name__}})
        return JSONResponse(error_body("internal_error", "An internal error occurred."), 500)
