"""Pure-ASGI middleware (safe for streaming responses).

* RequestContextMiddleware: request id (X-Request-ID), access log with latency, security headers.
* BodySizeLimitMiddleware: rejects oversized request bodies before they are fully buffered.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
import uuid
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import log_event, request_id_var

logger = logging.getLogger("app.access")

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
    (
        b"content-security-policy",
        b"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        b"connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    ),
]


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                existing = {k.lower() for k, _ in headers}
                headers.extend(h for h in SECURITY_HEADERS if h[0] not in existing)
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            path = scope.get("path", "")
            if not path.startswith(("/assets", "/health")):
                log_event(logger, "request", method=scope.get("method"), path=path, status=status,
                          duration_ms=round((time.perf_counter() - start) * 1000, 2))
            request_id_var.reset(token)


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, default_limit: int, upload_limit: int, upload_path_prefix: str):
        self.app = app
        self.default_limit = default_limit
        self.upload_limit = upload_limit
        self.upload_prefix = upload_path_prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        limit = self.upload_limit if path.startswith(self.upload_prefix) else self.default_limit
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > limit:
                    await self._reject(send, limit)
                    return
            except ValueError:
                pass
        received = 0
        exceeded = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            if exceeded:  # stop feeding the app once over the limit
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise _TooLarge()
            return message

        async def guarded_send(message: Message) -> None:
            # Whatever the app does after the limit trips (FastAPI turns body errors into 400),
            # suppress it: the client gets our 413 instead.
            if not exceeded:
                await send(message)

        with contextlib.suppress(_TooLarge):
            await self.app(scope, limited_receive, guarded_send)
        if exceeded:
            await self._reject(send, limit)

    @staticmethod
    async def _reject(send: Send, limit: int) -> None:
        body: dict[str, Any] = {"error": {"code": "payload_too_large",
                                          "message": f"Request body exceeds {limit // (1024 * 1024)} MB.",
                                          "request_id": request_id_var.get()}}
        payload = json.dumps(body).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(payload)).encode())]})
        await send({"type": "http.response.body", "body": payload})


class _TooLarge(Exception):
    pass
