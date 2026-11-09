"""FastAPI dependencies: container, DB session, authenticated user, rate limiting."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.container import Container
from app.core.exceptions import AuthenticationError, RateLimitedError
from app.core.security import RateLimiter
from app.models import User

_bearer = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


def get_session(container: ContainerDep) -> Iterator[Session]:
    session = container.db.session()
    try:
        yield session
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(get_session)]


def get_token(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise AuthenticationError()
    return credentials.credentials


TokenDep = Annotated[str, Depends(get_token)]


def get_current_user(container: ContainerDep, session: SessionDep, token: TokenDep) -> User:
    return container.auth.authenticate(session, token)


CurrentUser = Annotated[User, Depends(get_current_user)]


def client_ip(request: Request, container: Container) -> str:
    if container.settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce(limiter: RateLimiter, key: str) -> None:
    retry_after = limiter.hit(key)
    if retry_after is not None:
        raise RateLimitedError(retry_after)
