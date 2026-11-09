from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from app.api.deps import ContainerDep, CurrentUser, SessionDep, TokenDep, client_ip, enforce
from app.schemas import Credentials, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(body: Credentials, request: Request, container: ContainerDep, session: SessionDep) -> UserOut:
    enforce(container.rate_limits.auth, f"ip:{client_ip(request, container)}")
    user = container.auth.register(session, body.email, body.password)
    return UserOut.model_validate(user)


@router.post("/login", response_model=TokenResponse)
def login(body: Credentials, request: Request, container: ContainerDep, session: SessionDep) -> TokenResponse:
    enforce(container.rate_limits.auth, f"ip:{client_ip(request, container)}")
    token, expires = container.auth.login(session, body.email, body.password)
    return TokenResponse(access_token=token, expires_at=expires)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(_: CurrentUser, token: TokenDep, container: ContainerDep, session: SessionDep) -> Response:
    container.auth.logout(session, token)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)
