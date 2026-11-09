"""Registration, login, token validation and logout."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import (
    AuthenticationError,
    ConflictError,
    PermissionDeniedError,
    ValidationFailedError,
)
from app.core.logging import log_event
from app.core.security import (
    DUMMY_PASSWORD_HASH,
    hash_password,
    hash_token,
    new_access_token,
    verify_password,
)
from app.models import AuthToken, User

logger = logging.getLogger(__name__)


def _aware(dt: datetime) -> datetime:
    # SQLite drops tzinfo; values are always written in UTC.
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class AuthService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def register(self, session: Session, email: str, password: str) -> User:
        if not self.settings.allow_registration:
            raise PermissionDeniedError("Registration is disabled.")
        if len(password) < self.settings.password_min_length:
            raise ValidationFailedError(
                f"Password must be at least {self.settings.password_min_length} characters long."
            )
        if password.lower() == email.lower() or password.lower() in email.lower():
            raise ValidationFailedError("Password must not contain the e-mail address.")
        user = User(email=email, password_hash=hash_password(password))
        session.add(user)
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise ConflictError("An account with this e-mail already exists.") from exc
        log_event(logger, "user registered", user_id=user.id)
        return user

    def login(self, session: Session, email: str, password: str) -> tuple[str, datetime]:
        user = session.scalar(select(User).where(User.email == email))
        # Always run the hash so response time does not reveal whether the account exists.
        valid = verify_password(password, user.password_hash if user else DUMMY_PASSWORD_HASH)
        if not user or not valid or not user.is_active:
            log_event(logger, "login failed", logging.WARNING)
            raise AuthenticationError("Invalid e-mail or password.")
        now = datetime.now(UTC)
        token = new_access_token()
        expires = now + timedelta(hours=self.settings.access_token_ttl_hours)
        session.execute(delete(AuthToken).where(AuthToken.user_id == user.id, AuthToken.expires_at < now))
        session.add(AuthToken(user_id=user.id, token_hash=hash_token(token), expires_at=expires))
        session.commit()
        log_event(logger, "login succeeded", user_id=user.id)
        return token, expires

    def authenticate(self, session: Session, token: str) -> User:
        record = session.scalar(select(AuthToken).where(AuthToken.token_hash == hash_token(token)))
        if (
            record is None
            or record.revoked_at is not None
            or _aware(record.expires_at) <= datetime.now(UTC)
            or not record.user.is_active
        ):
            raise AuthenticationError("Invalid or expired access token.")
        return record.user

    def logout(self, session: Session, token: str) -> None:
        record = session.scalar(select(AuthToken).where(AuthToken.token_hash == hash_token(token)))
        if record is not None and record.revoked_at is None:
            record.revoked_at = datetime.now(UTC)
            session.commit()
