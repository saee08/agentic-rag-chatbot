"""Password hashing, opaque access tokens and a per-process rate limiter.

Passwords: scrypt (memory-hard, in the standard library) with a per-password random salt.
Access tokens: 256-bit random strings; only their SHA-256 digest is stored, so a database leak
does not yield usable tokens, and tokens can be revoked server-side (logout).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from collections import deque

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_BYTES = 16


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, n, r, p, salt, expected = encoded.split("$")
        if algo != "scrypt":
            return False
        expected_bytes = _unb64(expected)
        digest = hashlib.scrypt(
            password.encode(), salt=_unb64(salt), n=int(n), r=int(r), p=int(p), dklen=len(expected_bytes)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, expected_bytes)


# Used to equalise timing when a login targets an unknown e-mail address.
DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(16))


def new_access_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class RateLimiter:
    """Sliding-window limiter keyed by an arbitrary string (user id, client IP...).

    State is per process. With several API replicas the effective limit is multiplied by the
    replica count; put a shared limiter (reverse proxy or Redis) in front if that matters.
    """

    def __init__(self, limit: int, window_seconds: float = 60.0, max_keys: int = 100_000):
        self.limit = limit
        self.window = window_seconds
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> int | None:
        """Record a request. Returns None if allowed, else seconds until a slot frees up."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                if len(self._hits) >= self.max_keys:
                    self._evict(now)
                hits = self._hits[key] = deque()
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return int(self.window - (now - hits[0])) + 1
            hits.append(now)
            return None

    def _evict(self, now: float) -> None:
        stale = [k for k, h in self._hits.items() if not h or now - h[-1] >= self.window]
        for k in stale:
            del self._hits[k]
        if len(self._hits) >= self.max_keys:  # still full: drop the oldest half
            for k in list(self._hits)[: self.max_keys // 2]:
                del self._hits[k]
