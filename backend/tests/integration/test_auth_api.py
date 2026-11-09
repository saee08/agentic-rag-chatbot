from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from app.core.security import hash_token
from app.main import create_app
from app.models import AuthToken, User
from tests.conftest import auth_headers, make_settings
from tests.helpers import FakeEmbedder, FakeOpenRouter


def test_register_login_me_logout(client):
    r = client.post("/api/v1/auth/register", json={"email": " Alice@Example.com ", "password": "correct-horse-1"})
    assert r.status_code == 201
    assert r.json()["email"] == "alice@example.com" and "password" not in r.text

    r = client.post("/api/v1/auth/login", json={"email": "alice@example.com", "password": "correct-horse-1"})
    assert r.status_code == 200
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/v1/auth/me", headers=headers).json()["email"] == "alice@example.com"

    assert client.post("/api/v1/auth/logout", headers=headers).status_code == 204
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 401


def test_tokens_are_stored_hashed(client):
    headers = auth_headers(client)
    token = headers["Authorization"].split()[1]
    db = client.app.state.container.db
    with db.transaction() as s:
        stored = s.query(AuthToken).one()
        assert stored.token_hash == hash_token(token) and stored.token_hash != token
        assert s.query(User).one().password_hash.startswith("scrypt$")


def test_duplicate_registration_and_weak_passwords(client):
    auth_headers(client)
    r = client.post("/api/v1/auth/register", json={"email": "alice@example.com", "password": "correct-horse-1"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
    r = client.post("/api/v1/auth/register", json={"email": "b@example.com", "password": "short"})
    assert r.status_code == 422
    r = client.post("/api/v1/auth/register", json={"email": "not-an-email", "password": "correct-horse-1"})
    assert r.status_code == 422


def test_login_failures_do_not_reveal_which_part_was_wrong(client):
    auth_headers(client)
    wrong_pw = client.post("/api/v1/auth/login", json={"email": "alice@example.com", "password": "nope-nope-nope"})
    no_user = client.post("/api/v1/auth/login", json={"email": "ghost@example.com", "password": "nope-nope-nope"})
    assert wrong_pw.status_code == no_user.status_code == 401
    assert wrong_pw.json()["error"]["message"] == no_user.json()["error"]["message"]


def test_validation_errors_never_echo_passwords(client):
    r = client.post("/api/v1/auth/login", json={"email": "x", "password": "my-secret-value"})
    assert r.status_code == 422 and "my-secret-value" not in r.text


def test_protected_endpoints_require_a_valid_token(client):
    for method, path in [("get", "/api/v1/documents"), ("get", "/api/v1/conversations"),
                         ("post", "/api/v1/conversations"), ("get", "/api/v1/auth/me")]:
        r = getattr(client, method)(path)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        r = getattr(client, method)(path, headers={"Authorization": "Bearer forged-token"})
        assert r.status_code == 401


def test_expired_tokens_are_rejected(client):
    headers = auth_headers(client)
    with client.app.state.container.db.transaction() as s:
        s.query(AuthToken).one().expires_at = datetime.now(UTC) - timedelta(minutes=1)
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 401


def test_registration_can_be_disabled(tmp_path):
    app = create_app(make_settings(tmp_path, allow_registration=False), embedder=FakeEmbedder(),
                     llm_transport=FakeOpenRouter().transport())
    with TestClient(app) as c:
        r = c.post("/api/v1/auth/register", json={"email": "a@example.com", "password": "correct-horse-1"})
        assert r.status_code == 403


def test_login_is_rate_limited(tmp_path):
    app = create_app(make_settings(tmp_path, rate_limit_auth_per_minute=3), embedder=FakeEmbedder(),
                     llm_transport=FakeOpenRouter().transport())
    with TestClient(app) as c:
        codes = [c.post("/api/v1/auth/login", json={"email": "a@example.com", "password": "whatever-123"}).status_code
                 for _ in range(4)]
        assert codes == [401, 401, 401, 429]
