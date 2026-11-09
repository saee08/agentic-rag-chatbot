from __future__ import annotations

import os

# Must happen before any app module is imported: never read the developer's real .env
# (which may contain live credentials) and never inherit their LLM key from the environment.
os.environ["APP_ENV_FILE"] = ""
for _var in ("OPENROUTER_API_KEY", "DATABASE_URL", "EMBEDDING_DIM", "EMBEDDING_MODEL"):
    os.environ.pop(_var, None)

import time  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.helpers import FakeEmbedder, FakeOpenRouter  # noqa: E402


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "app_env": "test",
        "database_url": f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        "upload_dir": tmp_path / "uploads",
        "frontend_dir": None,
        "openrouter_api_key": "test-key",
        "log_json": False,
        "log_level": "WARNING",
        "embedding_model": "fake-hash-embedder",
        "embedding_preload": False,
        "rate_limit_auth_per_minute": 1000,
        "rate_limit_chat_per_minute": 1000,
        "rate_limit_upload_per_minute": 1000,
        "llm_max_retries": 2,
        # Calibrated for FakeEmbedder's hashed bag-of-words scores (MiniLM defaults are 0.20 / 0.55).
        "retrieval_min_score": 0.15,
        "retrieval_strong_score": 0.5,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def fake_llm() -> FakeOpenRouter:
    return FakeOpenRouter()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def client(settings: Settings, fake_embedder: FakeEmbedder, fake_llm: FakeOpenRouter) -> Iterator[TestClient]:
    app = create_app(settings, embedder=fake_embedder, llm_transport=fake_llm.transport())
    with TestClient(app) as test_client:
        yield test_client


# --- API helpers ---------------------------------------------------------------------------------


def auth_headers(
    client: TestClient, email: str = "alice@example.com", password: str = "correct-horse-1"  # noqa: S107
) -> dict:
    r = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert r.status_code in (201, 409), r.text
    r = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def upload(client: TestClient, headers: dict, name: str, content: bytes, expected: int = 202) -> dict:
    r = client.post("/api/v1/documents", headers=headers, files=[("files", (name, content))])
    assert r.status_code == expected, r.text
    return r.json()


def wait_for_document(client: TestClient, headers: dict, document_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        doc = client.get(f"/api/v1/documents/{document_id}", headers=headers).json()
        if doc["status"] in ("ready", "failed"):
            return doc
        time.sleep(0.05)
    raise AssertionError(f"document {document_id} not processed within {timeout}s")


def new_conversation(client: TestClient, headers: dict) -> str:
    r = client.post("/api/v1/conversations", headers=headers, json={})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def ask(client: TestClient, headers: dict, conversation_id: str, content: str, expected: int = 200) -> dict:
    r = client.post(f"/api/v1/conversations/{conversation_id}/messages", headers=headers, json={"content": content})
    assert r.status_code == expected, r.text
    return r.json()
