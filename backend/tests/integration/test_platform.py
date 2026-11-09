"""Isolation at the data layer, persistence across restarts, job recovery, migrations,
health/readiness, headers and error envelopes."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect

from app.db.migrate import alembic_config, upgrade
from app.main import create_app
from app.models import Document, DocumentStatus, IngestionJob, JobStatus, User
from app.rag.vector_store import VectorStore
from tests.conftest import ask, auth_headers, make_settings, new_conversation, upload, wait_for_document
from tests.helpers import ACME_REPORT, FakeEmbedder, FakeOpenRouter


def build(settings, llm=None):
    return create_app(settings, embedder=FakeEmbedder(), llm_transport=(llm or FakeOpenRouter()).transport())


def test_vector_search_is_owner_scoped_at_the_data_layer(client, fake_embedder):
    a, b = auth_headers(client, "a@example.com"), auth_headers(client, "b@example.com")
    for h, text in ((a, ACME_REPORT), (b, "Bob's private diary about customer retention secrets.")):
        doc = upload(client, h, "doc.txt", text.encode())["items"][0]["document"]
        wait_for_document(client, h, doc["id"])
    container = client.app.state.container
    with container.db.transaction() as s:
        ids = {u.email: u.id for u in s.query(User).all()}
        query = fake_embedder.embed(["customer retention"])[0]
        store = VectorStore(384)
        a_hits = store.search(s, ids["a@example.com"], query, 10, -1.0)
        b_hits = store.search(s, ids["b@example.com"], query, 10, -1.0)
        assert a_hits and b_hits
        assert all("diary" not in h.text for h in a_hits)
        assert all("Acme" not in h.text for h in b_hits)
        # Filtering by someone else's document id yields nothing.
        b_doc = s.query(Document).filter_by(owner_id=ids["b@example.com"]).one()
        assert store.search(s, ids["a@example.com"], query, 10, -1.0, document_ids=[b_doc.id]) == []


def test_vector_store_rejects_wrong_dimensions(client):
    with client.app.state.container.db.transaction() as s:
        try:
            VectorStore(384).search(s, "x", np.ones(10, dtype=np.float32), 5, 0.0)
        except ValueError as exc:
            assert "dimension" in str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_data_survives_restart(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(build(settings)) as c:
        h = auth_headers(c)
        doc = upload(c, h, "acme.txt", ACME_REPORT.encode())["items"][0]["document"]
        wait_for_document(c, h, doc["id"])
        conv = new_conversation(c, h)
        ask(c, h, conv, "Which factories were opened?")
    with TestClient(build(settings)) as c:  # new process, same database
        h = auth_headers(c)
        assert c.get(f"/api/v1/documents/{doc['id']}", headers=h).json()["status"] == "ready"
        assert len(c.get(f"/api/v1/conversations/{conv}", headers=h).json()["messages"]) == 2
        reply = ask(c, h, conv, "Who is the chief executive?")["assistant_message"]
        assert reply["status"] == "answered" and "Dana Whitfield" in reply["content"]


def test_unfinished_jobs_are_recovered_on_startup(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(build(settings)) as c:
        h = auth_headers(c)
        doc = upload(c, h, "acme.txt", ACME_REPORT.encode())["items"][0]["document"]
        wait_for_document(c, h, doc["id"])
        container = c.app.state.container
        with container.db.transaction() as s:  # simulate a crash mid-ingestion
            d = s.get(Document, doc["id"])
            d.status = DocumentStatus.PROCESSING
            s.add(IngestionJob(document_id=d.id, status=JobStatus.RUNNING))
    with TestClient(build(settings)) as c:
        h = auth_headers(c)
        c.app.state.container.worker.wait_idle()
        assert c.get(f"/api/v1/documents/{doc['id']}", headers=h).json()["status"] == "ready"


def test_embedding_model_change_is_refused(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(build(settings)):
        pass
    app = create_app(settings, embedder=FakeEmbedder(model_name="another-model"),
                     llm_transport=FakeOpenRouter().transport())
    try:
        with TestClient(app):
            raise AssertionError("startup should have failed")
    except RuntimeError as exc:
        assert "embedding_model" in str(exc)


def test_migrations_upgrade_and_downgrade(tmp_path):
    from alembic import command

    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    upgrade(url)
    tables = set(inspect(create_engine(url)).get_table_names())
    assert {"users", "documents", "chunks", "conversations", "messages", "ingestion_jobs", "auth_tokens"} <= tables
    command.downgrade(alembic_config(url), "base")
    assert set(inspect(create_engine(url)).get_table_names()) <= {"alembic_version"}
    upgrade(url)  # and back up again


def test_health_and_readiness(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["checks"] == {"database": "ok", "embedding_model": "loaded", "llm": "configured"}


def test_security_headers_request_id_and_error_envelope(client):
    r = client.get("/api/v1/documents", headers={"X-Request-ID": "trace-abc-12345"})
    assert r.status_code == 401
    assert r.headers["x-request-id"] == "trace-abc-12345"
    assert r.json()["error"]["request_id"] == "trace-abc-12345"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert client.get("/api/v1/nope").json()["error"]["code"] == "not_found"
    bogus = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    assert bogus.headers["x-request-id"] != "bad id with spaces"


def test_cors_only_allows_configured_origins(tmp_path):
    app = build(make_settings(tmp_path, cors_origins=["https://app.example.com"]))
    with TestClient(app) as c:
        ok = c.options("/api/v1/documents", headers={"Origin": "https://app.example.com",
                                                     "Access-Control-Request-Method": "GET"})
        assert ok.headers.get("access-control-allow-origin") == "https://app.example.com"
        bad = c.options("/api/v1/documents", headers={"Origin": "https://evil.example",
                                                      "Access-Control-Request-Method": "GET"})
        assert "access-control-allow-origin" not in bad.headers


def test_frontend_is_served(tmp_path):
    from app.core.config import REPO_DIR

    app = build(make_settings(tmp_path, frontend_dir=REPO_DIR / "frontend"))
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200 and "<title>" in r.text
        assert c.get("/app.js").status_code == 200


def test_job_claim_is_exclusive_and_respects_live_workers(client):
    from datetime import UTC, datetime

    h = auth_headers(client)
    doc = upload(client, h, "acme.txt", ACME_REPORT.encode())["items"][0]["document"]
    wait_for_document(client, h, doc["id"])
    container = client.app.state.container
    with container.db.transaction() as s:
        pending = IngestionJob(document_id=doc["id"], status=JobStatus.PENDING)
        live = IngestionJob(document_id=doc["id"], status=JobStatus.RUNNING, started_at=datetime.now(UTC))
        s.add_all([pending, live])
        s.flush()
        pending_id, live_id = pending.id, live.id
    assert container.ingestion._claim(pending_id) is not None  # first claimant wins
    assert container.ingestion._claim(pending_id) is None  # second loses
    assert container.ingestion._claim(live_id) is None  # a job running elsewhere is left alone
