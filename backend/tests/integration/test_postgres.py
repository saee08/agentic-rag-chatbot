"""Runs the production storage path: PostgreSQL + pgvector (embedded server via pgserver).

    pytest -m postgres
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app.main import create_app
from tests.conftest import ask, auth_headers, make_settings, new_conversation, upload, wait_for_document
from tests.helpers import ACME_REPORT, FakeEmbedder, FakeOpenRouter, make_pdf

pytestmark = pytest.mark.postgres


@pytest.fixture(scope="module")
def pg_url(tmp_path_factory):
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server(tmp_path_factory.mktemp("pg"), cleanup_mode="stop")
    server.psql("DROP DATABASE IF EXISTS ragtest;")
    server.psql("CREATE DATABASE ragtest;")
    uri = server.get_uri("ragtest")
    yield uri.replace("postgresql://", "postgresql+psycopg://", 1)
    server.cleanup()


def test_full_flow_on_postgres_pgvector(tmp_path, pg_url):
    settings = make_settings(tmp_path, database_url=pg_url)
    app = create_app(settings, embedder=FakeEmbedder(), llm_transport=FakeOpenRouter().transport())
    with TestClient(app) as client:
        engine = create_engine(pg_url)
        with engine.connect() as conn:
            col_type = conn.execute(text(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
            )).scalar()
            assert col_type == "vector(384)"
            indexdef = conn.execute(text(
                "SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_chunks_embedding_hnsw'"
            )).scalar()
            assert "hnsw" in indexdef and "vector_cosine_ops" in indexdef

        alice, bob = auth_headers(client, "alice@example.com"), auth_headers(client, "bob@example.com")
        doc = upload(client, alice, "acme.pdf", make_pdf([ACME_REPORT]))["items"][0]["document"]
        assert wait_for_document(client, alice, doc["id"])["status"] == "ready"
        upload(client, bob, "bob.txt", b"Bob notes: customer retention secrets for the board.")

        conv = new_conversation(client, alice)
        reply = ask(client, alice, conv, "What were the key performance indicators?")["assistant_message"]
        assert reply["status"] == "answered" and "94 percent" in reply["content"]
        assert reply["citations"][0]["document_id"] == doc["id"] and reply["citations"][0]["page"] == 1
        assert all("Bob notes" not in c["snippet"] for c in reply["citations"])

        unrelated = ask(client, alice, conv, "What is the boiling point of tungsten?")["assistant_message"]
        assert unrelated["status"] == "no_answer"

        assert client.delete(f"/api/v1/documents/{doc['id']}", headers=alice).status_code == 204
        with engine.connect() as conn:
            remaining = conn.execute(text("SELECT count(*) FROM chunks WHERE document_id = :d"), {"d": doc["id"]})
            assert remaining.scalar() == 0
        engine.dispose()
