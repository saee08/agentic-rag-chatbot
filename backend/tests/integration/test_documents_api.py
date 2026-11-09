from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.models import Chunk, IngestionJob
from tests.conftest import auth_headers, make_settings, upload, wait_for_document
from tests.helpers import ACME_REPORT, FakeEmbedder, FakeOpenRouter, make_docx, make_pptx


def stored_files(settings) -> list[Path]:
    return [p for p in Path(settings.upload_dir).rglob("*") if p.is_file()]


def test_upload_all_formats_and_list(client):
    h = auth_headers(client)
    files = {
        "notes.txt": ACME_REPORT.encode(),
        "readme.md": b"# Setup\nRun the installer twice.",
        "data.csv": b"region,revenue\nNorth,100\n",
        "memo.docx": make_docx(["Board memo about the merger."]),
        "deck.pptx": make_pptx([("Plan", "Expand to Asia")]),
    }
    ids = [upload(client, h, name, data)["items"][0]["document"]["id"] for name, data in files.items()]
    docs = [wait_for_document(client, h, i) for i in ids]
    assert all(d["status"] == "ready" and d["chunk_count"] >= 1 for d in docs), docs
    listed = client.get("/api/v1/documents", headers=h).json()["items"]
    assert {d["filename"] for d in listed} == set(files)


def test_duplicate_upload_is_idempotent(client, fake_embedder):
    h = auth_headers(client)
    first = upload(client, h, "a.txt", ACME_REPORT.encode())["items"][0]
    wait_for_document(client, h, first["document"]["id"])
    embedded = fake_embedder.texts_embedded
    second = upload(client, h, "renamed-copy.txt", ACME_REPORT.encode())["items"][0]
    assert second["duplicate"] is True
    assert second["document"]["id"] == first["document"]["id"]
    assert len(client.get("/api/v1/documents", headers=h).json()["items"]) == 1
    client.app.state.container.worker.wait_idle()
    assert fake_embedder.texts_embedded == embedded  # nothing re-embedded


def test_same_content_for_different_users_is_separate(client):
    a, b = auth_headers(client, "a@example.com"), auth_headers(client, "b@example.com")
    da = upload(client, a, "x.txt", b"shared content here")["items"][0]
    db = upload(client, b, "x.txt", b"shared content here")["items"][0]
    assert not da["duplicate"] and not db["duplicate"]
    assert da["document"]["id"] != db["document"]["id"]


def test_path_traversal_filename_is_neutralised(client, settings):
    h = auth_headers(client)
    doc = upload(client, h, "../../../evil.txt", b"hello world")["items"][0]["document"]
    assert doc["filename"] == "evil.txt"
    root = Path(settings.upload_dir).resolve()
    files = stored_files(settings)
    assert len(files) == 1 and files[0].resolve().is_relative_to(root)
    assert files[0].name != "evil.txt"  # server-generated storage name
    assert not (root.parent / "evil.txt").exists()


def test_unsupported_and_spoofed_files_are_rejected(client, settings):
    h = auth_headers(client)
    r = client.post("/api/v1/documents", headers=h, files=[("files", ("run.exe", b"MZ\x90\x00"))])
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_media_type"
    r = client.post("/api/v1/documents", headers=h, files=[("files", ("fake.pdf", b"<html>not a pdf</html>"))])
    assert r.status_code == 415
    r = client.post("/api/v1/documents", headers=h, files=[("files", ("empty.txt", b""))])
    assert r.status_code == 422
    assert stored_files(settings) == []  # rejected uploads leave nothing behind


def test_upload_size_limits(tmp_path):
    settings = make_settings(tmp_path, max_upload_mb=1, max_files_per_upload=2)
    app = create_app(settings, embedder=FakeEmbedder(), llm_transport=FakeOpenRouter().transport())
    with TestClient(app) as c:
        h = auth_headers(c)
        big = b"a " * (600 * 1024)  # 1.2 MB
        r = c.post("/api/v1/documents", headers=h, files=[("files", ("big.txt", big))])
        assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
        huge = b"a " * (2 * 1024 * 1024)  # over the whole-request limit: rejected by middleware
        r = c.post("/api/v1/documents", headers=h, files=[("files", ("huge.txt", huge))])
        assert r.status_code == 413
        r = c.post("/api/v1/documents", headers=h,
                   files=[("files", (f"{i}.txt", b"x")) for i in range(3)])
        assert r.status_code == 422
        assert stored_files(settings) == []


def test_corrupted_document_fails_with_useful_error(client):
    h = auth_headers(client)
    corrupt = b"PK\x03\x04" + b"\x00" * 100  # zip magic, broken container
    r = client.post("/api/v1/documents", headers=h, files=[("files", ("broken.docx", corrupt))])
    assert r.status_code == 422 and "corrupted" in r.json()["error"]["message"]


def test_document_without_text_is_marked_failed(client):
    h = auth_headers(client)
    doc = upload(client, h, "blank.txt", b"   \n\n  \t ")["items"][0]["document"]
    final = wait_for_document(client, h, doc["id"])
    assert final["status"] == "failed" and "No extractable text" in final["error"]
    # Failed documents can be retried.
    r = client.post(f"/api/v1/documents/{doc['id']}/reprocess", headers=h)
    assert r.status_code == 202
    assert wait_for_document(client, h, doc["id"])["status"] == "failed"


def test_delete_removes_chunks_vectors_jobs_and_file(client, settings):
    h = auth_headers(client)
    doc = upload(client, h, "a.txt", ACME_REPORT.encode())["items"][0]["document"]
    wait_for_document(client, h, doc["id"])
    db = client.app.state.container.db
    with db.transaction() as s:
        assert s.query(Chunk).filter_by(document_id=doc["id"]).count() >= 1
    assert len(stored_files(settings)) == 1
    assert client.delete(f"/api/v1/documents/{doc['id']}", headers=h).status_code == 204
    with db.transaction() as s:
        assert s.query(Chunk).count() == 0
        assert s.query(IngestionJob).count() == 0
    assert stored_files(settings) == []
    assert client.delete(f"/api/v1/documents/{doc['id']}", headers=h).status_code == 404


def test_reingestion_is_idempotent(client):
    h = auth_headers(client)
    doc = upload(client, h, "a.txt", ACME_REPORT.encode())["items"][0]["document"]
    first = wait_for_document(client, h, doc["id"])
    container = client.app.state.container
    with container.db.transaction() as s:
        ids_before = sorted(c.id for c in s.query(Chunk).all())
        job = IngestionJob(document_id=doc["id"])
        s.add(job)
        s.flush()
        job_id = job.id
    container.ingestion.process(job_id)
    with container.db.transaction() as s:
        assert sorted(c.id for c in s.query(Chunk).all()) == ids_before
    assert client.get(f"/api/v1/documents/{doc['id']}", headers=h).json()["chunk_count"] == first["chunk_count"]
