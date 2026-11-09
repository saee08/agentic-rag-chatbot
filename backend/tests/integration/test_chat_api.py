from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import ask, auth_headers, make_settings, new_conversation, upload, wait_for_document
from tests.helpers import ACME_REPORT, FakeEmbedder, FakeOpenRouter


def ready_user(client) -> dict:
    h = auth_headers(client)
    doc = upload(client, h, "acme.txt", ACME_REPORT.encode())["items"][0]["document"]
    assert wait_for_document(client, h, doc["id"])["status"] == "ready"
    return h


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_conversation_history_is_persisted(client):
    h = ready_user(client)
    conv = new_conversation(client, h)
    ask(client, h, conv, "Which factories were opened?")
    ask(client, h, conv, "Who is the chief executive?")
    detail = client.get(f"/api/v1/conversations/{conv}", headers=h).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "user", "assistant"]
    assert detail["title"] == "Which factories were opened?"
    assert "Dana Whitfield" in detail["messages"][3]["content"]
    assert detail["messages"][3]["usage"]["total_tokens"] > 0
    listed = client.get("/api/v1/conversations", headers=h).json()["items"]
    assert [c["id"] for c in listed] == [conv]


def test_follow_up_questions_receive_history(client, fake_llm):
    h = ready_user(client)
    conv = new_conversation(client, h)
    ask(client, h, conv, "Which factories were opened?")
    ask(client, h, conv, "Who is the chief executive?")
    last_answer_prompt = [r for r in fake_llm.requests if fake_llm._kind(r["messages"]) == "answer"][-1]
    assert "<conversation>\nuser: Which factories were opened?" in last_answer_prompt["messages"][1]["content"]


def test_streaming_endpoint_emits_steps_tokens_and_final(client):
    h = ready_user(client)
    conv = new_conversation(client, h)
    with client.stream("POST", f"/api/v1/conversations/{conv}/messages/stream", headers=h,
                       json={"content": "Which factories were opened?"}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r.read().decode())
    kinds = [e for e, _ in events]
    assert kinds[0] == "user_message" and kinds[-1] == "final"
    assert "step" in kinds and "token" in kinds
    streamed = "".join(d["delta"] for e, d in events if e == "token")
    final = events[-1][1]
    assert final["status"] == "answered" and final["citations"]
    assert streamed.strip() == final["content"]
    # The streamed message is persisted like a normal one.
    stored = client.get(f"/api/v1/conversations/{conv}", headers=h).json()["messages"]
    assert stored[-1]["id"] == final["id"]


def test_llm_outage_returns_503_and_records_error(tmp_path):
    llm = FakeOpenRouter(fail_with=[503, 503, 503])
    app = create_app(make_settings(tmp_path, retrieval_strong_score=0.2), embedder=FakeEmbedder(),
                     llm_transport=llm.transport())
    with TestClient(app) as c:
        h = ready_user(c)
        conv = new_conversation(c, h)
        r = c.post(f"/api/v1/conversations/{conv}/messages", headers=h,
                   json={"content": "key performance indicators customer retention"})
        assert r.status_code == 503
        body = r.json()["error"]
        assert body["code"] == "service_unavailable" and body["request_id"]
        messages = c.get(f"/api/v1/conversations/{conv}", headers=h).json()["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"] and messages[1]["status"] == "error"


def test_llm_outage_in_stream_emits_error_event(tmp_path):
    llm = FakeOpenRouter(fail_with=[503, 503, 503])
    app = create_app(make_settings(tmp_path, retrieval_strong_score=0.2), embedder=FakeEmbedder(),
                     llm_transport=llm.transport())
    with TestClient(app) as c:
        h = ready_user(c)
        conv = new_conversation(c, h)
        with c.stream("POST", f"/api/v1/conversations/{conv}/messages/stream", headers=h,
                      json={"content": "key performance indicators customer retention"}) as r:
            events = parse_sse(r.read().decode())
        assert events[-1][0] == "error" and events[-1][1]["code"] == "service_unavailable"


def test_missing_api_key_gives_clear_error(tmp_path):
    app = create_app(make_settings(tmp_path, openrouter_api_key=None, retrieval_strong_score=0.2),
                     embedder=FakeEmbedder(), llm_transport=FakeOpenRouter().transport())
    with TestClient(app) as c:
        h = ready_user(c)
        conv = new_conversation(c, h)
        r = c.post(f"/api/v1/conversations/{conv}/messages", headers=h,
                   json={"content": "key performance indicators customer retention"})
        assert r.status_code == 503 and "OPENROUTER_API_KEY" in r.json()["error"]["message"]


def test_message_validation(client):
    h = auth_headers(client)
    conv = new_conversation(client, h)
    for bad in ({"content": ""}, {"content": "   "}, {"content": "x" * 4001}, {}):
        r = client.post(f"/api/v1/conversations/{conv}/messages", headers=h, json=bad)
        assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    r = client.post("/api/v1/conversations/does-not-exist/messages", headers=h, json={"content": "hi"})
    assert r.status_code == 404


def test_chat_is_rate_limited(tmp_path):
    app = create_app(make_settings(tmp_path, rate_limit_chat_per_minute=2), embedder=FakeEmbedder(),
                     llm_transport=FakeOpenRouter().transport())
    with TestClient(app) as c:
        h = auth_headers(c)
        conv = new_conversation(c, h)
        codes = [c.post(f"/api/v1/conversations/{conv}/messages", headers=h, json={"content": "hi"}).status_code
                 for _ in range(3)]
        assert codes == [200, 200, 429]


def test_delete_conversation(client):
    h = auth_headers(client)
    conv = new_conversation(client, h)
    ask(client, h, conv, "hello")
    assert client.delete(f"/api/v1/conversations/{conv}", headers=h).status_code == 204
    assert client.get(f"/api/v1/conversations/{conv}", headers=h).status_code == 404
