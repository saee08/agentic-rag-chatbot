"""End-to-end: upload -> background ingestion -> grounded, cited answer -> honest refusal ->
deletion -> isolation. LLM calls go to the scripted fake OpenRouter; nothing external is used."""

from __future__ import annotations

from tests.conftest import ask, auth_headers, new_conversation, upload, wait_for_document
from tests.helpers import ACME_REPORT, make_pdf


def test_full_document_question_answering_flow(client, fake_llm):
    alice = auth_headers(client, "alice@example.com")
    bob = auth_headers(client, "bob@example.com")

    # 1. Upload a PDF and wait for background ingestion.
    pdf = make_pdf([ACME_REPORT, "Appendix. The auditors were Brightline LLP."])
    result = upload(client, alice, "acme-report.pdf", pdf)["items"][0]
    assert result["duplicate"] is False
    doc = wait_for_document(client, alice, result["document"]["id"])
    assert doc["status"] == "ready", doc
    assert doc["page_count"] == 2 and doc["chunk_count"] >= 2

    # 2. Ask a question answered by the document.
    conv = new_conversation(client, alice)
    reply = ask(client, alice, conv, "What were the key performance indicators?")["assistant_message"]
    assert reply["status"] == "answered"
    assert "94 percent" in reply["content"]
    assert reply["sources"], "answer must cite its sources"
    citation = reply["sources"][0]
    assert citation["document_id"] == doc["id"]
    assert citation["document_name"] == "acme-report.pdf"
    assert citation["page_number"] == 1 and citation["number"] == 1
    assert "customer retention" in citation["content"] and "customer retention" in citation["excerpt"]
    assert f"[{citation['number']}]" in reply["content"]
    assert any(step["kind"] == "search" for step in reply["steps"])
    assert reply["metadata"]["retrieval"]["documents"] == 1 and reply["error"] is None

    # 3. An unrelated question must not produce an invented answer.
    calls_before = fake_llm.calls
    unrelated = ask(client, alice, conv, "What is the boiling point of tungsten?")["assistant_message"]
    assert unrelated["status"] == "no_answer"
    assert unrelated["sources"] == []
    assert "couldn't find" in unrelated["content"]
    assert "answer" not in fake_llm.kinds()[calls_before:], "no answer generation without evidence"

    # 4. Another user cannot see, search or delete the document.
    assert client.get(f"/api/v1/documents/{doc['id']}", headers=bob).status_code == 404
    assert client.delete(f"/api/v1/documents/{doc['id']}", headers=bob).status_code == 404
    assert client.get("/api/v1/documents", headers=bob).json()["items"] == []
    bob_conv = new_conversation(client, bob)
    bob_reply = ask(client, bob, bob_conv, "What were the key performance indicators?")["assistant_message"]
    assert bob_reply["status"] == "no_documents"
    assert "94" not in bob_reply["content"]
    assert client.get(f"/api/v1/conversations/{conv}", headers=bob).status_code == 404

    # 5. Deleting the document removes it and its vectors from retrieval.
    assert client.delete(f"/api/v1/documents/{doc['id']}", headers=alice).status_code == 204
    assert client.get(f"/api/v1/documents/{doc['id']}", headers=alice).status_code == 404
    after = ask(client, alice, conv, "What were the key performance indicators?")["assistant_message"]
    assert after["status"] == "no_documents"
