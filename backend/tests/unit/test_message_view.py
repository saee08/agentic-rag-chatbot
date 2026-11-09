"""The server-side normalisation layer: every stored message shape -> the MessageOut contract."""

from __future__ import annotations

from datetime import UTC, datetime

from app.models import Message, MessageRole
from app.services.message_view import build_steps, to_message_out, to_message_outs

NOW = datetime(2026, 11, 9, tzinfo=UTC)


def msg(**kw) -> Message:
    values = {"id": "m1", "conversation_id": "c1", "role": MessageRole.ASSISTANT, "content": "Pune [1].",
              "status": "answered", "created_at": NOW}
    values.update(kw)
    return Message(**values)


SOURCE = {"number": 1, "document_id": "d1", "document_name": "case.pdf", "page_number": 1, "chunk_id": "c1",
          "score": 0.63, "match": "semantic", "excerpt": "The DC is located in Pune.",
          "content": "Company overview. The DC is located in Pune. Stores are in western India."}


def test_valid_answer_has_complete_contract():
    out = to_message_out(msg(citations=[SOURCE], usage={"prompt_tokens": 10, "completion_tokens": 5,
                                                        "llm_calls": 2, "elapsed_ms": 812.5, "grounded": True,
                                                        "model": "m/x", "passages_retrieved": 4}),
                         "Where is the DC?").model_dump()
    assert out["status"] == "answered" and out["content"] == "Pune [1]." and out["error"] is None
    assert out["sources"][0]["page_number"] == 1 and out["sources"][0]["excerpt"] == "The DC is located in Pune."
    meta = out["metadata"]
    assert meta["tokens"] == {"prompt": 10, "completion": 5, "total": 15}
    assert meta["retrieval"] == {"passages_retrieved": 4, "passages_cited": 1, "documents": 1, "pages": 1}
    assert meta["model"] == "m/x" and meta["latency_ms"] == 812.5 and meta["grounded"] is True


def test_legacy_citation_shape_is_normalised():
    legacy = {"number": 1, "document_id": "d1", "document_name": "case.pdf", "page": 2, "chunk_id": "c1",
              "score": 0.5, "snippet": "Bright Appliances ships NH-205 from Bengaluru with a 14 day lead time."}
    out = to_message_out(msg(citations=[legacy], usage={"prompt_tokens": 1}), "supplier?")
    source = out.sources[0]
    assert source.page_number == 2 and source.content.startswith("Bright Appliances")
    assert source.excerpt  # derived when the stored record has none
    assert out.metadata.tokens is None  # incomplete usage is reported as unknown, not as zeros
    assert out.metadata.retrieval.passages_retrieved is None


def test_missing_null_and_empty_fields():
    for kwargs in ({"citations": None, "usage": None, "agent_trace": None}, {"citations": [], "usage": {}}):
        out = to_message_out(msg(**kwargs))
        assert out.sources == [] and out.steps == []
        assert out.metadata.retrieval.passages_cited == 0
        assert "null" not in out.model_dump_json().replace("null,", "").replace("null}", "")


def test_unidentifiable_sources_are_dropped():
    bad = [{"number": 2, "document_name": "", "content": "x"}, {"document_name": "a.pdf", "content": "x"},
           {"number": 3, "document_name": "a.pdf"}, "not a dict", None]
    out = to_message_out(msg(citations=[SOURCE, *bad]))
    assert [s.number for s in out.sources] == [1]


def test_missing_or_null_answer_becomes_explicit_error():
    for content in ("", "   ", None):
        out = to_message_out(msg(content=content, status="answered"))
        assert out.status == "error" and out.error.code == "internal_error" and out.error.retryable
        assert out.content == out.error.message and out.content


def test_stored_error_maps_code_and_retryability():
    out = to_message_out(msg(content="The model was not found.", status="error", error_code="llm_not_configured"))
    assert out.error.model_dump() == {"code": "llm_not_configured", "message": "The model was not found.",
                                      "retryable": False}
    unknown = to_message_out(msg(content="boom", status="error", error_code="weird"))
    assert unknown.error.code == "internal_error"


def test_unknown_status_is_not_passed_through():
    assert to_message_out(msg(status="something_new")).status == "answered"


def test_user_messages_are_minimal():
    out = to_message_out(msg(role=MessageRole.USER, content="Where?", status=None)).model_dump()
    assert out["status"] == "sent" and out["sources"] == [] and out["metadata"] is None and out["error"] is None


def test_steps_are_user_safe_summaries():
    trace = [
        {"kind": "tool", "name": "search_documents", "detail": {"args": {"query": "dc location"}, "hits": 3}},
        {"kind": "decision", "name": "controller", "detail": {"action": "answer", "tokens": 99}},
        {"kind": "generation", "name": "answer", "detail": {"sources": 3, "tokens": 400}},
        {"kind": "stop", "name": "unexpected_internal_reason", "detail": {}},
        {"kind": "mystery"}, "junk", None,
    ]
    steps = [s.model_dump() for s in build_steps(trace)]
    assert steps == [
        {"kind": "search", "label": "Searched your documents", "detail": "“dc location” · 3 relevant passages"},
        {"kind": "assess", "label": "Assessed the retrieved evidence", "detail": None},
        {"kind": "generate", "label": "Generated the answer from 3 passages", "detail": None},
    ]
    assert "tokens" not in str(steps)


def test_excerpts_use_the_preceding_question():
    user = msg(id="u", role=MessageRole.USER, content="Which supplier ships NH-205?", status=None)
    legacy = {**SOURCE, "excerpt": None, "content": " ".join(["Filler sentence about stores."] * 12)
              + " Bright Appliances ships NH-205 from Bengaluru. " + " ".join(["More filler text here."] * 12)}
    out = to_message_outs([user, msg(citations=[legacy], content="Bright Appliances [1].")])
    assert "Bright Appliances ships NH-205" in out[1].sources[0].excerpt
