from __future__ import annotations

import json

import httpx
import pytest

from app.rag.llm import (
    LLMBadResponseError,
    LLMClient,
    LLMNotConfiguredError,
    LLMRateLimitedError,
    LLMUnavailableError,
)
from tests.conftest import make_settings

MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "hello"}]


def client_with(handler, tmp_path, **overrides) -> LLMClient:
    return LLMClient(make_settings(tmp_path, **overrides), transport=httpx.MockTransport(handler))


def ok(text="hi", usage=True):
    body = {"model": "m", "choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
    if usage:
        body["usage"] = {"prompt_tokens": 10, "completion_tokens": 2}
    return httpx.Response(200, json=body)


def test_success_parses_text_and_usage(tmp_path):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return ok(" answer ")

    result = client_with(handler, tmp_path, llm_max_tokens=123).chat(MSGS)
    assert result.text == "answer"
    assert result.usage.total_tokens == 12 and not result.usage_estimated
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["max_tokens"] == 123 and seen["body"]["stream"] is False


def test_missing_usage_is_estimated(tmp_path):
    result = client_with(lambda r: ok("some text", usage=False), tmp_path).chat(MSGS)
    assert result.usage_estimated and result.usage.total_tokens > 0


def test_retries_on_429_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr("app.rag.llm.time.sleep", lambda s: None)
    responses = [httpx.Response(429, headers={"Retry-After": "1"}), httpx.Response(503), ok("done")]
    calls = []

    def handler(request):
        calls.append(1)
        return responses.pop(0)

    assert client_with(handler, tmp_path, llm_max_retries=2).chat(MSGS).text == "done"
    assert len(calls) == 3


def test_retry_after_is_honoured_and_capped(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr("app.rag.llm.time.sleep", sleeps.append)
    responses = [httpx.Response(429, headers={"Retry-After": "3"}), httpx.Response(429, headers={"Retry-After": "999"}),
                 ok()]
    client_with(lambda r: responses.pop(0), tmp_path, llm_max_retries=2).chat(MSGS)
    assert sleeps == [3.0, 20.0]


def test_rate_limit_exhausts_retries(tmp_path, monkeypatch):
    monkeypatch.setattr("app.rag.llm.time.sleep", lambda s: None)
    with pytest.raises(LLMRateLimitedError):
        client_with(lambda r: httpx.Response(429), tmp_path, llm_max_retries=1).chat(MSGS)


def test_server_error_exhausts_retries(tmp_path, monkeypatch):
    monkeypatch.setattr("app.rag.llm.time.sleep", lambda s: None)
    with pytest.raises(LLMUnavailableError):
        client_with(lambda r: httpx.Response(500), tmp_path, llm_max_retries=1).chat(MSGS)


def test_timeout_is_retried_then_reported(tmp_path, monkeypatch):
    monkeypatch.setattr("app.rag.llm.time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMUnavailableError, match="timed out"):
        client_with(handler, tmp_path, llm_max_retries=2).chat(MSGS)
    assert len(calls) == 3


def test_auth_errors_are_not_retried(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401)

    with pytest.raises(LLMNotConfiguredError):
        client_with(handler, tmp_path).chat(MSGS)
    assert len(calls) == 1


@pytest.mark.parametrize("body", [b"not json", b'{"choices": []}', b'{"choices": [{"message": {}}]}', b"{}"])
def test_malformed_responses(tmp_path, body):
    with pytest.raises(LLMBadResponseError):
        client_with(lambda r: httpx.Response(200, content=body), tmp_path).chat(MSGS)


def test_not_configured_makes_no_request(tmp_path):
    calls = []
    client = client_with(lambda r: calls.append(1) or ok(), tmp_path, openrouter_api_key=None)
    assert not client.configured
    with pytest.raises(LLMNotConfiguredError):
        client.chat(MSGS)
    assert calls == []


def test_streaming_yields_deltas_and_usage(tmp_path):
    def handler(request):
        assert json.loads(request.content)["stream"] is True
        lines = [": OPENROUTER PROCESSING", "",
                 'data: {"choices":[{"delta":{"content":"Hel"}}]}', "",
                 'data: {"choices":[{"delta":{"content":"lo"}}]}', "",
                 'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                 '"usage":{"prompt_tokens":5,"completion_tokens":1}}',
                 "", "data: [DONE]", ""]
        return httpx.Response(200, content="\n".join(lines).encode())

    events = list(client_with(handler, tmp_path).stream_chat(MSGS))
    assert [e.delta for e in events if not e.done] == ["Hel", "lo"]
    final = events[-1]
    assert final.done and final.result.text == "Hello"
    assert final.result.usage.total_tokens == 6 and final.result.finish_reason == "stop"


def test_streaming_error_chunk(tmp_path):
    body = 'data: {"error": {"message": "overloaded"}}\n\n'
    with pytest.raises(LLMUnavailableError):
        list(client_with(lambda r: httpx.Response(200, content=body.encode()), tmp_path).stream_chat(MSGS))


def test_expired_deadline_fails_fast(tmp_path):
    import time

    with pytest.raises(LLMUnavailableError, match="ran out of time"):
        client_with(lambda r: ok(), tmp_path).chat(MSGS, deadline=time.monotonic() - 1)
