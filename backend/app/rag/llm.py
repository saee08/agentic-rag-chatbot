"""OpenAI-compatible chat-completions client (OpenRouter by default).

* Explicit connect/read timeouts on every request.
* Bounded retries with exponential backoff + jitter for 429, 5xx and network errors, honouring
  ``Retry-After``. Streaming requests are only retried before the first token arrives.
* Responses are validated; malformed payloads raise ``LLMBadResponseError``.
* The API key is sent only in the Authorization header and never logged.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.logging import log_event

logger = logging.getLogger(__name__)

_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_MAX_BACKOFF_SECONDS = 20.0


class LLMError(Exception):
    """Base class. Messages are safe to show to end users."""


class LLMNotConfiguredError(LLMError):
    pass


class LLMRateLimitedError(LLMError):
    pass


class LLMUnavailableError(LLMError):
    pass


class LLMBadResponseError(LLMError):
    pass


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens

    def as_dict(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens}


@dataclass
class LLMResult:
    text: str
    usage: Usage
    model: str
    latency_ms: float
    finish_reason: str | None = None
    usage_estimated: bool = False


@dataclass
class StreamEvent:
    delta: str = ""
    done: bool = False
    result: LLMResult | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class _Msg(BaseModel):
    content: str | None = None


class _Choice(BaseModel):
    message: _Msg
    finish_reason: str | None = None


class _UsageModel(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class _Completion(BaseModel):
    model: str | None = None
    choices: list[_Choice]
    usage: _UsageModel | None = None


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used when the provider omits usage."""
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(m.get("content", "")) + 4 for m in messages)


class LLMClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.model = settings.llm_model
        self._client = httpx.Client(
            base_url=settings.openrouter_base_url.rstrip("/"),
            timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=settings.llm_connect_timeout_seconds),
            transport=transport,
        )

    @property
    def configured(self) -> bool:
        return self.settings.llm_configured

    def close(self) -> None:
        self._client.close()

    def _headers(self) -> dict[str, str]:
        if not self.configured:
            raise LLMNotConfiguredError("The language model is not configured (OPENROUTER_API_KEY is not set).")
        assert self.settings.openrouter_api_key is not None
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-Title": self.settings.llm_app_name,
        }
        if self.settings.llm_app_url:
            headers["HTTP-Referer"] = self.settings.llm_app_url
        return headers

    def _payload(self, messages: list[dict[str, str]], max_tokens: int | None, temperature: float | None,
                 stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
            "stream": stream,
        }
        if stream:
            payload["usage"] = {"include": True}  # OpenRouter: usage in the final stream chunk
        return payload

    @staticmethod
    def _retry_delay(attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            header = response.headers.get("Retry-After")
            if header:
                try:
                    return min(float(header), _MAX_BACKOFF_SECONDS)
                except ValueError:
                    pass
        return min(_MAX_BACKOFF_SECONDS, 0.5 * 2**attempt) * (0.5 + random.random() / 2)  # noqa: S311

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        if status == 429:
            raise LLMRateLimitedError("The language model is rate-limited. Please try again shortly.")
        if status in (401, 403):
            raise LLMNotConfiguredError("The language model rejected the API credentials.")
        if status >= 500 or status in _RETRY_STATUS:
            raise LLMUnavailableError("The language model service is temporarily unavailable.")
        raise LLMBadResponseError(f"The language model request was rejected (HTTP {status}).")

    def _send(self, payload: dict[str, Any], stream: bool, deadline: float | None) -> httpx.Response:
        headers = self._headers()
        attempts = self.settings.llm_max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            response: httpx.Response | None = None
            try:
                timeout = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 1:
                        raise LLMUnavailableError("The request ran out of time.")
                    timeout = httpx.Timeout(min(self.settings.llm_timeout_seconds, remaining),
                                            connect=self.settings.llm_connect_timeout_seconds)
                request = self._client.build_request(
                    "POST", "/chat/completions", json=payload, headers=headers,
                    **({"timeout": timeout} if timeout else {}),
                )
                response = self._client.send(request, stream=stream)
                if response.status_code in _RETRY_STATUS and attempt < attempts - 1:
                    if stream:
                        response.close()
                    delay = self._retry_delay(attempt, response)
                    log_event(logger, "llm retry", logging.WARNING, status=response.status_code,
                              attempt=attempt + 1, delay_s=round(delay, 2))
                    time.sleep(delay)
                    continue
                if response.status_code >= 400:
                    if stream:
                        response.read()
                        response.close()
                    self._raise_for_status(response)
                return response
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    delay = self._retry_delay(attempt, None)
                    log_event(logger, "llm retry", logging.WARNING, error=type(exc).__name__,
                              attempt=attempt + 1, delay_s=round(delay, 2))
                    time.sleep(delay)
                    continue
        if isinstance(last_exc, httpx.TimeoutException):
            raise LLMUnavailableError("The language model timed out.") from last_exc
        raise LLMUnavailableError("Could not reach the language model service.") from last_exc

    def chat(self, messages: list[dict[str, str]], *, max_tokens: int | None = None,
             temperature: float | None = None, deadline: float | None = None) -> LLMResult:
        start = time.perf_counter()
        response = self._send(self._payload(messages, max_tokens, temperature, False), False, deadline)
        try:
            data = _Completion.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise LLMBadResponseError("The language model returned a malformed response.") from exc
        if not data.choices or data.choices[0].message.content is None:
            raise LLMBadResponseError("The language model returned an empty response.")
        text = data.choices[0].message.content.strip()
        if data.usage:
            usage, estimated = Usage(data.usage.prompt_tokens, data.usage.completion_tokens), False
        else:
            usage, estimated = Usage(estimate_messages_tokens(messages), estimate_tokens(text)), True
        result = LLMResult(text, usage, data.model or self.model, round((time.perf_counter() - start) * 1000, 2),
                           data.choices[0].finish_reason, estimated)
        log_event(logger, "llm call", model=result.model, latency_ms=result.latency_ms,
                  usage_estimated=estimated, finish_reason=result.finish_reason, **usage.as_dict())
        return result

    def stream_chat(self, messages: list[dict[str, str]], *, max_tokens: int | None = None,
                    temperature: float | None = None, deadline: float | None = None) -> Iterator[StreamEvent]:
        start = time.perf_counter()
        response = self._send(self._payload(messages, max_tokens, temperature, True), True, deadline)
        parts: list[str] = []
        usage: Usage | None = None
        model = self.model
        finish_reason: str | None = None
        try:
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue  # SSE comments (": OPENROUTER PROCESSING") and blank keep-alives
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise LLMBadResponseError("The language model returned a malformed stream.") from exc
                if "error" in chunk:
                    raise LLMUnavailableError("The language model reported an error while streaming.")
                model = chunk.get("model") or model
                if chunk.get("usage"):
                    u = chunk["usage"]
                    usage = Usage(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)))
                for choice in chunk.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content") or ""
                    finish_reason = choice.get("finish_reason") or finish_reason
                    if delta:
                        parts.append(delta)
                        yield StreamEvent(delta=delta)
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("The language model timed out while streaming.") from exc
        except httpx.TransportError as exc:
            raise LLMUnavailableError("The connection to the language model was interrupted.") from exc
        finally:
            response.close()
        text = "".join(parts).strip()
        estimated = usage is None
        if usage is None:
            usage = Usage(estimate_messages_tokens(messages), estimate_tokens(text))
        result = LLMResult(text, usage, model, round((time.perf_counter() - start) * 1000, 2), finish_reason, estimated)
        log_event(logger, "llm stream", model=model, latency_ms=result.latency_ms, usage_estimated=estimated,
                  finish_reason=finish_reason, **usage.as_dict())
        yield StreamEvent(done=True, result=result)
