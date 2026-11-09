"""The single place where stored messages become API responses.

Stored JSON columns (citations, agent_trace, usage) may come from older versions of the app with
different field names; everything is normalised here so the API always returns the documented
``MessageOut`` contract: no missing keys, no nulls where a value is required, and only
user-safe processing steps (never prompts or model reasoning).
"""

from __future__ import annotations

from typing import Any

from app.models import Message, MessageRole
from app.rag.excerpts import relevant_excerpt
from app.schemas import ErrorInfo, MessageMetadata, MessageOut, RetrievalStats, SourceOut, StepOut, TokenUsage

ERROR_CODES: dict[str, tuple[bool, str]] = {
    # code: (retryable, fallback message)
    "llm_not_configured": (False, "The language model is not configured."),
    "llm_rate_limited": (True, "The language model is rate-limited. Please try again shortly."),
    "llm_unavailable": (True, "The language model service is temporarily unavailable."),
    "llm_bad_response": (True, "The language model returned an invalid response."),
    "internal_error": (True, "Something went wrong while answering."),
}
_ANSWER_STATUSES = {"answered", "no_answer", "no_documents", "direct_reply", "error"}

_STOP_LABELS = {
    "sufficient_evidence": "Found enough relevant evidence",
    "insufficient_evidence": "No relevant information found in your documents",
    "no_ready_documents": "No processed documents available",
    "max_tool_calls": "Reached the search limit",
    "max_iterations": "Reached the step limit",
    "time_budget": "Reached the time limit",
    "token_budget": "Reached the processing budget",
}


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def build_source(raw: dict[str, Any], question: str, answer: str) -> SourceOut | None:
    """Normalise one stored citation (current or legacy shape). Returns None when the source
    cannot be identified, so it is never displayed."""
    number = _int(raw.get("number"))
    name = _str(raw.get("document_name"))
    content = _str(raw.get("content")) or _str(raw.get("snippet"))
    if number is None or name is None or content is None:
        return None
    page = _int(raw.get("page_number"))
    if page is None:
        page = _int(raw.get("page"))
    excerpt = _str(raw.get("excerpt")) or relevant_excerpt(content, question, answer)
    return SourceOut(number=number, document_id=_str(raw.get("document_id")), document_name=name,
                     page_number=page, chunk_id=_str(raw.get("chunk_id")), excerpt=excerpt, content=content,
                     score=_num(raw.get("score")), match=_str(raw.get("match")))


def build_steps(trace: Any) -> list[StepOut]:
    steps: list[StepOut] = []
    for raw in trace if isinstance(trace, list) else []:
        if not isinstance(raw, dict):
            continue
        kind, name = raw.get("kind"), raw.get("name")
        detail = raw.get("detail") if isinstance(raw.get("detail"), dict) else {}
        if kind == "tool" and name == "search_documents":
            hits = _int(detail.get("hits")) or 0
            query = _str((detail.get("args") or {}).get("query")) if isinstance(detail.get("args"), dict) else None
            steps.append(StepOut(kind="search", label="Searched your documents",
                                 detail=(f"“{query}” · " if query else "")
                                 + f"{hits} relevant passage{'' if hits == 1 else 's'}"))
        elif kind == "tool" and name == "list_documents":
            steps.append(StepOut(kind="library", label="Checked your document library"))
        elif kind == "decision":
            action = detail.get("action")
            if action == "search_documents":
                steps.append(StepOut(kind="assess", label="Evidence was not sufficient; refining the search"))
            elif action == "answer":
                steps.append(StepOut(kind="assess", label="Assessed the retrieved evidence"))
            elif action == "direct_reply":
                steps.append(StepOut(kind="assess", label="Recognised a conversational message"))
        elif kind == "generation":
            sources = _int(detail.get("sources")) or 0
            label = ("Wrote a reply" if name == "direct_reply"
                     else f"Generated the answer from {sources} passage{'' if sources == 1 else 's'}")
            steps.append(StepOut(kind="generate", label=label))
            calcs, fixed = _int(detail.get("calculations")) or 0, _int(detail.get("corrected_calculations")) or 0
            if calcs or fixed:
                steps.append(StepOut(kind="assess", label="Checked arithmetic exactly",
                                     detail=f"{calcs + fixed} calculation{'' if calcs + fixed == 1 else 's'}"
                                     + (f", {fixed} corrected" if fixed else "")))
        elif kind == "stop" and name in _STOP_LABELS:
            steps.append(StepOut(kind="stop", label=_STOP_LABELS[name]))
    return steps


def to_message_out(message: Message, question: str = "") -> MessageOut:
    role = message.role.value if isinstance(message.role, MessageRole) else str(message.role)
    if role == "user":
        return MessageOut(id=message.id, role="user", status="sent", content=message.content or "",
                          created_at=message.created_at)

    status = message.status if message.status in _ANSWER_STATUSES else "answered"
    content = message.content if isinstance(message.content, str) else ""
    error = None
    if status == "error" or not content.strip():
        code = message.error_code if message.error_code in ERROR_CODES else "internal_error"
        retryable, fallback = ERROR_CODES[code]
        error = ErrorInfo(code=code, message=_str(content) or fallback, retryable=retryable)
        status, content = "error", error.message
        return MessageOut(id=message.id, role="assistant", status=status, content=content, error=error,
                          created_at=message.created_at)

    sources = [s for s in (build_source(c, question, content) for c in (message.citations or [])
                           if isinstance(c, dict)) if s is not None]
    usage = message.usage if isinstance(message.usage, dict) else {}
    prompt, completion = _int(usage.get("prompt_tokens")), _int(usage.get("completion_tokens"))
    tokens = TokenUsage(prompt=prompt, completion=completion, total=prompt + completion) \
        if prompt is not None and completion is not None else None
    retrieval = RetrievalStats(
        passages_retrieved=_int(usage.get("passages_retrieved")),
        passages_cited=len(sources),
        documents=len({s.document_id or s.document_name for s in sources if s.chunk_id}),
        pages=len({(s.document_id or s.document_name, s.page_number) for s in sources if s.chunk_id}),
    )
    grounded = usage.get("grounded") if isinstance(usage.get("grounded"), bool) else None
    metadata = MessageMetadata(model=_str(usage.get("model")), latency_ms=_num(usage.get("elapsed_ms")),
                               llm_calls=_int(usage.get("llm_calls")), tokens=tokens, grounded=grounded,
                               retrieval=retrieval)
    return MessageOut(id=message.id, role="assistant", status=status, content=content, sources=sources,
                      metadata=metadata, steps=build_steps(message.agent_trace), created_at=message.created_at)


def to_message_outs(messages: list[Message]) -> list[MessageOut]:
    """Serialise a conversation; each reply's excerpts are chosen using the question before it."""
    out, question = [], ""
    for message in messages:
        out.append(to_message_out(message, question))
        if out[-1].role == "user":
            question = message.content or ""
    return out
