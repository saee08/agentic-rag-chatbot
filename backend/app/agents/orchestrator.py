"""Bounded agent loop for document question answering.

    1. Gate: no ready documents -> answer immediately, no LLM call.
    2. Act: search the documents with the user's question (fast path, no planning call).
    3. Evaluate sufficiency: strong hits -> go straight to answering. Otherwise ask the LLM
       controller for the next action: refine the search, list documents, answer, give up, or
       reply conversationally. Controller output is validated JSON; invalid output falls back
       to a safe deterministic decision.
    4. Generate a cited answer from the gathered evidence; verify citations; map the model's
       INSUFFICIENT_EVIDENCE signal to an honest "not found" reply.

Hard limits (all configurable): iterations, tool calls, wall-clock timeout and token budget.
The loop is a ``for`` over ``max_iterations`` so it cannot run unbounded, and every exit path
is recorded in the trace.

The runner is a generator of ``AgentEvent``s so the same code serves the JSON and the
streaming (SSE) endpoints.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select

from app.agents.tools import (
    ListDocumentsOutput,
    SearchDocumentsOutput,
    ToolContext,
    ToolRegistry,
    ToolResult,
)
from app.core.logging import log_event
from app.models import Document, DocumentStatus
from app.rag import prompts
from app.rag.generator import check_answer, is_insufficient
from app.rag.llm import LLMClient, LLMResult, Usage, estimate_messages_tokens
from app.rag.prompts import Evidence

logger = logging.getLogger(__name__)

AgentStatus = Literal["answered", "no_answer", "no_documents", "direct_reply"]

_MAX_EVIDENCE = 8
_CONTROLLER_MAX_TOKENS = 200


@dataclass(frozen=True)
class AgentLimits:
    max_iterations: int = 4
    max_tool_calls: int = 4
    timeout_seconds: float = 90.0
    token_budget: int = 12_000
    answer_max_tokens: int = 800
    strong_score: float = 0.55


@dataclass
class AgentStep:
    kind: Literal["tool", "decision", "generation", "stop"]
    name: str
    detail: dict[str, Any]
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "detail": self.detail, "duration_ms": self.duration_ms}


@dataclass
class AgentOutcome:
    status: AgentStatus
    answer: str
    citations: list[Evidence]
    grounded: bool
    trace: list[AgentStep]
    usage: Usage
    llm_calls: int
    elapsed_ms: float


@dataclass
class AgentEvent:
    type: Literal["step", "token", "final"]
    step: AgentStep | None = None
    delta: str = ""
    outcome: AgentOutcome | None = None


# --- controller decisions ----------------------------------------------------------------------


class _Decision(BaseModel):
    action: Literal["search_documents", "list_documents", "answer", "cannot_answer", "direct_reply"]
    query: str | None = Field(default=None, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=10)


def parse_decision(text: str) -> _Decision | None:
    """Extract and validate the first JSON object in the model output."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        decision = _Decision.model_validate(json.loads(text[start : i + 1]))
                    except (json.JSONDecodeError, ValidationError):
                        break
                    if decision.action == "search_documents" and not (decision.query or "").strip():
                        return None
                    return decision
        start = text.find("{", start + 1)
    return None


# --- run state ---------------------------------------------------------------------------------


@dataclass
class _State:
    started: float
    deadline: float
    usage: Usage = field(default_factory=Usage)
    llm_calls: int = 0
    tool_calls: int = 0
    trace: list[AgentStep] = field(default_factory=list)
    evidence: dict[str, Evidence] = field(default_factory=dict)  # chunk_id -> evidence
    library: ListDocumentsOutput | None = None
    observations: list[str] = field(default_factory=list)
    tried_queries: set[str] = field(default_factory=set)

    def add_hits(self, output: SearchDocumentsOutput) -> int:
        added = 0
        for hit in output.hits:
            existing = self.evidence.get(hit.chunk_id)
            if existing is None:
                self.evidence[hit.chunk_id] = Evidence(0, hit.text, hit.document_id, hit.document_name,
                                                       hit.page, hit.chunk_id, hit.score)
                added += 1
            elif (existing.score or 0) < hit.score:
                self.evidence[hit.chunk_id] = Evidence(0, hit.text, hit.document_id, hit.document_name,
                                                       hit.page, hit.chunk_id, hit.score)
        return added

    def numbered_evidence(self) -> list[Evidence]:
        ranked = sorted(self.evidence.values(), key=lambda e: e.score or 0, reverse=True)[:_MAX_EVIDENCE]
        # Present sources in document order so the answer reads naturally.
        ranked.sort(key=lambda e: (e.document_name, e.page or 0))
        items = [Evidence(i, e.text, e.document_id, e.document_name, e.page, e.chunk_id, e.score)
                 for i, e in enumerate(ranked, start=1)]
        if self.library is not None:
            listing = "\n".join(
                f"- {d.name} ({d.file_type.lstrip('.')}, status: {d.status}"
                + (f", {d.pages} pages" if d.pages else "") + ")"
                for d in self.library.documents
            ) or "(no documents)"
            items.append(Evidence(len(items) + 1, listing, None, "Your document library", None, None, None))
        return items

    def strong_hits(self, threshold: float) -> int:
        return sum(1 for e in self.evidence.values() if (e.score or 0) >= threshold)


def _normalise_query(q: str) -> str:
    return re.sub(r"\W+", " ", q.lower()).strip()


class AgentRunner:
    def __init__(self, llm: LLMClient, tools: ToolRegistry, limits: AgentLimits):
        self.llm = llm
        self.tools = tools
        self.limits = limits

    # -- helpers ------------------------------------------------------------------------------

    def _remaining_s(self, st: _State) -> float:
        return st.deadline - time.monotonic()

    def _generation_reserve(self, st: _State, question: str, history: list[tuple[str, str]]) -> int:
        msgs = prompts.build_answer_messages(question, history, st.numbered_evidence())
        return estimate_messages_tokens(msgs) + self.limits.answer_max_tokens

    def _call_tool(self, st: _State, ctx: ToolContext, name: str, args: dict[str, Any]) -> tuple[ToolResult, AgentStep]:
        st.tool_calls += 1
        result = self.tools.execute(name, args, ctx)
        detail: dict[str, Any] = {"args": result.args}
        if not result.ok:
            detail["error"] = result.error
            st.observations.append(f"Tool {name} failed: {result.error}")
        elif isinstance(result.output, SearchDocumentsOutput):
            out = result.output
            st.tried_queries.add(_normalise_query(out.query))
            new = st.add_hits(out)
            detail.update(hits=len(out.hits), new_hits=new, best_score=out.best_score, **out.timings_ms)
            lines = [f'search_documents("{out.query}") -> {len(out.hits)} relevant passages']
            lines += [f"  [{h.score:.2f}] {h.document_name}"
                      + (f" p.{h.page}" if h.page else "") + f": {prompts.preview(h.text)}"
                      for h in out.hits[:5]]
            st.observations.append("\n".join(lines))
        elif isinstance(result.output, ListDocumentsOutput):
            st.library = result.output
            detail["documents"] = len(result.output.documents)
            st.observations.append(
                "list_documents -> " + (", ".join(f"{d.name} ({d.status})" for d in result.output.documents)
                                        or "no documents")
            )
        step = AgentStep("tool", name, detail, result.duration_ms)
        st.trace.append(step)
        return result, step

    def _llm(self, st: _State, messages: list[dict[str, str]], max_tokens: int) -> LLMResult:
        result = self.llm.chat(messages, max_tokens=max_tokens, temperature=0.0, deadline=st.deadline)
        st.llm_calls += 1
        st.usage.add(result.usage)
        return result

    def _decide(self, st: _State, question: str, history: list[tuple[str, str]]) -> tuple[_Decision, AgentStep]:
        remaining_calls = self.limits.max_tool_calls - st.tool_calls
        messages = prompts.build_controller_messages(question, history, st.observations, remaining_calls)
        start = time.perf_counter()
        result = self._llm(st, messages, _CONTROLLER_MAX_TOKENS)
        decision = parse_decision(result.text)
        fallback = decision is None
        if decision is None:
            decision = _Decision(action="answer" if st.evidence or st.library else "cannot_answer")
        if decision.action == "search_documents" and _normalise_query(decision.query or "") in st.tried_queries:
            decision = _Decision(action="answer" if st.evidence else "cannot_answer")
            fallback = True
        detail = {"action": decision.action, "fallback": fallback, "tokens": result.usage.total_tokens}
        if decision.query:
            detail["query"] = decision.query
        step = AgentStep("decision", "controller", detail, round((time.perf_counter() - start) * 1000, 2))
        st.trace.append(step)
        return decision, step

    def _stop(self, st: _State, reason: str) -> AgentStep:
        step = AgentStep("stop", reason, {"tool_calls": st.tool_calls, "llm_calls": st.llm_calls,
                                          "tokens": st.usage.total_tokens})
        st.trace.append(step)
        return step

    # -- main loop ----------------------------------------------------------------------------

    def run(self, ctx: ToolContext, question: str, history: list[tuple[str, str]]) -> AgentOutcome:
        outcome = None
        for event in self.run_events(ctx, question, history, stream=False):
            if event.type == "final":
                outcome = event.outcome
        assert outcome is not None
        return outcome

    def run_events(self, ctx: ToolContext, question: str, history: list[tuple[str, str]],
                   stream: bool) -> Iterator[AgentEvent]:
        now = time.monotonic()
        st = _State(started=now, deadline=now + self.limits.timeout_seconds)

        def final(status: AgentStatus, answer: str, cited: list[Evidence], grounded: bool) -> AgentEvent:
            outcome = AgentOutcome(status, answer, cited, grounded, st.trace, st.usage, st.llm_calls,
                                   round((time.monotonic() - st.started) * 1000, 2))
            log_event(logger, "agent run", status=status, tool_calls=st.tool_calls, llm_calls=st.llm_calls,
                      tokens=st.usage.total_tokens, elapsed_ms=outcome.elapsed_ms, citations=len(cited))
            return AgentEvent("final", outcome=outcome)

        ready, total = self._document_counts(ctx)
        if ready == 0:
            yield AgentEvent("step", step=self._stop(st, "no_ready_documents"))
            message = prompts.NO_DOCUMENTS_MESSAGE if total == 0 else (
                "Your documents are still being processed (or failed to process). "
                "Please wait until at least one shows as ready.")
            yield final("no_documents", message, [], False)
            return

        # Iteration 1: search with the user's own words.
        _, step = self._call_tool(st, ctx, "search_documents", {"query": question[:500]})
        yield AgentEvent("step", step=step)

        action = "answer"
        for _ in range(1, self.limits.max_iterations):
            if st.strong_hits(self.limits.strong_score) >= 2 or (
                st.strong_hits(self.limits.strong_score + 0.1) >= 1
            ):
                yield AgentEvent("step", step=self._stop(st, "sufficient_evidence"))
                break
            if st.tool_calls >= self.limits.max_tool_calls:
                yield AgentEvent("step", step=self._stop(st, "max_tool_calls"))
                break
            if self._remaining_s(st) < max(10.0, self.limits.timeout_seconds * 0.4):
                yield AgentEvent("step", step=self._stop(st, "time_budget"))
                break
            reserve = self._generation_reserve(st, question, history)
            controller_cost = estimate_messages_tokens(
                prompts.build_controller_messages(question, history, st.observations, 0)
            ) + _CONTROLLER_MAX_TOKENS
            if st.usage.total_tokens + controller_cost + reserve > self.limits.token_budget:
                yield AgentEvent("step", step=self._stop(st, "token_budget"))
                break

            decision, step = self._decide(st, question, history)
            yield AgentEvent("step", step=step)
            action = decision.action
            if action == "search_documents":
                args: dict[str, Any] = {"query": decision.query}
                if decision.top_k:
                    args["top_k"] = decision.top_k
                _, step = self._call_tool(st, ctx, "search_documents", args)
                yield AgentEvent("step", step=step)
                action = "answer"
            elif action == "list_documents":
                _, step = self._call_tool(st, ctx, "list_documents", {})
                yield AgentEvent("step", step=step)
                action = "answer"
            else:
                break
        else:
            yield AgentEvent("step", step=self._stop(st, "max_iterations"))

        if action == "direct_reply":
            yield from self._generate(st, prompts.build_direct_reply_messages(question, history), [], stream,
                                      direct=True, final=final)
            return
        evidence = st.numbered_evidence()
        if action == "cannot_answer" or not evidence:
            yield AgentEvent("step", step=self._stop(st, "insufficient_evidence"))
            yield final("no_answer", prompts.NO_ANSWER_MESSAGE, [], False)
            return
        yield from self._generate(st, prompts.build_answer_messages(question, history, evidence), evidence,
                                  stream, direct=False, final=final)

    def _generate(self, st: _State, messages: list[dict[str, str]], evidence: list[Evidence], stream: bool,
                  *, direct: bool, final: Any) -> Iterator[AgentEvent]:
        start = time.perf_counter()
        if stream:
            text, result = yield from self._stream_text(st, messages)
        else:
            result = self._llm(st, messages, self.limits.answer_max_tokens)
            text = result.text
        step = AgentStep("generation", "direct_reply" if direct else "answer",
                         {"sources": len(evidence), "tokens": result.usage.total_tokens,
                          "finish_reason": result.finish_reason},
                         round((time.perf_counter() - start) * 1000, 2))
        st.trace.append(step)
        yield AgentEvent("step", step=step)
        if direct:
            yield final("direct_reply", text, [], False)
            return
        checked = check_answer(text, evidence)
        step.detail.update(grounded=checked.grounded, cited=len(checked.cited),
                           removed_citations=checked.removed_citations)
        if checked.insufficient or not checked.text:
            yield final("no_answer", prompts.NO_ANSWER_MESSAGE, [], False)
            return
        yield final("answered", checked.text, checked.cited, checked.grounded)

    def _stream_text(self, st: _State, messages: list[dict[str, str]]):  # type: ignore[no-untyped-def]
        """Stream tokens, holding back output while it could still be the INSUFFICIENT_EVIDENCE
        sentinel so the client never sees it."""
        sentinel = prompts.INSUFFICIENT_EVIDENCE
        buffer, released = "", False
        result: LLMResult | None = None
        for ev in self.llm.stream_chat(messages, max_tokens=self.limits.answer_max_tokens,
                                       temperature=None, deadline=st.deadline):
            if ev.done:
                result = ev.result
                break
            if released:
                yield AgentEvent("token", delta=ev.delta)
                continue
            buffer += ev.delta
            probe = buffer.lstrip().upper()
            could_be_sentinel = sentinel.startswith(probe[: len(sentinel)]) and len(probe) <= len(sentinel) + 2
            if not could_be_sentinel and not is_insufficient(buffer):
                released = True
                yield AgentEvent("token", delta=buffer)
        assert result is not None
        st.llm_calls += 1
        st.usage.add(result.usage)
        if not released and buffer and not is_insufficient(buffer):
            yield AgentEvent("token", delta=buffer)
        return result.text, result

    @staticmethod
    def _document_counts(ctx: ToolContext) -> tuple[int, int]:
        rows = ctx.session.execute(
            select(Document.status, func.count()).where(Document.owner_id == ctx.owner_id).group_by(Document.status)
        ).all()
        counts = {status: n for status, n in rows}
        return counts.get(DocumentStatus.READY, 0), sum(counts.values())
