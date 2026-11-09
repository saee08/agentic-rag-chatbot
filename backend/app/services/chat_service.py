"""Conversations, messages and agent invocation (owner-scoped)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.agents.orchestrator import AgentOutcome, AgentRunner
from app.agents.tools import ToolContext
from app.core.exceptions import ConflictError, NotFoundError, ServiceUnavailableError
from app.core.logging import log_event
from app.db.session import Database
from app.models import Conversation, Message, MessageRole, utcnow
from app.rag.excerpts import relevant_excerpt
from app.rag.generator import citing_text, merge_duplicate_sources
from app.rag.llm import (
    LLMBadResponseError,
    LLMError,
    LLMNotConfiguredError,
    LLMRateLimitedError,
    LLMUnavailableError,
)
from app.rag.prompts import Evidence
from app.services.message_view import build_steps, to_message_out

logger = logging.getLogger(__name__)

_TITLE_CHARS = 60
_ERROR_CODES: list[tuple[type[LLMError], str]] = [
    (LLMNotConfiguredError, "llm_not_configured"),
    (LLMRateLimitedError, "llm_rate_limited"),
    (LLMUnavailableError, "llm_unavailable"),
    (LLMBadResponseError, "llm_bad_response"),
]


def error_code_for(exc: LLMError) -> str:
    return next((code for cls, code in _ERROR_CODES if isinstance(exc, cls)), "llm_unavailable")


def source_dict(ev: Evidence, question: str, answer: str) -> dict[str, Any]:
    """Stored source record. The excerpt is chosen using the answer sentences that cite it."""
    return {
        "number": ev.number, "document_id": ev.document_id, "document_name": ev.document_name,
        "page_number": ev.page, "chunk_id": ev.chunk_id, "score": ev.score, "match": ev.match,
        "excerpt": relevant_excerpt(ev.text, question, citing_text(answer, ev.number)), "content": ev.text,
    }


def _title_from(question: str) -> str:
    text = " ".join(question.split())
    return text if len(text) <= _TITLE_CHARS else text[: _TITLE_CHARS - 1].rstrip() + "…"


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


class ChatService:
    def __init__(self, db: Database, runner_factory: Callable[[], AgentRunner], history_messages: int):
        self.db = db
        self.runner_factory = runner_factory
        self.history_messages = history_messages

    # -- conversations ---------------------------------------------------------------------------

    @staticmethod
    def create(session: Session, owner_id: str, title: str | None) -> Conversation:
        conv = Conversation(owner_id=owner_id, title=(title or "").strip()[:200] or "New conversation")
        session.add(conv)
        session.commit()
        return conv

    @staticmethod
    def list(session: Session, owner_id: str) -> list[Conversation]:
        return list(session.scalars(
            select(Conversation).where(Conversation.owner_id == owner_id).order_by(Conversation.updated_at.desc())
        ))

    @staticmethod
    def get(session: Session, owner_id: str, conversation_id: str) -> Conversation:
        conv = session.scalar(
            select(Conversation).where(Conversation.id == conversation_id, Conversation.owner_id == owner_id)
        )
        if conv is None:
            raise NotFoundError("Conversation not found.")
        return conv

    def delete(self, session: Session, owner_id: str, conversation_id: str) -> None:
        session.delete(self.get(session, owner_id, conversation_id))
        session.commit()

    # -- helpers ---------------------------------------------------------------------------------

    def _history(self, session: Session, conversation_id: str, before: datetime | None = None
                 ) -> list[tuple[str, str]]:
        if self.history_messages == 0:
            return []
        stmt = select(Message).where(Message.conversation_id == conversation_id,
                                     or_(Message.status.is_(None), Message.status != "error"))
        if before is not None:
            stmt = stmt.where(Message.created_at < before)
        rows = session.scalars(stmt.order_by(Message.created_at.desc()).limit(self.history_messages)).all()
        return [(m.role.value, m.content) for m in reversed(rows)]

    def _start(self, session: Session, owner_id: str, conversation_id: str, content: str
               ) -> tuple[Conversation, Message, list[tuple[str, str]]]:
        conv = self.get(session, owner_id, conversation_id)
        history = self._history(session, conv.id)
        if not history and conv.title == "New conversation":
            conv.title = _title_from(content)
        user_msg = Message(conversation_id=conv.id, role=MessageRole.USER, content=content)
        session.add(user_msg)
        conv.updated_at = utcnow()
        session.commit()
        return conv, user_msg, history

    @staticmethod
    def _save_outcome(session: Session, conv: Conversation, question: str, outcome: AgentOutcome) -> Message:
        sources = [source_dict(c, question, outcome.answer) for c in outcome.citations]
        answer, sources = merge_duplicate_sources(outcome.answer, sources)
        msg = Message(
            conversation_id=conv.id, role=MessageRole.ASSISTANT, content=answer, status=outcome.status,
            citations=sources,
            agent_trace=[s.as_dict() for s in outcome.trace],
            usage={**outcome.usage.as_dict(), "llm_calls": outcome.llm_calls, "elapsed_ms": outcome.elapsed_ms,
                   "grounded": outcome.grounded, "model": outcome.model,
                   "passages_retrieved": outcome.passages_retrieved},
        )
        session.add(msg)
        conv.updated_at = utcnow()
        session.commit()
        return msg

    @staticmethod
    def _save_error(session: Session, conv: Conversation, code: str, message: str) -> Message:
        msg = Message(conversation_id=conv.id, role=MessageRole.ASSISTANT, content=message, status="error",
                      error_code=code)
        session.add(msg)
        session.commit()
        return msg

    # -- messaging -------------------------------------------------------------------------------

    def send(self, session: Session, owner_id: str, conversation_id: str, content: str) -> tuple[Message, Message]:
        conv, user_msg, history = self._start(session, owner_id, conversation_id, content)
        try:
            outcome = self.runner_factory().run(ToolContext(session, owner_id), content, history)
        except LLMError as exc:
            log_event(logger, "chat failed", logging.WARNING, error_type=type(exc).__name__)
            self._save_error(session, conv, error_code_for(exc), str(exc))
            raise ServiceUnavailableError(str(exc), code=error_code_for(exc)) from exc
        return user_msg, self._save_outcome(session, conv, content, outcome)

    def prepare_regenerate(self, session: Session, owner_id: str, conversation_id: str, user_message_id: str
                           ) -> None:
        """Validate a regenerate request and remove the failed reply it replaces.

        Only the last user message of a conversation can be regenerated, and only when it has no
        successful reply (its replies, if any, are errors)."""
        conv = self.get(session, owner_id, conversation_id)
        messages = conv.messages
        index = next((i for i, m in enumerate(messages) if m.id == user_message_id), None)
        if index is None or messages[index].role != MessageRole.USER:
            raise NotFoundError("Message not found.")
        later = messages[index + 1:]
        if any(m.role == MessageRole.USER for m in later):
            raise ConflictError("Only the most recent question can be retried.")
        if any(m.status != "error" for m in later):
            raise ConflictError("This question already has an answer.")
        for m in later:
            session.delete(m)
        session.commit()

    def stream(self, owner_id: str, conversation_id: str, content: str) -> Iterator[str]:
        """SSE stream for a new question. The route checks ownership before streaming starts;
        this generator owns its own DB session because it outlives the request scope."""
        session = self.db.session()
        try:
            conv, user_msg, history = self._start(session, owner_id, conversation_id, content)
            yield sse("user_message", to_message_out(user_msg).model_dump(mode="json"))
            yield from self._run_stream(session, owner_id, conv, content, history)
        finally:
            session.close()

    def stream_regenerate(self, owner_id: str, conversation_id: str, user_message_id: str) -> Iterator[str]:
        session = self.db.session()
        try:
            conv = self.get(session, owner_id, conversation_id)
            user_msg = session.get(Message, user_message_id)
            if user_msg is None or user_msg.conversation_id != conv.id:
                yield sse("error", {"code": "not_found", "message": "Message not found.", "retryable": False})
                return
            history = self._history(session, conv.id, before=user_msg.created_at)
            yield from self._run_stream(session, owner_id, conv, user_msg.content, history)
        finally:
            session.close()

    def _run_stream(self, session: Session, owner_id: str, conv: Conversation, question: str,
                    history: list[tuple[str, str]]) -> Iterator[str]:
        runner = self.runner_factory()
        try:
            for event in runner.run_events(ToolContext(session, owner_id), question, history, stream=True):
                if event.type == "step" and event.step is not None:
                    for step in build_steps([event.step.as_dict()]):
                        yield sse("step", step.model_dump(mode="json"))
                elif event.type == "token":
                    yield sse("token", {"delta": event.delta})
                elif event.type == "final" and event.outcome is not None:
                    msg = self._save_outcome(session, conv, question, event.outcome)
                    yield sse("final", to_message_out(msg, question).model_dump(mode="json"))
        except LLMError as exc:
            log_event(logger, "chat stream failed", logging.WARNING, error_type=type(exc).__name__)
            msg = self._save_error(session, conv, error_code_for(exc), str(exc))
            yield sse("final", to_message_out(msg, question).model_dump(mode="json"))
        except Exception:
            logger.exception("chat stream crashed")
            session.rollback()
            yield sse("error", {"code": "internal_error", "message": "Something went wrong while answering.",
                                "retryable": True})
