"""Conversations, messages and agent invocation (owner-scoped)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.agents.orchestrator import AgentOutcome, AgentRunner
from app.agents.tools import ToolContext
from app.core.exceptions import NotFoundError, ServiceUnavailableError
from app.core.logging import log_event
from app.db.session import Database
from app.models import Conversation, Message, MessageRole, utcnow
from app.rag.llm import LLMError
from app.rag.prompts import Evidence
from app.schemas import MessageOut

logger = logging.getLogger(__name__)

_SNIPPET_CHARS = 1200
_TITLE_CHARS = 60


def citation_dict(ev: Evidence) -> dict[str, Any]:
    snippet = ev.text if len(ev.text) <= _SNIPPET_CHARS else ev.text[:_SNIPPET_CHARS] + "…"
    return {"number": ev.number, "document_id": ev.document_id, "document_name": ev.document_name,
            "page": ev.page, "chunk_id": ev.chunk_id, "score": ev.score, "snippet": snippet}


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

    # -- messaging -------------------------------------------------------------------------------

    def _history(self, session: Session, conversation_id: str) -> list[tuple[str, str]]:
        if self.history_messages == 0:
            return []
        rows = session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id,
                   or_(Message.status.is_(None), Message.status != "error"))
            .order_by(Message.created_at.desc()).limit(self.history_messages)
        ).all()
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
    def _save_outcome(session: Session, conv: Conversation, outcome: AgentOutcome) -> Message:
        msg = Message(
            conversation_id=conv.id, role=MessageRole.ASSISTANT, content=outcome.answer, status=outcome.status,
            citations=[citation_dict(c) for c in outcome.citations],
            agent_trace=[s.as_dict() for s in outcome.trace],
            usage={**outcome.usage.as_dict(), "llm_calls": outcome.llm_calls, "elapsed_ms": outcome.elapsed_ms,
                   "grounded": outcome.grounded},
        )
        session.add(msg)
        conv.updated_at = utcnow()
        session.commit()
        return msg

    @staticmethod
    def _save_error(session: Session, conv: Conversation, message: str) -> Message:
        msg = Message(conversation_id=conv.id, role=MessageRole.ASSISTANT, content=message, status="error")
        session.add(msg)
        session.commit()
        return msg

    def send(self, session: Session, owner_id: str, conversation_id: str, content: str) -> tuple[Message, Message]:
        conv, user_msg, history = self._start(session, owner_id, conversation_id, content)
        runner = self.runner_factory()
        try:
            outcome = runner.run(ToolContext(session, owner_id), content, history)
        except LLMError as exc:
            log_event(logger, "chat failed", logging.WARNING, error_type=type(exc).__name__)
            self._save_error(session, conv, str(exc))
            raise ServiceUnavailableError(str(exc)) from exc
        return user_msg, self._save_outcome(session, conv, outcome)

    def stream(self, owner_id: str, conversation_id: str, content: str) -> Iterator[str]:
        """SSE event stream. The route checks conversation ownership before streaming starts;
        this generator owns its own DB session because it outlives the request scope."""
        session = self.db.session()
        try:
            conv, user_msg, history = self._start(session, owner_id, conversation_id, content)
            yield sse("user_message", MessageOut.model_validate(user_msg).model_dump(mode="json"))
            runner = self.runner_factory()
            try:
                for event in runner.run_events(ToolContext(session, owner_id), content, history, stream=True):
                    if event.type == "step" and event.step is not None:
                        yield sse("step", event.step.as_dict())
                    elif event.type == "token":
                        yield sse("token", {"delta": event.delta})
                    elif event.type == "final" and event.outcome is not None:
                        msg = self._save_outcome(session, conv, event.outcome)
                        yield sse("final", MessageOut.model_validate(msg).model_dump(mode="json"))
            except LLMError as exc:
                log_event(logger, "chat stream failed", logging.WARNING, error_type=type(exc).__name__)
                msg = self._save_error(session, conv, str(exc))
                yield sse("error", {"code": "service_unavailable", "message": str(exc),
                                    "message_id": msg.id})
            except Exception:
                logger.exception("chat stream crashed")
                session.rollback()
                yield sse("error", {"code": "internal_error", "message": "An internal error occurred."})
        finally:
            session.close()
