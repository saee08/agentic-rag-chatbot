from __future__ import annotations

from fastapi import APIRouter, Response, status
from fastapi.responses import StreamingResponse

from app.api.deps import ContainerDep, CurrentUser, SessionDep, enforce
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationCreate,
    ConversationDetail,
    ConversationOut,
    ListResponse,
    MessageOut,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
def create_conversation(body: ConversationCreate, user: CurrentUser, container: ContainerDep,
                        session: SessionDep) -> ConversationOut:
    return ConversationOut.model_validate(container.chat.create(session, user.id, body.title))


@router.get("", response_model=ListResponse[ConversationOut])
def list_conversations(user: CurrentUser, container: ContainerDep,
                       session: SessionDep) -> ListResponse[ConversationOut]:
    return ListResponse(items=[ConversationOut.model_validate(c) for c in container.chat.list(session, user.id)])


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str, user: CurrentUser, container: ContainerDep,
                     session: SessionDep) -> ConversationDetail:
    conv = container.chat.get(session, user.id, conversation_id)
    return ConversationDetail(
        id=conv.id, title=conv.title, created_at=conv.created_at, updated_at=conv.updated_at,
        messages=[MessageOut.model_validate(m) for m in conv.messages],
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, user: CurrentUser, container: ContainerDep,
                        session: SessionDep) -> Response:
    container.chat.delete(session, user.id, conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{conversation_id}/messages", response_model=ChatResponse)
def send_message(conversation_id: str, body: ChatRequest, user: CurrentUser, container: ContainerDep,
                 session: SessionDep) -> ChatResponse:
    enforce(container.rate_limits.chat, f"user:{user.id}")
    user_msg, reply = container.chat.send(session, user.id, conversation_id, body.content)
    return ChatResponse(user_message=MessageOut.model_validate(user_msg),
                        assistant_message=MessageOut.model_validate(reply))


@router.post("/{conversation_id}/messages/stream")
def stream_message(conversation_id: str, body: ChatRequest, user: CurrentUser, container: ContainerDep,
                   session: SessionDep) -> StreamingResponse:
    """Server-Sent Events: ``user_message``, ``step`` (agent trace), ``token`` (answer deltas),
    then ``final`` (the stored assistant message) or ``error``."""
    enforce(container.rate_limits.chat, f"user:{user.id}")
    container.chat.get(session, user.id, conversation_id)  # 404 before streaming starts
    return StreamingResponse(
        container.chat.stream(user.id, conversation_id, body.content),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
