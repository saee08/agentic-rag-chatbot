"""Agent orchestration: sufficiency gating, refinement, bounds, fallbacks, permissions."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from app.agents.orchestrator import AgentRunner, parse_decision
from app.agents.tools import READ_DOCUMENTS, ListDocumentsTool, SearchDocumentsTool, ToolContext, ToolRegistry
from app.container import Container
from app.db.migrate import upgrade
from app.models import Document, DocumentStatus, IngestionJob, User
from app.rag import prompts
from tests.conftest import make_settings
from tests.helpers import ACME_REPORT, FakeEmbedder, FakeOpenRouter


@pytest.fixture
def env(tmp_path: Path):
    """A migrated DB with one user and one READY document, plus a runner factory."""
    settings = make_settings(tmp_path)
    upgrade(settings.database_url)
    llm = FakeOpenRouter()
    container = Container(settings, embedder=FakeEmbedder(), llm_transport=llm.transport())
    upload_dir = Path(settings.upload_dir)
    with container.db.transaction() as s:
        user = User(email="a@example.com", password_hash="x")
        s.add(user)
        s.flush()
        doc = Document(owner_id=user.id, filename="acme.txt", file_type=".txt", size_bytes=1, content_sha256="h",
                       storage_key=f"{user.id}/d.txt")
        s.add(doc)
        s.flush()
        (upload_dir / user.id).mkdir(parents=True)
        (upload_dir / user.id / "d.txt").write_text(ACME_REPORT)
        job = IngestionJob(document_id=doc.id)
        s.add(job)
        s.flush()
        user_id, job_id = user.id, job.id
    container.ingestion.process(job_id)
    session = container.db.session()

    def runner(**limit_overrides) -> AgentRunner:
        return AgentRunner(container.llm, container.tools, dataclasses.replace(container.limits, **limit_overrides))

    yield type("Env", (), {"llm": llm, "ctx": ToolContext(session, user_id), "runner": staticmethod(runner),
                           "container": container, "session": session})
    session.close()
    container.close()


def statuses(outcome):
    return [(s.kind, s.name) for s in outcome.trace]


def test_strong_evidence_skips_the_planner(env):
    outcome = env.runner(strong_score=0.2).run(env.ctx, "key performance indicators customer retention", [])
    assert outcome.status == "answered" and outcome.grounded
    assert env.llm.kinds() == ["answer"]  # no controller call needed
    assert ("stop", "sufficient_evidence") in statuses(outcome)


def test_weak_evidence_triggers_refined_search(env):
    queries = []

    def controller(user: str) -> str:
        if "search_documents(\"customer retention" in user:
            return '{"action": "answer"}'
        queries.append(1)
        return json.dumps({"action": "search_documents", "query": "customer retention net promoter score"})

    env.llm.controller = controller
    env.llm.answer = lambda q, sources: f"Customer retention was 94 percent [{sources[0][0]}]."
    outcome = env.runner().run(env.ctx, "How loyal are the clients?", [])
    searches = [s for s in outcome.trace if s.name == "search_documents"]
    assert len(searches) == 2
    assert searches[1].detail["args"]["query"] == "customer retention net promoter score"
    assert outcome.status == "answered"
    assert "94 percent" in outcome.answer


def test_loop_is_bounded_by_tool_calls(env):
    counter = iter(range(100))
    env.llm.controller = lambda user: json.dumps({"action": "search_documents", "query": f"unrelated {next(counter)}"})
    outcome = env.runner(max_iterations=10, max_tool_calls=3).run(env.ctx, "zebra migration patterns", [])
    assert sum(1 for s in outcome.trace if s.kind == "tool") == 3
    assert ("stop", "max_tool_calls") in statuses(outcome)
    assert outcome.status == "no_answer"


def test_loop_is_bounded_by_iterations(env):
    counter = iter(range(100))
    env.llm.controller = lambda user: json.dumps({"action": "search_documents", "query": f"unrelated {next(counter)}"})
    outcome = env.runner(max_iterations=2, max_tool_calls=10).run(env.ctx, "zebra migration patterns", [])
    assert ("stop", "max_iterations") in statuses(outcome)
    assert sum(1 for s in outcome.trace if s.kind == "decision") == 1


def test_invalid_controller_output_falls_back_safely(env):
    env.llm.controller = lambda user: "I think we should search more!!!"
    outcome = env.runner().run(env.ctx, "zebra migration patterns", [])
    decision = next(s for s in outcome.trace if s.kind == "decision")
    assert decision.detail["fallback"] is True and decision.detail["action"] == "cannot_answer"
    assert outcome.status == "no_answer"


def test_repeated_query_is_not_executed_again(env):
    env.llm.controller = lambda user: '{"action": "search_documents", "query": "Zebra migration patterns?"}'
    outcome = env.runner().run(env.ctx, "zebra migration patterns", [])
    assert sum(1 for s in outcome.trace if s.kind == "tool") == 1
    assert next(s for s in outcome.trace if s.kind == "decision").detail["fallback"] is True


def test_direct_reply_for_conversational_messages(env):
    env.llm.controller = lambda user: '{"action": "direct_reply"}'
    outcome = env.runner().run(env.ctx, "hello there!", [])
    assert outcome.status == "direct_reply" and outcome.citations == []
    assert env.llm.kinds()[-1] == "direct"


def test_list_documents_tool_answers_library_questions(env):
    env.llm.controller = lambda user: (
        '{"action": "answer"}' if "list_documents ->" in user else '{"action": "list_documents"}'
    )
    env.llm.answer = lambda q, sources: f"You have acme.txt [{sources[-1][0]}]."
    outcome = env.runner().run(env.ctx, "which files have I uploaded?", [])
    assert outcome.status == "answered"
    assert outcome.citations[-1].document_name == "Your document library"


def test_model_insufficient_evidence_becomes_honest_no_answer(env):
    env.llm.answer = lambda q, s: prompts.INSUFFICIENT_EVIDENCE
    outcome = env.runner(strong_score=0.2).run(env.ctx, "key performance indicators customer retention", [])
    assert outcome.status == "no_answer" and outcome.answer == prompts.NO_ANSWER_MESSAGE


def test_ungrounded_answer_is_flagged(env):
    env.llm.answer = lambda q, s: "Retention was great."
    outcome = env.runner(strong_score=0.2).run(env.ctx, "key performance indicators customer retention", [])
    assert outcome.status == "answered" and outcome.grounded is False


def test_token_budget_prevents_planner_calls(env):
    outcome = env.runner(token_budget=1000).run(env.ctx, "zebra migration patterns", [])
    assert ("stop", "token_budget") in statuses(outcome)
    assert "controller" not in env.llm.kinds()


def test_time_budget_stops_planning(env):
    outcome = env.runner(timeout_seconds=5).run(env.ctx, "zebra migration patterns", [])
    assert ("stop", "time_budget") in statuses(outcome)


def test_no_documents_means_no_llm_calls(env):
    with env.container.db.transaction() as s:
        for doc in s.query(Document).all():
            doc.status = DocumentStatus.PROCESSING
    outcome = env.runner().run(env.ctx, "anything", [])
    assert outcome.status == "no_documents" and env.llm.calls == 0


def test_streaming_hides_the_insufficient_sentinel(env):
    env.llm.answer = lambda q, s: prompts.INSUFFICIENT_EVIDENCE
    events = list(env.runner(strong_score=0.2).run_events(
        env.ctx, "key performance indicators customer retention", [], stream=True))
    assert not [e for e in events if e.type == "token"]
    assert events[-1].outcome.status == "no_answer"


def test_streaming_emits_tokens_matching_final_answer(env):
    events = list(env.runner(strong_score=0.2).run_events(
        env.ctx, "key performance indicators customer retention", [], stream=True))
    streamed = "".join(e.delta for e in events if e.type == "token")
    assert events[-1].outcome.status == "answered"
    assert re.sub(r"\s*\[\d+\]", "", streamed).strip() == re.sub(r"\s*\[\d+\]", "", events[-1].outcome.answer)


def test_registry_enforces_permissions_and_validation(env):
    ctx = env.ctx
    locked = ToolRegistry([SearchDocumentsTool(env.container.retriever), ListDocumentsTool()], set())
    assert locked.names == []
    assert "not granted" in locked.execute("search_documents", {"query": "x"}, ctx).error
    open_ = ToolRegistry([SearchDocumentsTool(env.container.retriever)], {READ_DOCUMENTS})
    assert "unknown tool" in open_.execute("delete_everything", {}, ctx).error
    assert "invalid arguments" in open_.execute("search_documents", {"query": "x", "top_k": 500}, ctx).error
    # The model cannot smuggle an owner id: extra fields are forbidden.
    assert "invalid arguments" in open_.execute("search_documents", {"query": "x", "owner_id": "other"}, ctx).error


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"action": "answer"}', "answer"),
        ('Sure! {"action": "search_documents", "query": "q2 revenue"} done', "search_documents"),
        ('```json\n{"action": "cannot_answer"}\n```', "cannot_answer"),
        ('{"action": "search_documents"}', None),  # missing query
        ('{"action": "rm -rf"}', None),
        ("no json here", None),
        ('{"action": "answer", "query": {"nested": 1}}', None),
    ],
)
def test_parse_decision(text, expected):
    decision = parse_decision(text)
    assert (decision.action if decision else None) == expected


def test_planner_cannot_skip_generation_when_evidence_exists(env):
    """The planner sees only previews; with relevant passages the answer step must still read them."""
    env.llm.controller = lambda user: '{"action": "cannot_answer"}'
    env.llm.answer = lambda q, sources: f"The documents do not specify this value [{sources[0][0]}]."
    outcome = env.runner(strong_score=0.99).run(env.ctx, "customer retention target", [])
    assert "answer" in env.llm.kinds()
    assert outcome.status == "answered" and "do not specify" in outcome.answer
