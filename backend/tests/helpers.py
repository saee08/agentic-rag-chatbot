"""Test doubles and file builders.

* ``FakeEmbedder``: deterministic hashed bag-of-words vectors. Texts sharing content words have
  high cosine similarity, unrelated texts ~0, so retrieval relevance can be asserted without
  downloading a model.
* ``FakeOpenRouter``: an httpx transport that emulates the OpenRouter chat-completions API
  (JSON and SSE streaming), so the real ``LLMClient`` code path is exercised.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

from app.rag import prompts

STOPWORDS = frozenset(
    (
        "a an the of to in on for and or is are was were be been what which who whom how why when where "
        "does do did this that these those it its with by from as at about into than then there their our "
        "your my me i we you he she they them his her can could would should will shall may might please "
        "tell give list describe explain summarize summarise mentioned document documents file"
    ).split()
)


def content_words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in STOPWORDS]


class FakeEmbedder:
    def __init__(self, dim: int = 384, model_name: str = "fake-hash-embedder"):
        self.dim = dim
        self.model_name = model_name
        self.calls = 0
        self.texts_embedded = 0

    @property
    def loaded(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.texts_embedded += len(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in content_words(text):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim  # noqa: S324
                out[row, bucket] += 1.0
            if not out[row].any():
                out[row, 0] = 1e-3  # avoid zero vectors for stopword-only text
            out[row] /= np.linalg.norm(out[row])
        return out


# --- fake LLM ----------------------------------------------------------------------------------

_SOURCE_RE = re.compile(r'<source id="(\d+)"[^>]*>\n(.*?)\n</source>', re.S)
_QUESTION_RE = re.compile(r"<question>\n(.*?)\n</question>", re.S)


@dataclass
class FakeOpenRouter:
    """Scripted OpenRouter. Override ``controller`` / ``answer`` callables to change behaviour,
    or set ``fail_with`` to a list of HTTP status codes returned before succeeding."""

    controller: Callable[[str], str] | None = None
    answer: Callable[[str, list[tuple[int, str]]], str] | None = None
    fail_with: list[int] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    include_usage: bool = True

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    @property
    def calls(self) -> int:
        return len(self.requests)

    def kinds(self) -> list[str]:
        return [self._kind(r["messages"]) for r in self.requests]

    @staticmethod
    def _kind(messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        if system == prompts.CONTROLLER_SYSTEM:
            return "controller"
        if system == prompts.ANSWER_SYSTEM:
            return "answer"
        if system == prompts.DIRECT_REPLY_SYSTEM:
            return "direct"
        return "other"

    def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"].startswith("Bearer ")
        payload = json.loads(request.content)
        self.requests.append(payload)
        if self.fail_with:
            return httpx.Response(self.fail_with.pop(0), json={"error": {"message": "scripted failure"}},
                                  headers={"Retry-After": "0"})
        messages = payload["messages"]
        user = messages[-1]["content"]
        kind = self._kind(messages)
        if kind == "controller":
            text = (self.controller or self.default_controller)(user)
        elif kind == "answer":
            question = _QUESTION_RE.search(user).group(1)  # type: ignore[union-attr]
            sources = [(int(n), t) for n, t in _SOURCE_RE.findall(user)]
            text = (self.answer or self.default_answer)(question, sources)
        elif kind == "direct":
            text = "Hello! Ask me anything about your uploaded documents."
        else:
            text = "ok"
        usage = {"prompt_tokens": sum(len(m["content"]) // 4 for m in messages),
                 "completion_tokens": max(1, len(text) // 4)}
        if payload.get("stream"):
            return self._stream(text, usage)
        body: dict[str, Any] = {"model": payload["model"],
                                "choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
        if self.include_usage:
            body["usage"] = usage
        return httpx.Response(200, json=body)

    def _stream(self, text: str, usage: dict[str, int]) -> httpx.Response:
        lines = [": OPENROUTER PROCESSING", ""]
        for piece in re.findall(r"\S+\s*", text):
            chunk = {"model": "fake", "choices": [{"delta": {"content": piece}, "finish_reason": None}]}
            lines += [f"data: {json.dumps(chunk)}", ""]
        final = {"model": "fake", "choices": [{"delta": {}, "finish_reason": "stop"}]}
        if self.include_usage:
            final["usage"] = usage
        lines += [f"data: {json.dumps(final)}", "", "data: [DONE]", ""]
        return httpx.Response(200, content="\n".join(lines).encode(),
                              headers={"content-type": "text/event-stream"})

    @staticmethod
    def default_controller(user: str) -> str:
        if "relevant passages" in user and "-> 0 relevant" not in user:
            return '{"action": "answer"}'
        return '{"action": "cannot_answer"}'

    @staticmethod
    def default_answer(question: str, sources: list[tuple[int, str]]) -> str:
        """Grounded extractive answer: quote the best-overlapping sentence and cite it."""
        q = set(content_words(question))
        best: tuple[int, int, str] | None = None
        for number, text in sources:
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                overlap = len(q & set(content_words(sentence)))
                if overlap and (best is None or overlap > best[0]):
                    best = (overlap, number, sentence.strip())
        if best is None:
            return prompts.INSUFFICIENT_EVIDENCE
        return f"{best[2]} [{best[1]}]"


# --- file builders -----------------------------------------------------------------------------


def make_pdf(pages: list[str]) -> bytes:
    """Minimal valid PDF with one text line per sentence on each page (Helvetica)."""

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects: list[bytes] = []
    n_pages = len(pages)
    font_id = 3
    page_ids = [4 + 2 * i for i in range(n_pages)]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        lines = [ln for ln in re.split(r"(?<=[.!?])\s+", text) if ln] or [""]
        ops = "BT /F1 11 Tf 14 TL 50 760 Td " + " ".join(f"({esc(ln)}) Tj T*" for ln in lines) + " ET"
        stream = ops.encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {page_ids[i] + 1} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def make_docx(paragraphs: list[str], table: list[list[str]] | None = None) -> bytes:
    import docx

    document = docx.Document()
    for p in paragraphs:
        document.add_paragraph(p)
    if table:
        t = document.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                t.cell(r, c).text = value
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def make_pptx(slides: list[tuple[str, str]], notes: dict[int, str] | None = None) -> bytes:
    from pptx import Presentation

    presentation = Presentation()
    for index, (title, body) in enumerate(slides):
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
        if notes and index in notes:
            slide.notes_slide.notes_text_frame.text = notes[index]
    buf = io.BytesIO()
    presentation.save(buf)
    return buf.getvalue()


ACME_REPORT = (
    "Acme Corporation Annual Report 2024. "
    "Total revenue grew to 4.2 billion dollars in fiscal 2024, up twelve percent. "
    "The key performance indicators were customer retention of 94 percent and net promoter score of 61. "
    "The company opened three new factories in Ohio, Texas and Bavaria. "
    "Chief executive Dana Whitfield announced a carbon neutrality target for 2030."
)
