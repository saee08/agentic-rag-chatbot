"""Real embedding model (downloads all-MiniLM-L6-v2 on first run).

    pytest -m slow

Checks that the production retrieval thresholds separate relevant from unrelated passages with
the real model, and that the full upload -> ingest -> answer flow works with real vectors.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.rag.embeddings import SentenceTransformerEmbedder
from tests.conftest import ask, auth_headers, make_settings, new_conversation, upload, wait_for_document
from tests.helpers import ACME_REPORT, FakeOpenRouter

pytestmark = pytest.mark.slow

DEFAULTS = Settings(_env_file=None)


@pytest.fixture(scope="module")
def embedder():
    e = SentenceTransformerEmbedder(DEFAULTS.embedding_model, DEFAULTS.embedding_dim)
    e.load()
    return e


def test_embeddings_are_normalised_and_batched(embedder):
    vectors = embedder.embed(["one", "two", "three"])
    assert vectors.shape == (3, 384) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert embedder.embed([]).shape == (0, 384)


FILLER = (
    "The board met four times during the year to review governance, audit findings and risk management. "
    "Employees completed safety training and the company launched a new mentoring programme across all regions. "
    "Supply chain disruptions eased in the second half, and inventory levels returned to normal by December. "
)


@pytest.mark.parametrize(
    ("query", "relevant"),
    [
        ("What were the key performance indicators?", True),
        ("How much revenue did Acme make?", True),
        ("Where were new factories opened?", True),
        ("Who is the chief executive?", True),
        ("What is the carbon neutrality target?", True),
        ("What was customer retention?", True),
        ("What is the boiling point of tungsten?", False),
        ("Recipe for chocolate chip cookies", False),
        ("Who won the football world cup?", False),
        ("How do I reset my router password?", False),
        ("What is the capital of Australia?", False),
        ("Explain quantum entanglement", False),
    ],
)
def test_default_threshold_separates_relevant_from_unrelated(embedder, query, relevant):
    """Scores are computed against realistic ~180-word multi-topic chunks, as produced by ingestion."""
    from app.rag.chunker import chunk_sections
    from app.rag.parser import Section

    chunks = chunk_sections([Section(ACME_REPORT + " " + FILLER * 3)], DEFAULTS.chunk_size_words,
                            DEFAULTS.chunk_overlap_words)
    vectors = embedder.embed([c.text for c in chunks] + [ACME_REPORT])
    best = float((vectors @ embedder.embed([query])[0]).max())
    if relevant:
        assert best >= DEFAULTS.retrieval_min_score, f"{query!r} scored {best:.3f}"
    else:
        assert best < DEFAULTS.retrieval_min_score, f"{query!r} scored {best:.3f}"


def test_full_flow_with_real_embeddings(tmp_path, embedder):
    settings = make_settings(tmp_path, embedding_model=DEFAULTS.embedding_model,
                             retrieval_min_score=DEFAULTS.retrieval_min_score,
                             retrieval_strong_score=DEFAULTS.retrieval_strong_score)
    app = create_app(settings, embedder=embedder, llm_transport=FakeOpenRouter().transport())
    with TestClient(app) as client:
        h = auth_headers(client)
        doc = upload(client, h, "acme.txt", ACME_REPORT.encode())["items"][0]["document"]
        assert wait_for_document(client, h, doc["id"])["status"] == "ready"
        conv = new_conversation(client, h)
        reply = ask(client, h, conv, "What were the key performance indicators?")["assistant_message"]
        assert reply["status"] == "answered" and reply["citations"]
        unrelated = ask(client, h, conv, "What is the boiling point of tungsten?")["assistant_message"]
        assert unrelated["status"] == "no_answer"
