from __future__ import annotations

import pytest

from app.rag.chunker import chunk_sections
from app.rag.parser import Section


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_chunks_are_strings_regression_for_original_bug():
    chunks = chunk_sections([Section(words(700))], chunk_size=300, overlap=50)
    assert chunks and all(isinstance(c.text, str) for c in chunks)


def test_empty_and_short_input():
    assert chunk_sections([], 100, 10) == []
    assert chunk_sections([Section("")], 100, 10) == []
    [only] = chunk_sections([Section("just a few words")], 100, 10)
    assert only.text == "just a few words" and only.word_count == 4


def test_size_limit_overlap_and_full_coverage():
    text = words(1000)
    chunks = chunk_sections([Section(text)], chunk_size=180, overlap=30)
    assert all(c.word_count <= 180 for c in chunks)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.text.split()[-30:] == b.text.split()[:30]
    covered = {w for c in chunks for w in c.text.split()}
    assert covered == set(text.split())


def test_no_redundant_tail_chunk():
    chunks = chunk_sections([Section(words(180))], chunk_size=180, overlap=30)
    assert len(chunks) == 1


def test_sentences_are_kept_whole_when_they_fit():
    s1 = "Alpha beta gamma delta."
    s2 = "Epsilon zeta eta theta."
    chunks = chunk_sections([Section(f"{s1} {s2}")], chunk_size=6, overlap=0)
    assert [c.text for c in chunks] == [s1, s2]


def test_chunks_never_cross_pages():
    chunks = chunk_sections([Section(words(50, "a"), page=1), Section(words(50, "b"), page=2)], 180, 30)
    assert [(c.page, c.text.split()[0][0]) for c in chunks] == [(1, "a"), (2, "b")]


def test_oversized_sentence_is_split():
    chunks = chunk_sections([Section(words(500))], chunk_size=100, overlap=10)
    assert all(c.word_count <= 100 for c in chunks)
    assert len(chunks) >= 5


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, 20), (10, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_sections([Section("x")], size, overlap)
