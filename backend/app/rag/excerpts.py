"""Pick the most relevant part of a cited passage to show by default.

The excerpt is always a verbatim span of the passage (with "…" where it was cut). Terms from the
answer weigh more than terms from the question, so the excerpt shows the evidence for what was
actually said.
"""

from __future__ import annotations

import re

from app.rag.keyword_search import query_terms, stem

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITATION_RE = re.compile(r"\[\d+\]")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _weights(question: str, answer: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for term in query_terms(question):
        weights[stem(term)] = 1.0
    for term in query_terms(_CITATION_RE.sub(" ", answer)):
        weights[stem(term)] = weights.get(stem(term), 0.0) + 2.0
    return weights


def _score(words: list[str], weights: dict[str, float]) -> float:
    stems = {stem(w) for w in (_WORD_RE.findall(" ".join(words).lower()))}
    return sum(weight for term, weight in weights.items() if term in stems)


def relevant_excerpt(passage: str, question: str, answer: str = "", max_words: int = 45) -> str:
    words_all = passage.split()
    if len(words_all) <= max_words:
        return passage.strip()
    weights = _weights(question, answer)

    # Candidate spans: each sentence; sentences longer than max_words become sliding windows.
    spans: list[tuple[int, int]] = []
    position = 0
    for sentence in _SENTENCE_SPLIT.split(passage.strip()):
        n = len(sentence.split())
        if n == 0:
            continue
        if n <= max_words:
            spans.append((position, position + n))
        else:
            step = max(5, max_words // 3)
            for start in range(position, position + n - max_words + step, step):
                spans.append((start, min(start + max_words, position + n)))
        position += n

    def key(span: tuple[int, int]) -> tuple[float, int]:
        return _score(words_all[span[0]:span[1]], weights), -(span[1] - span[0])

    start, end = max(spans, key=key) if weights else (0, max_words)
    if key((start, end))[0] == 0:
        start, end = 0, max_words
    # Extend a short best sentence with its neighbour so the excerpt has some context.
    if end - start < max_words // 2 and end < len(words_all):
        end = min(len(words_all), start + max_words // 2 + (end - start))
    text = " ".join(words_all[start:end])
    return ("… " if start > 0 else "") + text + (" …" if end < len(words_all) else "")
