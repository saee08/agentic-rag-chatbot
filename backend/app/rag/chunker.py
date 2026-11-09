"""Sentence-aware chunking with word-based size and overlap.

* Chunks never span two pages/slides, so each chunk has an exact page reference.
* Sentences are packed whole while they fit; a sentence longer than ``size - overlap`` words
  is split on word boundaries, so the overlap carried into the next chunk always fits.
* Consecutive chunks share ~``overlap`` trailing words so facts on a boundary stay retrievable.

Default size (180 words) keeps chunks under the 256 word-piece input limit of
all-MiniLM-L6-v2; longer chunks would be silently truncated by the embedding model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.parser import Section

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|\n+")


@dataclass(frozen=True)
class TextChunk:
    text: str
    page: int | None
    word_count: int


def _split_units(text: str, max_words: int) -> list[list[str]]:
    """Sentences (or lines) as word lists, with oversize sentences cut to max_words."""
    units: list[list[str]] = []
    for sentence in _SENTENCE_END.split(text):
        words = sentence.split()
        for start in range(0, len(words), max_words):
            piece = words[start : start + max_words]
            if piece:
                units.append(piece)
    return units


def chunk_sections(sections: list[Section], chunk_size: int, overlap: int) -> list[TextChunk]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("require chunk_size > 0 and 0 <= overlap < chunk_size")

    chunks: list[TextChunk] = []
    for section in sections:
        current: list[str] = []
        new_words = 0  # words in `current` not carried over from the previous chunk
        for unit in _split_units(section.text, chunk_size - overlap):
            if current and len(current) + len(unit) > chunk_size:
                chunks.append(TextChunk(" ".join(current), section.page, len(current)))
                current = current[-overlap:] if overlap else []
                # Never let carried overlap plus the next unit exceed the size limit.
                if len(current) + len(unit) > chunk_size:
                    current = current[len(current) + len(unit) - chunk_size :]
                new_words = 0
            current.extend(unit)
            new_words += len(unit)
        if current and new_words:
            chunks.append(TextChunk(" ".join(current), section.page, len(current)))
    return chunks
