"""Post-processing of generated answers: citation validation, renumbering and grounding checks."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass

from app.rag.calc import apply_calculations
from app.rag.prompts import INSUFFICIENT_EVIDENCE, Evidence

_CITATION_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
_REPEATED_MARKER_RE = re.compile(r"(\[\d+\])(?:\s*\1)+")


@dataclass(frozen=True)
class CheckedAnswer:
    text: str
    cited: list[Evidence]  # sources referenced by the answer, renumbered 1..n in order of first use
    insufficient: bool  # model signalled the evidence does not answer the question
    grounded: bool  # at least one valid citation backs the answer
    removed_citations: int  # references to non-existent sources that were stripped
    calculations: int = 0  # [[calc: ...]] expressions evaluated deterministically
    corrections: int = 0  # explicit equations whose wrong result was corrected


def is_insufficient(text: str) -> bool:
    return text.strip().strip(".").upper().startswith(INSUFFICIENT_EVIDENCE)


def check_answer(text: str, evidence: list[Evidence]) -> CheckedAnswer:
    """Validate citations against the evidence actually shown to the model.

    * References to source numbers that do not exist are removed (never displayed).
    * Remaining sources are renumbered 1..n in order of first appearance, so the numbers in the
      answer always match the displayed source list.
    * Repeated adjacent markers ("[1][1]") collapse to one.
    """
    if is_insufficient(text):
        return CheckedAnswer("", [], True, False, 0)
    text = text.replace(INSUFFICIENT_EVIDENCE, "").strip()  # sentinel appended to a real answer
    calc = apply_calculations(text)
    text = calc.text
    by_number = {e.number: e for e in evidence}
    order: dict[int, int] = {}  # original number -> new number
    removed = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal removed
        numbers = [int(n) for n in re.split(r"\s*,\s*", match.group(1))]
        out = []
        for n in numbers:
            if n not in by_number:
                removed += 1
                continue
            new = order.setdefault(n, len(order) + 1)
            if new not in out:
                out.append(new)
        return "".join(f"[{n}]" for n in sorted(out))

    cleaned = _CITATION_RE.sub(_replace, text)
    cleaned = _REPEATED_MARKER_RE.sub(r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned).strip()
    cited = [dataclasses.replace(by_number[old], number=new) for old, new in sorted(order.items(), key=lambda x: x[1])]
    return CheckedAnswer(cleaned, cited, False, bool(cited), removed, calc.calculations, calc.corrections)


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORDS_RE = re.compile(r"[a-z0-9]+")


def citing_text(answer: str, number: int) -> str:
    """The answer sentences that cite source ``number`` (used to pick that source's excerpt)."""
    marker = f"[{number}]"
    sentences = [s for s in _SENTENCE_RE.split(answer) if marker in s]
    return " ".join(sentences) or answer


def _similar(a: str, b: str, threshold: float = 0.6) -> bool:
    wa, wb = set(_WORDS_RE.findall(a.lower())), set(_WORDS_RE.findall(b.lower()))
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= threshold


def renumber(answer: str, mapping: dict[int, int]) -> tuple[str, dict[int, int]]:
    """Apply ``mapping`` (old -> canonical number) to the markers, then renumber 1..n by first
    appearance. Returns the new text and canonical -> final number."""
    order: dict[int, int] = {}

    def _replace(match: re.Match[str]) -> str:
        numbers = [mapping.get(int(n), int(n)) for n in re.split(r"\s*,\s*", match.group(1))]
        out = sorted({order.setdefault(n, len(order) + 1) for n in numbers})
        return "".join(f"[{n}]" for n in out)

    text = _REPEATED_MARKER_RE.sub(r"\1", _CITATION_RE.sub(_replace, answer))
    return text, order


def merge_duplicate_sources(answer: str, sources: list[dict]) -> tuple[str, list[dict]]:
    """Merge cited passages that show essentially the same text from the same page (overlapping
    chunks), so one fact is not presented as several independent sources."""
    mapping: dict[int, int] = {}
    kept: list[dict] = []
    for source in sources:
        twin = next((k for k in kept if k["document_id"] == source["document_id"]
                     and k["page_number"] == source["page_number"] and k["chunk_id"] and source["chunk_id"]
                     and _similar(k["excerpt"], source["excerpt"])), None)
        mapping[source["number"]] = twin["number"] if twin else source["number"]
        if twin is None:
            kept.append(source)
    if len(kept) == len(sources):
        return answer, sources
    text, final_numbers = renumber(answer, mapping)
    merged = [{**s, "number": final_numbers[s["number"]]} for s in kept if s["number"] in final_numbers]
    return text, sorted(merged, key=lambda s: s["number"])
