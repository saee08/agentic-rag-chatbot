"""Post-processing of generated answers: citation validation and grounding checks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.prompts import INSUFFICIENT_EVIDENCE, Evidence

_CITATION_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")


@dataclass(frozen=True)
class CheckedAnswer:
    text: str
    cited: list[Evidence]  # sources actually referenced, in source-number order
    insufficient: bool  # model signalled the evidence does not answer the question
    grounded: bool  # at least one valid citation backs the answer
    removed_citations: int  # references to non-existent sources that were stripped


def is_insufficient(text: str) -> bool:
    return text.strip().strip(".").upper().startswith(INSUFFICIENT_EVIDENCE)


def check_answer(text: str, evidence: list[Evidence]) -> CheckedAnswer:
    if is_insufficient(text):
        return CheckedAnswer("", [], True, False, 0)
    by_number = {e.number: e for e in evidence}
    used: set[int] = set()
    removed = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal removed
        numbers = [int(n) for n in re.split(r"\s*,\s*", match.group(1))]
        valid = [n for n in numbers if n in by_number]
        removed += len(numbers) - len(valid)
        used.update(valid)
        return "".join(f"[{n}]" for n in valid)

    cleaned = _CITATION_RE.sub(_replace, text)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned).strip()
    cited = [by_number[n] for n in sorted(used)]
    return CheckedAnswer(cleaned, cited, False, bool(cited), removed)
