"""Keyword (full-text) retrieval, used alongside vector search.

Dense embeddings are weak at exact identifiers ("NH-412") and specific phrases ("reorder
point"); measured on the evaluation PDF, the only passage answering "What is the exact reorder
point for NH-412?" ranked last by cosine similarity (0.18, below the relevance floor). Keyword
search recovers such passages.

Candidates come from the database's own full-text engine (PostgreSQL ``tsvector`` + GIN index,
SQLite FTS5), always filtered by owner. A candidate only counts as relevant when it covers enough
of the query's content terms, so a single shared word ("point" in "boiling point of tungsten")
cannot drag in an unrelated passage.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from app.models import DocumentStatus

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MAX_TERMS = 12
MIN_COVERAGE = 0.5

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers him his how i if in into is it its itself just me more most my no nor not now
    of off on once only or other our ours out over own same she should so some such than that the their
    theirs them then there these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours tell give show list describe explain
    please know find mentioned document documents file files say says said""".split()
)


def stem(word: str) -> str:
    """Light suffix stripping so 'quantities'/'quantity' and 'located'/'locate' match."""
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)] + replacement
    return word


def query_terms(query: str) -> list[str]:
    seen: dict[str, None] = {}
    for token in _TOKEN_RE.findall(query.lower()):
        if token not in STOPWORDS and (len(token) > 1 or token.isdigit()):
            seen.setdefault(token, None)
    return list(seen)[:_MAX_TERMS]


def coverage(terms: list[str], passage: str) -> tuple[int, float]:
    """(number of query terms present in the passage, fraction of terms present)."""
    if not terms:
        return 0, 0.0
    stems = {stem(t) for t in _TOKEN_RE.findall(passage.lower())}
    matched = sum(1 for t in terms if stem(t) in stems)
    return matched, matched / len(terms)


def is_relevant(terms: list[str], passage: str) -> bool:
    matched, fraction = coverage(terms, passage)
    return matched >= min(2, len(terms)) and fraction >= MIN_COVERAGE


@dataclass(frozen=True)
class KeywordHit:
    chunk_id: str
    rank: int  # 1-based rank from the full-text engine


class KeywordIndex:
    def search(self, session: Session, owner_id: str, query: str, limit: int,
               document_ids: Sequence[str] | None = None) -> list[KeywordHit]:
        terms = query_terms(query)
        if not terms or limit <= 0 or (document_ids is not None and not document_ids):
            return []
        dialect = session.get_bind().dialect.name
        if dialect == "postgresql":
            rows = self._search_pg(session, owner_id, terms, limit * 3, document_ids)
        elif dialect == "sqlite":
            rows = self._search_sqlite(session, owner_id, terms, limit * 3, document_ids)
        else:
            return []
        relevant = [cid for cid, passage in rows if is_relevant(terms, passage)]
        return [KeywordHit(cid, i) for i, cid in enumerate(relevant[:limit], start=1)]

    @staticmethod
    def _filters(document_ids: Sequence[str] | None) -> str:
        # Constant SQL fragment; all values are bound parameters (no user input is interpolated).
        return " AND c.document_id IN :doc_ids" if document_ids is not None else ""

    def _search_sqlite(self, session, owner_id, terms, limit, document_ids):  # type: ignore[no-untyped-def]
        # Each term is a quoted FTS5 string, so user input can never inject FTS query syntax.
        match = " OR ".join('"' + t.replace('"', "") + '"' for t in terms)
        sql = text(
            "SELECT c.id, c.text FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid "
            "JOIN documents d ON d.id = c.document_id "
            "WHERE chunks_fts MATCH :match AND c.owner_id = :owner AND d.owner_id = :owner "
            f"AND d.status = :ready{self._filters(document_ids)} ORDER BY bm25(chunks_fts) LIMIT :limit"
        )
        params = {"match": match, "owner": owner_id, "ready": DocumentStatus.READY.value, "limit": limit}
        if document_ids is not None:
            sql = sql.bindparams(bindparam("doc_ids", expanding=True))
            params["doc_ids"] = list(document_ids)
        return [(r[0], r[1]) for r in session.execute(sql, params)]

    def _search_pg(self, session, owner_id, terms, limit, document_ids):  # type: ignore[no-untyped-def]
        # websearch_to_tsquery never raises on user input; "or" joins the terms.
        sql = text(
            "SELECT c.id, c.text FROM chunks c JOIN documents d ON d.id = c.document_id, "
            "websearch_to_tsquery('english', :q) q "
            "WHERE to_tsvector('english', c.text) @@ q AND c.owner_id = :owner AND d.owner_id = :owner "
            f"AND d.status = :ready{self._filters(document_ids)} "
            "ORDER BY ts_rank_cd(to_tsvector('english', c.text), q) DESC LIMIT :limit"
        )
        params = {"q": " or ".join(terms), "owner": owner_id, "ready": DocumentStatus.READY.value, "limit": limit}
        if document_ids is not None:
            sql = sql.bindparams(bindparam("doc_ids", expanding=True))
            params["doc_ids"] = list(document_ids)
        return [(r[0], r[1]) for r in session.execute(sql, params)]

