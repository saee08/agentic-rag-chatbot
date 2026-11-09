"""Vector search over the ``chunks`` table.

Vectors live in the same row as their chunk text and metadata, so inserts and deletes are
transactional: there is no separate index that can drift out of sync.

* PostgreSQL: pgvector cosine distance (``<=>``) with an HNSW index, filtered by owner in SQL.
* Other databases (SQLite for dev/tests): owner-filtered rows are scored with NumPy in
  batches. This is exact search and fine for tens of thousands of chunks per user; use
  PostgreSQL for larger collections.

Every query is scoped by ``owner_id`` and only searches documents whose status is READY.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sqlalchemy import Float, bindparam, select
from sqlalchemy.orm import Session

from app.db.types import EmbeddingVector
from app.models import Chunk, Document, DocumentStatus

_SCAN_BATCH = 2048


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None
    ordinal: int
    text: str
    score: float  # cosine similarity in [-1, 1]


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim

    def search(
        self,
        session: Session,
        owner_id: str,
        query_vector: np.ndarray,
        top_k: int,
        min_score: float,
        document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        if top_k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(f"query vector has dimension {query.shape[0]}, expected {self.dim}")
        if document_ids is not None and not document_ids:
            return []
        if session.get_bind().dialect.name == "postgresql":
            return self._search_pg(session, owner_id, query, top_k, min_score, document_ids)
        return self._search_scan(session, owner_id, query, top_k, min_score, document_ids)

    def _base_filter(self, stmt, owner_id: str, document_ids: Sequence[str] | None):  # type: ignore[no-untyped-def]
        stmt = stmt.join(Document, Document.id == Chunk.document_id).where(
            Chunk.owner_id == owner_id,
            Document.owner_id == owner_id,  # defence in depth
            Document.status == DocumentStatus.READY,
        )
        if document_ids is not None:
            stmt = stmt.where(Chunk.document_id.in_(list(document_ids)))
        return stmt

    def _search_pg(self, session, owner_id, query, top_k, min_score, document_ids) -> list[ScoredChunk]:  # type: ignore[no-untyped-def]
        q = bindparam("query_vec", value=query, type_=EmbeddingVector(self.dim))
        distance = Chunk.embedding.op("<=>", return_type=Float)(q)
        stmt = select(
            Chunk.id, Chunk.document_id, Document.filename, Chunk.page_number, Chunk.ordinal, Chunk.text,
            distance.label("distance"),
        )
        stmt = self._base_filter(stmt, owner_id, document_ids).order_by(distance).limit(top_k)
        results = []
        for row in session.execute(stmt):
            score = 1.0 - float(row.distance)
            if score >= min_score:
                results.append(ScoredChunk(row.id, row.document_id, row.filename, row.page_number,
                                           row.ordinal, row.text, round(score, 4)))
        return results

    def _search_scan(self, session, owner_id, query, top_k, min_score, document_ids) -> list[ScoredChunk]:  # type: ignore[no-untyped-def]
        stmt = self._base_filter(select(Chunk.id, Chunk.embedding), owner_id, document_ids)
        best: list[tuple[float, str]] = []
        result = session.execute(stmt.execution_options(yield_per=_SCAN_BATCH))
        for batch in result.partitions(_SCAN_BATCH):
            ids = [r.id for r in batch]
            matrix = np.vstack([r.embedding for r in batch])
            scores = matrix @ query
            best.extend(zip(scores.tolist(), ids, strict=True))
            if len(best) > top_k * 4:
                best = sorted(best, reverse=True)[:top_k]
        best = [b for b in sorted(best, reverse=True)[:top_k] if b[0] >= min_score]
        if not best:
            return []
        rows = {
            r.id: r
            for r in session.execute(
                select(Chunk.id, Chunk.document_id, Document.filename, Chunk.page_number, Chunk.ordinal,
                       Chunk.text)
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.id.in_([cid for _, cid in best]), Chunk.owner_id == owner_id)
            )
        }
        return [
            ScoredChunk(cid, rows[cid].document_id, rows[cid].filename, rows[cid].page_number,
                        rows[cid].ordinal, rows[cid].text, round(float(score), 4))
            for score, cid in best
            if cid in rows
        ]
