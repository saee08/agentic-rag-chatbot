"""Parse -> chunk -> embed -> persist, for one ingestion job.

Idempotent: chunk ids are uuid5(document_id, ordinal) and existing chunks are replaced inside the
same transaction that marks the document READY, so a retried or repeated job never duplicates
vectors and readers never see a half-indexed document.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import and_, delete, or_, select, update
from sqlalchemy.exc import IntegrityError

from app.core.config import Settings
from app.core.exceptions import DocumentParsingError, UnsupportedMediaTypeError
from app.core.logging import log_event, timed
from app.db.session import Database
from app.models import Chunk, Document, DocumentStatus, IngestionJob, JobStatus
from app.rag.chunker import chunk_sections
from app.rag.embeddings import Embedder
from app.rag.parser import parse_file
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)

_CHUNK_NAMESPACE = uuid.UUID("8f9d6a52-3c1e-4c55-9a43-5b8e0f1f2a77")
_GENERIC_FAILURE = "Processing failed due to an internal error. Please try again later."


def chunk_id(document_id: str, ordinal: int) -> str:
    return str(uuid.uuid5(_CHUNK_NAMESPACE, f"{document_id}:{ordinal}"))


class IngestionService:
    def __init__(self, settings: Settings, db: Database, embedder: Embedder, documents: DocumentService):
        self.settings = settings
        self.db = db
        self.embedder = embedder
        self.documents = documents

    def claimable(self) -> object:
        """SQL condition for jobs that may be (re)started: pending, or running but stale
        (its worker died). Shared by claiming and startup recovery."""
        stale_before = datetime.now(UTC) - timedelta(minutes=self.settings.ingestion_stale_minutes)
        return or_(
            IngestionJob.status == JobStatus.PENDING,
            and_(IngestionJob.status == JobStatus.RUNNING,
                 or_(IngestionJob.started_at.is_(None), IngestionJob.started_at < stale_before)),
        )

    def _claim(self, job_id: str) -> tuple[str, str, str] | None:
        """Atomically mark the job running. A conditional UPDATE means that when several
        processes race for the same job, exactly one wins. Returns (document_id, owner_id,
        storage path) or None if the job is not claimable."""
        with self.db.transaction() as session:
            claimed = session.execute(
                update(IngestionJob)
                .where(IngestionJob.id == job_id, self.claimable())
                .values(status=JobStatus.RUNNING, attempts=IngestionJob.attempts + 1, started_at=datetime.now(UTC))
                .execution_options(synchronize_session=False)
            ).rowcount
            if claimed != 1:
                return None
            job = session.get(IngestionJob, job_id)
            doc = session.get(Document, job.document_id) if job else None
            if doc is None:
                return None
            doc.status = DocumentStatus.PROCESSING
            doc.error = None
            return doc.id, doc.owner_id, str(self.documents.path_for(doc))

    def _finish(self, job_id: str, document_id: str, *, error: str | None, timings: dict[str, float]) -> None:
        with self.db.transaction() as session:
            job = session.get(IngestionJob, job_id)
            doc = session.get(Document, document_id)
            if job is not None:
                job.status = JobStatus.FAILED if error else JobStatus.SUCCEEDED
                job.error = error
                job.finished_at = datetime.now(UTC)
                job.timings_ms = timings
            if doc is not None and error:
                doc.status = DocumentStatus.FAILED
                doc.error = error

    def process(self, job_id: str) -> None:
        claimed = self._claim(job_id)
        if claimed is None:
            return
        document_id, owner_id, path_str = claimed
        timings: dict[str, float] = {}
        path = Path(path_str)
        try:
            with self.db.transaction() as session:
                doc = session.get(Document, document_id)
                ext = doc.file_type if doc else path.suffix
            with timed() as t:
                parsed = parse_file(path, ext, max_pages=self.settings.max_pdf_pages,
                                    max_chars=self.settings.max_extracted_chars)
            timings["parse_ms"] = t["ms"]
            with timed() as t:
                pieces = chunk_sections(parsed.sections, self.settings.chunk_size_words,
                                        self.settings.chunk_overlap_words)
            timings["chunk_ms"] = t["ms"]
            if not pieces:
                raise DocumentParsingError("No extractable text was found in the document.")
            with timed() as t:
                vectors = self.embedder.embed([p.text for p in pieces])
            timings["embed_ms"] = t["ms"]

            with timed() as t, self.db.transaction() as session:
                doc = session.get(Document, document_id)
                if doc is None:  # deleted while we were working
                    log_event(logger, "ingestion skipped: document deleted", document_id=document_id)
                    return
                session.execute(delete(Chunk).where(Chunk.document_id == document_id))
                session.add_all([
                    Chunk(id=chunk_id(document_id, i), document_id=document_id, owner_id=owner_id, ordinal=i,
                          page_number=p.page, text=p.text, word_count=p.word_count, embedding=vectors[i])
                    for i, p in enumerate(pieces)
                ])
                doc.status = DocumentStatus.READY
                doc.page_count = parsed.page_count
                doc.chunk_count = len(pieces)
                doc.embedding_model = self.embedder.model_name
                doc.error = None
            timings["persist_ms"] = t["ms"]
            self._finish(job_id, document_id, error=None, timings=timings)
            log_event(logger, "ingestion succeeded", document_id=document_id, chunks=len(pieces),
                      pages=parsed.page_count, **timings)
        except (DocumentParsingError, UnsupportedMediaTypeError) as exc:
            message = exc.message if isinstance(exc, UnsupportedMediaTypeError) else str(exc)
            log_event(logger, "ingestion rejected document", logging.WARNING, document_id=document_id,
                      reason=message)
            self._finish(job_id, document_id, error=message, timings=timings)
        except IntegrityError:
            # Most likely the document was deleted concurrently (FK violation): nothing to keep.
            if self._exists(document_id):
                logger.exception("ingestion integrity error", extra={"fields": {"document_id": document_id}})
                self._finish(job_id, document_id, error=_GENERIC_FAILURE, timings=timings)
        except Exception:
            logger.exception("ingestion failed", extra={"fields": {"document_id": document_id}})
            self._finish(job_id, document_id, error=_GENERIC_FAILURE, timings=timings)

    def _exists(self, document_id: str) -> bool:
        with self.db.transaction() as session:
            return session.scalar(select(Document.id).where(Document.id == document_id)) is not None
