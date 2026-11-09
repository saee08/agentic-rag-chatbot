"""Document upload, listing and deletion (owner-scoped).

Uploads are streamed to a temporary file in the owner's directory while their size and SHA-256
are computed, then validated and renamed to a server-generated name. The client-supplied
filename is used only as a display label. Re-uploading identical content returns the existing
document (idempotent ingestion).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import (
    DocumentParsingError,
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    ValidationFailedError,
)
from app.core.logging import log_event
from app.models import Document, DocumentStatus, IngestionJob, JobStatus, new_id
from app.rag.parser import SUPPORTED_EXTENSIONS, file_extension, validate_file

logger = logging.getLogger(__name__)

_COPY_CHUNK = 1024 * 1024


def safe_display_name(filename: str | None) -> str:
    """Strip any path components and control characters from a client-supplied name."""
    name = (filename or "").replace("\\", "/").split("/")[-1]
    name = unicodedata.normalize("NFC", name)
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C").strip().strip(".")
    name = re.sub(r"\s+", " ", name)
    if len(name) > 200:
        stem, ext = os.path.splitext(name)
        name = stem[: 200 - len(ext)] + ext
    return name or "document"


class DocumentService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = Path(settings.upload_dir).resolve()

    def path_for(self, doc: Document) -> Path:
        path = (self.root / doc.storage_key).resolve()
        if not path.is_relative_to(self.root):  # storage keys are server-generated; belt and braces
            raise ValueError("storage path escapes the upload directory")
        return path

    def _owner_dir(self, owner_id: str) -> Path:
        directory = (self.root / owner_id).resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError("invalid owner directory")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def store_upload(self, session: Session, owner_id: str, filename: str | None,
                     stream: BinaryIO) -> tuple[Document, bool, IngestionJob | None]:
        """Persist one uploaded file. Returns (document, duplicate, job_to_enqueue)."""
        display = safe_display_name(filename)
        ext = file_extension(display)
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedMediaTypeError(
                f"'{display}': unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        directory = self._owner_dir(owner_id)
        digest = hashlib.sha256()
        size = 0
        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".upload-", suffix=ext)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as out:
                while block := stream.read(_COPY_CHUNK):
                    size += len(block)
                    if size > self.settings.max_upload_bytes:
                        raise PayloadTooLargeError(
                            f"'{display}' exceeds the {self.settings.max_upload_mb} MB upload limit."
                        )
                    digest.update(block)
                    out.write(block)
            if size == 0:
                raise ValidationFailedError(f"'{display}' is empty.")
            try:
                validate_file(tmp, ext)
            except DocumentParsingError as exc:
                raise ValidationFailedError(f"'{display}': {exc}") from exc
            except UnsupportedMediaTypeError as exc:
                raise UnsupportedMediaTypeError(f"'{display}': {exc.message}") from exc

            sha = digest.hexdigest()
            existing = self._find_by_hash(session, owner_id, sha)
            if existing is not None:
                return existing, True, None

            doc_id = new_id()
            storage_key = f"{owner_id}/{doc_id}{ext}"
            doc = Document(id=doc_id, owner_id=owner_id, filename=display, file_type=ext, size_bytes=size,
                           content_sha256=sha, storage_key=storage_key, status=DocumentStatus.PENDING)
            job = IngestionJob(document_id=doc_id, status=JobStatus.PENDING)
            session.add_all([doc, job])
            try:
                session.flush()
            except IntegrityError:  # identical content uploaded concurrently
                session.rollback()
                existing = self._find_by_hash(session, owner_id, sha)
                if existing is None:
                    raise
                return existing, True, None
            os.replace(tmp, self.path_for(doc))
            session.commit()
            log_event(logger, "document uploaded", document_id=doc_id, file_type=ext, size_bytes=size)
            return doc, False, job
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _find_by_hash(session: Session, owner_id: str, sha: str) -> Document | None:
        return session.scalar(select(Document).where(Document.owner_id == owner_id, Document.content_sha256 == sha))

    @staticmethod
    def list(session: Session, owner_id: str) -> list[Document]:
        return list(session.scalars(
            select(Document).where(Document.owner_id == owner_id).order_by(Document.created_at.desc())
        ))

    @staticmethod
    def get(session: Session, owner_id: str, document_id: str) -> Document:
        doc = session.scalar(select(Document).where(Document.id == document_id, Document.owner_id == owner_id))
        if doc is None:  # same response for "missing" and "someone else's": no enumeration
            raise NotFoundError("Document not found.")
        return doc

    def delete(self, session: Session, owner_id: str, document_id: str) -> None:
        doc = self.get(session, owner_id, document_id)
        path = self.path_for(doc)
        session.delete(doc)  # chunks, vectors and jobs cascade in the database
        session.commit()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log_event(logger, "stored file could not be removed", logging.WARNING, document_id=document_id)
        log_event(logger, "document deleted", document_id=document_id)

    def reprocess(self, session: Session, owner_id: str, document_id: str) -> tuple[Document, IngestionJob]:
        doc = self.get(session, owner_id, document_id)
        if doc.status in (DocumentStatus.PENDING, DocumentStatus.PROCESSING):
            raise ValidationFailedError("The document is already being processed.")
        doc.status = DocumentStatus.PENDING
        doc.error = None
        job = IngestionJob(document_id=doc.id, status=JobStatus.PENDING)
        session.add(job)
        session.commit()
        return doc, job
