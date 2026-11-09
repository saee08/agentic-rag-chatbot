"""In-process background ingestion.

Jobs are rows in ``ingestion_jobs`` (the durable queue); this worker executes them on a small
thread pool. On startup, jobs left pending/running by a previous process are re-queued, so a
crash or restart never strands a document. No external broker is needed for a single API
instance; for several instances, run one dedicated worker process (see README).
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from sqlalchemy import select

from app.core.logging import log_event
from app.db.session import Database
from app.models import IngestionJob
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(self, service: IngestionService, db: Database, workers: int = 1):
        self.service = service
        self.db = db
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest")
        self._pending: set[Future[None]] = set()
        self._lock = threading.Lock()
        self._closed = False

    def submit(self, job_id: str) -> None:
        with self._lock:
            if self._closed:
                return
            future = self._pool.submit(self._run, job_id)
            self._pending.add(future)
        future.add_done_callback(self._discard)

    def _discard(self, future: Future[None]) -> None:
        with self._lock:
            self._pending.discard(future)

    def _run(self, job_id: str) -> None:
        try:
            self.service.process(job_id)
        except Exception:  # the service records failures; this guards the pool thread itself
            logger.exception("ingestion worker crashed", extra={"fields": {"job_id": job_id}})

    def recover(self) -> int:
        """Re-queue jobs that never finished: pending ones, and running ones whose worker died
        (older than INGESTION_STALE_MINUTES). Safe with several processes: claiming is atomic."""
        with self.db.transaction() as session:
            ids = list(session.scalars(
                select(IngestionJob.id).where(self.service.claimable()).order_by(IngestionJob.created_at)
            ))
        for job_id in ids:
            self.submit(job_id)
        if ids:
            log_event(logger, "re-queued unfinished ingestion jobs", count=len(ids))
        return len(ids)

    def wait_idle(self, timeout: float = 60.0) -> bool:
        """Block until all submitted jobs finish (used by tests and graceful shutdown)."""
        with self._lock:
            pending = list(self._pending)
        for future in pending:
            try:
                future.result(timeout=timeout)
            except TimeoutError:
                return False
        return True

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=wait, cancel_futures=not wait)
