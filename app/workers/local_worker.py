# app/workers/local_worker.py
"""Worker loop: picks up PENDING jobs and runs them, one at a time.

Run it as its own process (recommended for production):
    EMBEDDED_WORKER=false uvicorn app.main:app
    python -m app.workers.local_worker

or let the API process run it in a background thread (EMBEDDED_WORKER=true, default).
Run only ONE worker per GPU: on start-up it requeues jobs left in PROCESSING by a
crashed run, which would steal a job that another live worker is processing.
"""
import logging
import threading
from uuid import UUID

from app.config import settings
from app.jobs.job_manager import JobManager
from app.jobs.job_repository import JobRepository
from app.jobs.job_states import JobStatus
from app.logging_config import setup_logging
from app.workers.task_dispatcher import TaskDispatcher

logger = logging.getLogger(__name__)


class LocalWorker:
    @staticmethod
    def execute(job_id: UUID) -> None:
        """Run one specific PENDING job (used by scripts)."""
        TaskDispatcher.dispatch_job(job_id)

    @staticmethod
    def recover_interrupted() -> int:
        """Jobs still in PROCESSING when the worker starts were interrupted. Requeue
        them (they resume from scratch) up to MAX_JOB_ATTEMPTS, then fail them."""
        handled = 0
        for job in JobRepository.get_jobs_by_status(JobStatus.PROCESSING, limit=100):
            job_id = UUID(job["job_id"])
            attempt = job.get("attempt")
            try:
                if attempt is None or attempt >= settings.max_job_attempts:
                    # Without the 'attempt' column we cannot count retries: do not loop forever.
                    JobManager.mark_failed(
                        job_id, "The worker was interrupted while processing this job.",
                        stage=job.get("current_stage"), code="worker_interrupted")
                else:
                    JobManager.requeue_interrupted(job_id, attempt + 1)
                handled += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not recover job %s: %s", job_id, exc)
        if handled:
            logger.warning("Recovered %d interrupted job(s)", handled)
        return handled

    @staticmethod
    def run_forever(stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event or threading.Event()
        logger.info("[Worker] Loop started (poll every %.1fs)", settings.worker_poll_seconds)
        try:
            LocalWorker.recover_interrupted()
        except Exception:  # noqa: BLE001
            logger.exception("[Worker] Recovery failed")
        while not stop_event.is_set():
            try:
                job_id = TaskDispatcher.dispatch_next_pending_job()
            except Exception:  # noqa: BLE001
                logger.exception("[Worker] Error while dispatching")
                job_id = None
            if job_id is None:
                stop_event.wait(settings.worker_poll_seconds)
        logger.info("[Worker] Loop stopped")

    @staticmethod
    def start_in_background() -> tuple:
        stop = threading.Event()
        thread = threading.Thread(target=LocalWorker.run_forever, args=(stop,),
                                  name="orca-worker", daemon=True)
        thread.start()
        return thread, stop


def main() -> None:
    setup_logging(settings.log_level)
    LocalWorker.run_forever()


if __name__ == "__main__":
    main()
