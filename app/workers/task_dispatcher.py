# app/workers/task_dispatcher.py
import logging
from typing import Optional
from uuid import UUID

from app.jobs.errors import TransitionConflict
from app.jobs.job_manager import JobManager
from app.jobs.job_repository import JobRepository
from app.workers.worker_executor import WorkerExecutor

logger = logging.getLogger(__name__)


class TaskDispatcher:
    """Claims PENDING jobs (atomically) and hands them to the WorkerExecutor."""

    @staticmethod
    def dispatch_job(job_id: UUID) -> None:
        """Claim one specific job (PENDING -> PROCESSING) and run it to the end.
        Raises JobNotFound / TransitionConflict if it is not claimable."""
        job = JobManager.claim_for_processing(job_id)
        WorkerExecutor.run_job(job_id, job.get("method", "method_1"))

    @staticmethod
    def dispatch_next_pending_job() -> Optional[str]:
        """Claim and run the oldest PENDING job. Returns its id, or None if the
        queue is empty. Losing a race for a job just moves on to the next one."""
        for job in JobRepository.get_pending_jobs(limit=5):
            job_id = UUID(job["job_id"])
            try:
                claimed = JobManager.claim_for_processing(job_id)
            except TransitionConflict:
                continue                      # another worker got it first
            WorkerExecutor.run_job(job_id, claimed.get("method", "method_1"))
            return str(job_id)
        return None
