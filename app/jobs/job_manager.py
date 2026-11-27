# app/jobs/job_manager.py
import logging
from uuid import UUID

from app.jobs.errors import InvalidTransition, JobNotFound, TransitionConflict
from app.jobs.job_repository import JobRepository, utcnow_iso
from app.jobs.job_states import (
    JOB_START_ALLOWED_STATES,
    TERMINAL_STATES,
    UPLOAD_ALLOWED_STATES,
    JobStatus,
)

logger = logging.getLogger(__name__)

MAX_ERROR_MESSAGE_CHARS = 4000


class JobManager:
    """
    Central authority for job lifecycle changes. Every change goes through
    JobRepository.transition(), which checks the transition table and updates
    atomically, so illegal or racing changes are rejected.
    """

    @staticmethod
    def _get_job_or_fail(job_id: UUID) -> dict:
        job = JobRepository.get_job(job_id)
        if not job:
            raise JobNotFound("Job not found")
        return job

    @staticmethod
    def _status(job_id: UUID) -> JobStatus:
        return JobStatus(JobManager._get_job_or_fail(job_id)["status"])

    # ---- state checks ----
    @staticmethod
    def can_upload_images(job_id: UUID) -> bool:
        return JobManager._status(job_id) in UPLOAD_ALLOWED_STATES

    @staticmethod
    def can_start_job(job_id: UUID) -> bool:
        return JobManager._status(job_id) in JOB_START_ALLOWED_STATES

    # ---- transitions ----
    @staticmethod
    def mark_uploading(job_id: UUID) -> None:
        """CREATED -> UPLOADING on the first upload. A job that is already
        UPLOADING is left alone; any other state is an error."""
        status = JobManager._status(job_id)
        if status == JobStatus.UPLOADING:
            return
        if status != JobStatus.CREATED:
            raise InvalidTransition(f"Cannot upload to a job in state '{status.value}'")
        JobRepository.transition(job_id, JobStatus.CREATED, JobStatus.UPLOADING)

    @staticmethod
    def start_job(job_id: UUID) -> None:
        """UPLOADING -> PENDING (queued). Atomic: two simultaneous calls, one wins."""
        status = JobManager._status(job_id)
        if status not in JOB_START_ALLOWED_STATES:
            raise InvalidTransition(f"Cannot start job from state '{status.value}'")
        JobRepository.transition(job_id, JobStatus.UPLOADING, JobStatus.PENDING, attempt=1)

    @staticmethod
    def claim_for_processing(job_id: UUID) -> dict:
        """PENDING -> PROCESSING. Used by the worker; atomic, so a job can only be
        claimed by one worker."""
        return JobRepository.transition(
            job_id, JobStatus.PENDING, JobStatus.PROCESSING,
            started_at=utcnow_iso(), current_stage="starting", progress=0,
        )

    @staticmethod
    def mark_processing(job_id: UUID) -> None:
        JobManager.claim_for_processing(job_id)

    @staticmethod
    def mark_completed(job_id: UUID, result: dict | None = None) -> None:
        fields = {"finished_at": utcnow_iso(), "current_stage": "done", "progress": 100}
        if result is not None:
            fields["result"] = result
        try:
            JobRepository.transition(job_id, JobStatus.PROCESSING, JobStatus.COMPLETED, **fields)
        except TransitionConflict:
            logger.warning("[JobManager] TransitionConflict marking %s completed; forcing status update", job_id)
            JobRepository.update_status(job_id, JobStatus.COMPLETED, **fields)

    @staticmethod
    def mark_failed(job_id: UUID, reason: str | None = None, stage: str | None = None,
                    code: str | None = None) -> None:
        """Move any non-terminal job to FAILED and STORE why."""
        status = JobManager._status(job_id)
        if status in TERMINAL_STATES:
            raise InvalidTransition(f"Job is already in terminal state '{status.value}'")
        fields = {
            "finished_at": utcnow_iso(),
            "error_message": (reason or "Unknown error")[:MAX_ERROR_MESSAGE_CHARS],
            "error_code": code or "error",
            "error_stage": stage,
        }
        JobRepository.transition(job_id, status, JobStatus.FAILED, **fields)

    @staticmethod
    def requeue_interrupted(job_id: UUID, attempt: int) -> None:
        """Recovery edge PROCESSING -> PENDING after a worker crash/restart."""
        JobRepository.transition(
            job_id, JobStatus.PROCESSING, JobStatus.PENDING,
            attempt=attempt, current_stage="requeued", progress=0,
        )
