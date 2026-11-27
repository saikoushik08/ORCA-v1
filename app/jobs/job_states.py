# app/jobs/job_states.py
from enum import Enum


class JobStatus(str, Enum):
    CREATED = "created"
    UPLOADING = "uploading"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# States in which image uploads are allowed
UPLOAD_ALLOWED_STATES = {JobStatus.CREATED, JobStatus.UPLOADING}

# States in which a job can be started (queued). CREATED is excluded on purpose:
# a job in CREATED has never received an image.
JOB_START_ALLOWED_STATES = {JobStatus.UPLOADING}

# Terminal states (no further transitions allowed)
TERMINAL_STATES = {JobStatus.COMPLETED, JobStatus.FAILED}

# Allowed transitions: current_state -> {next_states}.
# This table is the single rule book; JobManager and JobRepository.transition enforce it.
# CREATED -> PENDING is NOT allowed: a job must pass through UPLOADING (i.e. have images).
# PROCESSING -> PENDING is the one recovery edge, used only when a worker was interrupted.
ALLOWED_TRANSITIONS = {
    JobStatus.CREATED: {JobStatus.UPLOADING, JobStatus.FAILED},
    JobStatus.UPLOADING: {JobStatus.PENDING, JobStatus.FAILED},
    JobStatus.PENDING: {JobStatus.PROCESSING, JobStatus.FAILED},
    JobStatus.PROCESSING: {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.PENDING},
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
}


def is_terminal_state(status: JobStatus) -> bool:
    return status in TERMINAL_STATES


def can_upload_images(status: JobStatus) -> bool:
    return status in UPLOAD_ALLOWED_STATES


def can_start_job(status: JobStatus) -> bool:
    return status in JOB_START_ALLOWED_STATES


def is_valid_transition(current_status: JobStatus, next_status: JobStatus) -> bool:
    """Validate whether a job can move from current_status to next_status."""
    return next_status in ALLOWED_TRANSITIONS.get(current_status, set())
