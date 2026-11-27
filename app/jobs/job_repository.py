# app/jobs/job_repository.py
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from app.jobs.errors import InvalidTransition, JobNotFound, TransitionConflict
from app.jobs.job_states import JobStatus, is_valid_transition
from app.storage.supabase_client import supabase

logger = logging.getLogger(__name__)

# Columns that older databases may not have yet (see schema.sql). If the database
# rejects a write that contains them, the write is retried without them so the
# pipeline keeps working; run schema.sql to get the full feature set.
_OPTIONAL_JOB_COLUMNS = {
    "current_stage", "progress", "error_message", "error_code", "error_stage",
    "attempt", "started_at", "finished_at", "result", "preset", "input_type",
    "heartbeat_at", "owner_id", "parent_job_id",
}
_OPTIONAL_IMAGE_COLUMNS = {
    "original_name", "sha256", "width", "height", "size_bytes",
    "camera_key", "cloud_synced", "source_type", "source_video",
    "frame_index", "frame_time_s",
}
_warned_missing = set()


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _warn_once(key: str, message: str):
    if key not in _warned_missing:
        _warned_missing.add(key)
        logger.warning(message)


class JobRepository:
    """
    Handles ALL database operations related to jobs and job images.
    No API or worker should talk to Supabase directly.
    """

    # -----------------------------
    # JOB READS
    # -----------------------------

    @staticmethod
    def get_job(job_id: UUID) -> Optional[dict]:
        # .limit(1) instead of .single(): single() raises when no row exists.
        response = (
            supabase.table("jobs").select("*").eq("job_id", str(job_id)).limit(1).execute()
        )
        rows = response.data or []
        return rows[0] if rows else None

    @staticmethod
    def get_jobs_by_status(status: JobStatus, limit: int = 50) -> list:
        response = (
            supabase.table("jobs")
            .select("*")
            .eq("status", status.value)
            .order("created_at")
            .limit(limit)
            .execute()
        )
        return response.data or []

    @staticmethod
    def get_pending_jobs(limit: int = 1) -> list:
        """Oldest PENDING jobs first."""
        return JobRepository.get_jobs_by_status(JobStatus.PENDING, limit)

    # -----------------------------
    # JOB WRITES
    # -----------------------------

    @staticmethod
    def create_job(
        job_id: UUID,
        method: str,
        preset: str = "standard",
        input_type: str = "photos",
        parent_job_id: Optional[UUID] = None,
        owner_id: Optional[UUID] = None,
    ) -> dict:
        payload = {
            "job_id": str(job_id),
            "method": method,
            "status": JobStatus.CREATED.value,
            "created_at": utcnow_iso(),
            "preset": preset,
            "input_type": input_type,
        }
        if parent_job_id is not None:
            payload["parent_job_id"] = str(parent_job_id)
        if owner_id is not None:
            payload["owner_id"] = str(owner_id)
        try:
            response = supabase.table("jobs").insert(payload).execute()
        except Exception as exc:
            # Fall back for older database schemas without preset/input_type
            core = {
                "job_id": str(job_id),
                "method": method,
                "status": JobStatus.CREATED.value,
                "created_at": utcnow_iso(),
            }
            _warn_once("jobs_cols_create", f"Database lacks new job columns ({exc}); run schema.sql.")
            response = supabase.table("jobs").insert(core).execute()
        if not response.data:
            raise RuntimeError("Database did not confirm job creation")
        return response.data[0] if isinstance(response.data, list) else response.data

    @staticmethod
    def _update_where(job_id: UUID, payload: dict, from_status: Optional[JobStatus]) -> list:
        """UPDATE jobs SET payload WHERE job_id = ? [AND status = from_status].
        Retries without optional columns if the database does not have them yet."""

        def run(data):
            q = supabase.table("jobs").update(data).eq("job_id", str(job_id))
            if from_status is not None:
                q = q.eq("status", from_status.value)
            return q.execute().data or []

        try:
            return run(payload)
        except Exception as exc:
            extras = set(payload) & _OPTIONAL_JOB_COLUMNS
            if not extras:
                raise
            _warn_once(
                "jobs_cols",
                f"Database rejected optional job columns {sorted(extras)} ({exc}). "
                "Run schema.sql to enable stage, progress and failure reasons.",
            )
            core = {k: v for k, v in payload.items() if k not in _OPTIONAL_JOB_COLUMNS}
            return run(core)

    @staticmethod
    def transition(job_id: UUID, from_status: JobStatus, to_status: JobStatus, **fields) -> dict:
        """Atomic, validated state change in ONE database call.

        The UPDATE only matches a row that is still in `from_status`, so if two
        callers race, exactly one wins; the other gets TransitionConflict.
        """
        if not is_valid_transition(from_status, to_status):
            raise InvalidTransition(
                f"Illegal transition '{from_status.value}' -> '{to_status.value}'"
            )
        payload = {"status": to_status.value, "updated_at": utcnow_iso(), **fields}
        rows = JobRepository._update_where(job_id, payload, from_status)
        if rows:
            JobRepository.log_event(
                job_id, "transition", from_status=from_status.value,
                to_status=to_status.value, message=fields.get("error_message"),
            )
            return rows[0]

        current = JobRepository.get_job(job_id)
        if current is None:
            raise JobNotFound("Job not found")
        raise TransitionConflict(
            f"Job is '{current['status']}', expected '{from_status.value}'"
        )

    @staticmethod
    def update_status(job_id: UUID, next_status: JobStatus, **fields) -> list:
        """Unconditional status write (kept for scripts/tests). Prefer transition()."""
        job = JobRepository.get_job(job_id)
        if not job:
            raise JobNotFound("Job not found")
        payload = {"status": next_status.value, "updated_at": utcnow_iso(), **fields}
        return JobRepository._update_where(job_id, payload, None)

    @staticmethod
    def update_fields(job_id: UUID, **fields) -> None:
        """Update non-status fields (stage, progress, result...). Never raises:
        progress reporting must not be able to kill a running job."""
        try:
            payload = {"updated_at": utcnow_iso(), **fields}
            JobRepository._update_where(job_id, payload, None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not update job fields %s: %s", sorted(fields), exc)

    # -----------------------------
    # AUDIT TRAIL (best effort)
    # -----------------------------

    @staticmethod
    def log_event(job_id: UUID, event_type: str, from_status: str | None = None,
                  to_status: str | None = None, stage: str | None = None,
                  message: str | None = None, data: dict | None = None) -> None:
        try:
            supabase.table("job_events").insert({
                "job_id": str(job_id), "event_type": event_type,
                "from_status": from_status, "to_status": to_status,
                "stage": stage, "message": message, "data": data,
            }).execute()
        except Exception as exc:  # noqa: BLE001
            _warn_once("job_events", f"job_events table not available ({exc}); run schema.sql.")

    @staticmethod
    def list_events(job_id: UUID, limit: int = 200) -> list:
        try:
            response = (
                supabase.table("job_events").select("*").eq("job_id", str(job_id))
                .order("id").limit(limit).execute()
            )
            return response.data or []
        except Exception:  # noqa: BLE001
            return []

    # -----------------------------
    # IMAGE OPERATIONS
    # -----------------------------

    @staticmethod
    def count_images(job_id: UUID) -> int:
        response = (
            supabase.table("images").select("id", count="exact")
            .eq("job_id", str(job_id)).execute()
        )
        return response.count or 0

    @staticmethod
    def add_image(job_id: UUID, image_path: str, **extra):
        """Insert an images row. `extra` may hold original_name, sha256, width,
        height, size_bytes; they are dropped automatically on older schemas."""
        base = {
            "job_id": str(job_id),
            "image_path": image_path,
            "uploaded_at": utcnow_iso(),
        }
        try:
            return supabase.table("images").insert({**base, **extra}).execute().data
        except Exception as exc:
            if not (set(extra) & _OPTIONAL_IMAGE_COLUMNS):
                raise
            _warn_once("images_cols", f"images table lacks optional columns ({exc}); run schema.sql.")
            return supabase.table("images").insert(base).execute().data

    @staticmethod
    def list_images(job_id: UUID) -> list:
        response = (
            supabase.table("images").select("*").eq("job_id", str(job_id)).execute()
        )
        return response.data or []
