"""Smoke test of the job queue WITHOUT COLMAP: creates a job with no real images,
queues it and lets the worker run it. Expected result: status 'failed' with a
stored reason (missing images). Proves state transitions + failure reasons work.

    python -m scripts.queue_smoke_test
"""
from uuid import uuid4

from app.jobs.job_manager import JobManager
from app.jobs.job_repository import JobRepository
from app.logging_config import setup_logging
from app.workers.task_dispatcher import TaskDispatcher


def main():
    setup_logging("INFO")
    job_id = uuid4()
    JobRepository.create_job(job_id, "method_1")
    JobManager.mark_uploading(job_id)
    for i in range(5):
        JobRepository.add_image(job_id, f"{job_id}/image_{i + 1}.jpg")
    JobManager.start_job(job_id)
    print("Queued:", JobRepository.get_job(job_id)["status"])

    TaskDispatcher.dispatch_job(job_id)

    job = JobRepository.get_job(job_id)
    print("Final status:", job["status"])
    print("Reason      :", job.get("error_message"))


if __name__ == "__main__":
    main()
