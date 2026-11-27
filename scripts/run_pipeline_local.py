"""Run the full Method-1 pipeline on a folder of photos, without the HTTP API.

Usage (from the project root, venv active, COLMAP + CUDA installed):
    python -m scripts.run_pipeline_local path/to/photos

Needs a working .env (Supabase) because job state is stored in the database.
"""
import shutil
import sys
from pathlib import Path
from uuid import uuid4

from app.config import settings
from app.jobs.job_manager import JobManager
from app.jobs.job_repository import JobRepository
from app.logging_config import setup_logging
from app.services.method1_photogrammetry.colmap_runner import IMAGE_EXTENSIONS
from app.workers.local_worker import LocalWorker


def main(source: Path) -> int:
    setup_logging("INFO")
    images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(images) < settings.min_images_per_job:
        print(f"Need at least {settings.min_images_per_job} images, found {len(images)}")
        return 2

    job_id = uuid4()
    JobRepository.create_job(job_id, "method_1")
    JobManager.mark_uploading(job_id)

    target = settings.uploads_root / str(job_id) / "images"
    target.mkdir(parents=True, exist_ok=True)
    for p in images:
        shutil.copy2(p, target / p.name)
        JobRepository.add_image(job_id, f"{job_id}/{p.name}")
    print(f"Job {job_id}: {len(images)} images registered")

    JobManager.start_job(job_id)              # UPLOADING -> PENDING
    LocalWorker.execute(job_id)               # PENDING -> PROCESSING -> COMPLETED/FAILED

    job = JobRepository.get_job(job_id)
    print("Status:", job["status"])
    if job["status"] == "failed":
        print("Reason:", job.get("error_message"))
        print("Stage :", job.get("error_stage"), "| code:", job.get("error_code"))
        return 1
    mesh = settings.outputs_root / str(job_id) / "final_mesh.ply"
    print("Mesh  :", mesh, "(exists)" if mesh.exists() else "(MISSING)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
