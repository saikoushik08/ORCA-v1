import json
import logging
import shutil
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.config import settings
from app.jobs.errors import InvalidTransition, JobNotFound, TransitionConflict
from app.jobs.job_manager import JobManager
from app.jobs.job_repository import JobRepository
from app.jobs.job_states import JobStatus
from app.logging_config import tail_file
from app.services.preflight import run_preflight_check
from app.services.presets import get_preset

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["Jobs"])

# Files a client may download. A fixed whitelist: the name never becomes a free-form path.
DOWNLOADABLE_FILES = {
    "final_mesh.glb": "model/gltf-binary",
    "final_mesh.ply": "application/octet-stream",
    "final_mesh.obj": "text/plain",
    "final_mesh.stl": "model/stl",
    "fused_isolated.ply": "application/octet-stream",
    "fused.ply": "application/octet-stream",
    "meshed-poisson.ply": "application/octet-stream",
    "quality.json": "application/json",
    "mesh_report.json": "application/json",
    "report.json": "application/json",
    "preflight_report.json": "application/json",
    "job.log": "text/plain",
}


class ReconstructionMethod(str, Enum):
    """Allowed reconstruction methods. method_2 (AI/NeRF) is not implemented."""
    method_1 = "method_1"
    method_2 = "method_2"


def _get_or_404(job_id: UUID) -> dict:
    job = JobRepository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _outputs_dir(job_id: UUID):
    return settings.outputs_root / str(job_id)


# -----------------------------
# CREATE NEW JOB
# -----------------------------
@router.post("/")
@router.post("", include_in_schema=False)
def create_job(
    method: ReconstructionMethod = Query(
        default=ReconstructionMethod.method_1,
        description="Reconstruction method (only method_1 is implemented)",
    ),
    preset: str = Query(
        default="standard",
        description="Quality preset: draft | standard | high",
    ),
):
    if method != ReconstructionMethod.method_1:
        raise HTTPException(status_code=501, detail=f"Method '{method.value}' is not implemented")

    job_id = uuid4()
    JobRepository.create_job(job_id, method.value, preset=preset)
    logger.info("Created job %s (method=%s, preset=%s)", job_id, method.value, preset)
    return {
        "job_id": str(job_id),
        "method": method.value,
        "preset": preset,
        "status": JobStatus.CREATED.value,
        "message": "Job created successfully",
    }


# -----------------------------
# GET JOB STATUS
# -----------------------------
@router.get("/{job_id}")
def get_job_status(job_id: UUID):
    job = _get_or_404(job_id)
    result = job.get("result") or {}
    quality = result.get("quality") or {}
    return {
        "job_id": job["job_id"],
        "method": job["method"],
        "preset": job.get("preset", "standard"),
        "input_type": job.get("input_type", "photos"),
        "status": job["status"],
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "stage": job.get("current_stage"),
        "progress": job.get("progress"),
        "images": JobRepository.count_images(job_id),
        "error": ({
            "message": job.get("error_message"),
            "code": job.get("error_code"),
            "stage": job.get("error_stage"),
        } if job["status"] == JobStatus.FAILED.value else None),
        "quality": ({
            "verdict": quality.get("verdict"),
            "faces": quality.get("faces"),
            "watertight": quality.get("watertight"),
            "warnings": quality.get("warnings"),
            "registered_ratio": (result.get("sparse") or {}).get("registered_ratio"),
            "fallback_used": (result.get("mesh") or {}).get("fallback_used"),
        } if quality else None),
    }


# -----------------------------
# VIDEO FRAMES REPORT
# -----------------------------
@router.get("/{job_id}/frames")
def get_job_frames(job_id: UUID):
    _get_or_404(job_id)
    vfile = settings.uploads_root / str(job_id) / "video_report.json"
    if not vfile.exists():
        raise HTTPException(status_code=404, detail="No video frames report found for this job")
    return json.loads(vfile.read_text(encoding="utf-8"))


# -----------------------------
# LOGS / EVENTS / QUALITY
# -----------------------------
@router.get("/{job_id}/logs")
def get_job_logs(job_id: UUID, tail: int = Query(default=200, ge=1, le=2000)):
    _get_or_404(job_id)
    log_file = settings.workspace_root / str(job_id) / "job.log"
    return {"job_id": str(job_id), "lines": tail_file(log_file, tail)}


@router.get("/{job_id}/events")
def get_job_events(job_id: UUID):
    _get_or_404(job_id)
    return {"job_id": str(job_id), "events": JobRepository.list_events(job_id)}


@router.get("/{job_id}/quality")
def get_job_quality(job_id: UUID):
    job = _get_or_404(job_id)
    qfile = _outputs_dir(job_id) / "quality.json"
    if qfile.exists():
        return json.loads(qfile.read_text(encoding="utf-8"))
    quality = (job.get("result") or {}).get("quality")
    if quality:
        return quality
    raise HTTPException(status_code=404, detail="No quality report yet")


# -----------------------------
# PRE-FLIGHT CHECK (Stage 0 CPU validation)
# -----------------------------
@router.post("/{job_id}/preflight")
def run_job_preflight(job_id: UUID):
    job = _get_or_404(job_id)
    images_dir = settings.uploads_root / str(job_id) / "images"
    if not images_dir.exists() or JobRepository.count_images(job_id) == 0:
        raise HTTPException(status_code=400, detail="No images have been uploaded for this job yet")

    report = run_preflight_check(images_dir, is_video=(job.get("input_type") == "video"))
    report_file = settings.uploads_root / str(job_id) / "preflight_report.json"
    report_file.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report.to_dict()


@router.get("/{job_id}/preflight")
def get_job_preflight(job_id: UUID):
    _get_or_404(job_id)
    report_file = settings.uploads_root / str(job_id) / "preflight_report.json"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="No pre-flight report found for this job")
    return json.loads(report_file.read_text(encoding="utf-8"))


# -----------------------------
# RESULT FILES
# -----------------------------
@router.get("/{job_id}/result")
def get_job_result(job_id: UUID):
    job = _get_or_404(job_id)
    if job["status"] != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=409,
            detail=f"Job is '{job['status']}'; results exist only for completed jobs.",
        )
    out = _outputs_dir(job_id)
    files = [n for n in DOWNLOADABLE_FILES if (out / n).exists()]

    # Format model URLs (glb prioritised for web viewers, fallback to ply/obj/stl)
    model_urls = {
        ext: f"/jobs/{job_id}/result/download?file=final_mesh.{ext}"
        for ext in ("glb", "ply", "obj", "stl")
        if f"final_mesh.{ext}" in files
    }
    primary_mesh_url = model_urls.get("glb") or model_urls.get("ply")

    return {
        "job_id": str(job_id),
        "status": job["status"],
        "files": [{"name": n, "url": f"/jobs/{job_id}/result/download?file={n}"} for n in files],
        "mesh_url": primary_mesh_url,
        "model_urls": model_urls,
    }


@router.get("/{job_id}/result/download")
def download_result_file(
    job_id: UUID,
    file: Optional[str] = Query(default=None),
    format: Optional[str] = Query(default=None),
):
    _get_or_404(job_id)
    target_file = file
    if format:
        fmt = format.lower().strip(".")
        target_file = f"final_mesh.{fmt}"
    if not target_file:
        target_file = "final_mesh.ply"

    if target_file not in DOWNLOADABLE_FILES:
        raise HTTPException(status_code=400, detail=f"Unknown or unsupported file: {target_file}")
    path = _outputs_dir(job_id) / target_file
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{target_file} not available")
    return FileResponse(path, media_type=DOWNLOADABLE_FILES[target_file], filename=target_file)


# -----------------------------
# RE-RUN JOB (linked job re-using images and cached features, F-44 & F-35)
# -----------------------------
@router.post("/{job_id}/rerun", status_code=201)
def rerun_job(
    job_id: UUID,
    preset: Optional[str] = Query(default=None),
):
    parent_job = _get_or_404(job_id)
    new_preset = preset or parent_job.get("preset", "standard")
    get_preset(new_preset)  # Validates preset name

    # Create new child job linked to parent
    new_job_id = uuid4()
    JobRepository.create_job(
        job_id=new_job_id,
        method=parent_job.get("method", "method_1"),
        preset=new_preset,
        input_type=parent_job.get("input_type", "photos"),
        parent_job_id=job_id,
    )

    # Copy uploaded images from parent to child
    parent_uploads = settings.uploads_root / str(job_id)
    child_uploads = settings.uploads_root / str(new_job_id)
    if parent_uploads.exists():
        shutil.copytree(parent_uploads, child_uploads, dirs_exist_ok=True)

    # Clone database.db and features_manifest.json to child workspace for feature reuse (F-35, F-44)
    parent_colmap = settings.workspace_root / str(job_id) / "colmap"
    child_colmap = settings.workspace_root / str(new_job_id) / "colmap"
    if (parent_colmap / "database.db").exists():
        child_colmap.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent_colmap / "database.db", child_colmap / "database.db")
        if (parent_colmap / "features_manifest.json").exists():
            shutil.copy2(parent_colmap / "features_manifest.json", child_colmap / "features_manifest.json")

    # Clone images database rows
    parent_images = JobRepository.list_images(job_id)
    for img in parent_images:
        old_path = img.get("image_path", "")
        filename = old_path.split("/")[-1] if "/" in old_path else old_path
        new_path = f"{new_job_id}/{filename}"
        JobRepository.add_image(
            job_id=new_job_id,
            image_path=new_path,
            original_name=img.get("original_name"),
            sha256=img.get("sha256"),
            width=img.get("width"),
            height=img.get("height"),
            size_bytes=img.get("size_bytes"),
            camera_key=img.get("camera_key"),
            source_type=img.get("source_type", "photo"),
            source_video=img.get("source_video"),
            frame_index=img.get("frame_index"),
            frame_time_s=img.get("frame_time_s"),
        )

    return {
        "job_id": str(new_job_id),
        "parent_job_id": str(job_id),
        "preset": new_preset,
        "status": JobStatus.CREATED.value,
        "message": "Re-run job created with parent images and cached feature database",
    }


# NOTE: the former PUT /jobs/{job_id}/status admin route stays removed: state
# changes happen only inside the backend (JobManager).


# -----------------------------
# START JOB (queue it; the worker runs it)
# -----------------------------
@router.post("/{job_id}/start", status_code=202)
def start_job(job_id: UUID):
    _get_or_404(job_id)

    image_count = JobRepository.count_images(job_id)
    if image_count < settings.min_images_per_job:
        raise HTTPException(
            status_code=400,
            detail=f"At least {settings.min_images_per_job} images required. Found {image_count}",
        )

    try:
        JobManager.start_job(job_id)          # UPLOADING -> PENDING, atomic
    except JobNotFound:
        raise HTTPException(status_code=404, detail="Job not found")
    except TransitionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except InvalidTransition as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Returns immediately; the worker process/thread picks the job up.
    return {
        "job_id": str(job_id),
        "status": JobStatus.PENDING.value,
        "images": image_count,
        "status_url": f"/jobs/{job_id}",
        "message": "Job queued. Poll the status URL for progress.",
    }
