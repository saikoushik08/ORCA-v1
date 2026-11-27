import json
import logging
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile

from app.config import settings
from app.jobs.errors import InvalidTransition, JobNotFound
from app.jobs.job_manager import JobManager
from app.jobs.job_repository import JobRepository
from app.services.image_utils import CheckedImage, ImageRejected, check_image
from app.services.video_utils import VideoRejected, extract_and_select_keyframes
from app.storage.supabase_client import supabase

# No prefix here: main.py mounts this router under /upload.
router = APIRouter(tags=["Uploads"])

logger = logging.getLogger(__name__)

_MIME = {".jpg": "image/jpeg", ".png": "image/png", ".tif": "image/tiff"}


def _display_name(raw_name: str | None) -> str:
    """Client file name reduced to a harmless label (never used as a path)."""
    name = PureWindowsPath(PurePosixPath(raw_name or "image").name).name
    return (name or "image")[:120]


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.iterdir() if p.is_file()) if path.exists() else 0


def _sync_to_cloud(job_id: UUID, items: list) -> None:
    """Best-effort copy to Supabase storage with retries. The local file and the
    database row are the source of truth, so a cloud outage never loses an image."""
    for storage_path, local_path, extension in items:
        for attempt in range(3):
            try:
                with open(local_path, "rb") as fh:
                    supabase.storage.from_(settings.storage_bucket_images).upload(
                        storage_path, fh.read(),
                        {"content-type": _MIME.get(extension, "application/octet-stream"),
                         "upsert": "true"},
                    )
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    logger.warning("Cloud sync failed for %s: %s", storage_path, exc)
                else:
                    time.sleep(2 ** attempt)


@router.post("/image/{job_id}")
def upload_images(
    job_id: UUID,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    # A normal `def` endpoint runs in FastAPI's thread pool, so the blocking file
    # and database I/O below does not block the event loop.

    # 1. Job must exist and still accept uploads.
    job = JobRepository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not JobManager.can_upload_images(job_id):
        raise HTTPException(status_code=400, detail=f"Uploads are locked. Job is '{job['status']}'.")
    if job.get("input_type") == "video":
        raise HTTPException(
            status_code=400,
            detail="Job was created with video input. Mixed photos and video in one job are not supported.",
        )

    # 2. Validate EVERYTHING before writing anything (all-or-nothing).
    max_bytes = settings.max_file_mb * 1024 * 1024
    existing = JobRepository.list_images(job_id)
    seen_hashes = {r["sha256"] for r in existing if r.get("sha256")}

    accepted: list[tuple[str, CheckedImage]] = []
    skipped_duplicates: list[str] = []
    rejected: list[str] = []

    for f in files:
        name = _display_name(f.filename)
        raw = f.file.read(max_bytes + 1)
        try:
            checked = check_image(raw, name, settings.allowed_formats,
                                  settings.min_image_short_side, max_bytes)
        except ImageRejected as exc:
            rejected.append(str(exc))
            continue
        if checked.sha256 in seen_hashes:
            skipped_duplicates.append(name)
            continue
        seen_hashes.add(checked.sha256)
        accepted.append((name, checked))

    if rejected:
        raise HTTPException(
            status_code=400,
            detail={"message": "Some files were rejected; nothing was saved.",
                    "rejected": rejected},
        )

    local_images_dir = settings.uploads_root / str(job_id) / "images"
    existing_count = JobRepository.count_images(job_id)
    if existing_count + len(accepted) > settings.max_images_per_job:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image limit exceeded. Current: {existing_count}, "
                f"Trying to add: {len(accepted)}, Max allowed: {settings.max_images_per_job}"
            ),
        )
    new_bytes = sum(c.size_bytes for _, c in accepted)
    if _dir_size(local_images_dir) + new_bytes > settings.max_job_gb * 1024 ** 3:
        raise HTTPException(
            status_code=400,
            detail=f"Total upload size would exceed {settings.max_job_gb:g} GB for this job.",
        )

    if not accepted:
        return {
            "job_id": str(job_id), "uploaded_count": 0,
            "skipped_duplicates": skipped_duplicates,
            "total_images": existing_count, "status": job["status"],
            "uploaded_images": [], "message": "No new images (all were duplicates)",
        }

    # 3. Limits passed: now the job may move CREATED -> UPLOADING.
    try:
        JobManager.mark_uploading(job_id)
    except JobNotFound:
        raise HTTPException(status_code=404, detail="Job not found")
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # 4. Write files under server-chosen names, then record them.
    local_images_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    uploaded = []
    cloud_items = []
    try:
        for name, checked in accepted:
            stored_name = f"{uuid4().hex}{checked.extension}"
            local_path = local_images_dir / stored_name
            local_path.write_bytes(checked.data)
            written.append(local_path)

            storage_path = f"{job_id}/{stored_name}"
            JobRepository.add_image(
                job_id, storage_path,
                original_name=name, sha256=checked.sha256, width=checked.width,
                height=checked.height, size_bytes=checked.size_bytes,
            )
            written.pop()  # recorded in the database: keep it
            cloud_items.append((storage_path, local_path, checked.extension))
            uploaded.append({
                "filename": name, "stored_as": stored_name,
                "width": checked.width, "height": checked.height,
                "orientation_fixed": checked.orientation_fixed,
            })
    except Exception as exc:
        for leftover in written:  # file saved but never recorded
            leftover.unlink(missing_ok=True)
        logger.exception("Saving uploaded images failed")
        raise HTTPException(status_code=500, detail=f"Failed to save images: {exc}")

    background_tasks.add_task(_sync_to_cloud, job_id, cloud_items)

    updated_job = JobRepository.get_job(job_id)
    return {
        "job_id": str(job_id),
        "uploaded_count": len(uploaded),
        "skipped_duplicates": skipped_duplicates,
        "total_images": JobRepository.count_images(job_id),
        "status": updated_job["status"] if updated_job else job["status"],
        "uploaded_images": uploaded,
        "message": "Images uploaded successfully",
    }


@router.post("/video/{job_id}")
def upload_video(
    job_id: UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_frames: Optional[int] = Query(None, description="Optional override for frame budget"),
):
    """Upload a video file, validate limits, extract and select keyframes (F-47)."""
    # 1. Job must exist and still accept uploads.
    job = JobRepository.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not JobManager.can_upload_images(job_id):
        raise HTTPException(status_code=400, detail=f"Uploads are locked. Job is '{job['status']}'.")

    # Reject mixed photos + video
    if JobRepository.count_images(job_id) > 0 and job.get("input_type") != "video":
        raise HTTPException(
            status_code=400,
            detail="Job already contains photo uploads. Mixed photos and video in one job are not supported.",
        )

    # 2. Check extension & size
    clean_name = _display_name(file.filename)
    ext = Path(clean_name).suffix.lower().lstrip(".")
    if ext not in settings.allowed_video_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Video format '{ext}' is not supported. Allowed: {', '.join(sorted(settings.allowed_video_exts))}",
        )

    max_video_bytes = settings.video_max_file_mb * 1024 * 1024
    content = file.file.read(max_video_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty.")
    if len(content) > max_video_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Video exceeds maximum allowed size of {settings.video_max_file_mb} MB.",
        )

    # 3. Save raw video temporarily
    video_dir = settings.uploads_root / str(job_id) / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    temp_video_path = video_dir / f"{uuid4().hex}.{ext}"
    temp_video_path.write_bytes(content)

    # 4. Determine budget from job preset or query override
    preset = job.get("preset", "standard")
    if max_frames and max_frames > 0:
        budget = min(max_frames, settings.max_images_per_job)
    else:
        budget = settings.video_frame_budget_standard
        if preset == "draft":
            budget = settings.video_frame_budget_draft
        elif preset == "high":
            budget = settings.video_frame_budget_high

    try:
        keyframes, report = extract_and_select_keyframes(
            temp_video_path,
            frame_budget=budget,
            candidate_fps=settings.video_candidate_fps,
            min_parallax_ratio=settings.video_min_parallax_ratio,
            min_blur_ratio=settings.video_min_blur_ratio,
        )
    except VideoRejected as exc:
        temp_video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        temp_video_path.unlink(missing_ok=True)
        logger.exception("Failed to process video")
        raise HTTPException(status_code=500, detail=f"Failed to process video: {exc}")

    if len(keyframes) < settings.min_images_per_job:
        temp_video_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only {len(keyframes)} usable frames could be extracted from video (minimum {settings.min_images_per_job}). "
                "Move more slowly, maintain steady lighting, and orbit 360 degrees around the object."
            ),
        )

    # 5. Move job to UPLOADING state
    try:
        JobManager.mark_uploading(job_id)
    except JobNotFound:
        raise HTTPException(status_code=404, detail="Job not found")
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # 6. Save extracted keyframes
    local_images_dir = settings.uploads_root / str(job_id) / "images"
    local_images_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    cloud_items = []
    saved_frames = []

    try:
        for kf in keyframes:
            stored_name = f"{uuid4().hex}.jpg"
            local_path = local_images_dir / stored_name
            local_path.write_bytes(kf.image_bytes)
            written.append(local_path)

            storage_path = f"{job_id}/{stored_name}"
            JobRepository.add_image(
                job_id,
                storage_path,
                original_name=f"frame_{kf.frame_index:05d}.jpg",
                sha256=kf.sha256,
                width=kf.width,
                height=kf.height,
                size_bytes=kf.size_bytes,
                source_type="video_frame",
                source_video=clean_name,
                frame_index=kf.frame_index,
                frame_time_s=kf.timestamp_s,
            )
            written.pop()
            cloud_items.append((storage_path, local_path, ".jpg"))
            saved_frames.append({
                "frame_index": kf.frame_index,
                "timestamp_s": kf.timestamp_s,
                "stored_as": stored_name,
            })
    except Exception as exc:
        for leftover in written:
            leftover.unlink(missing_ok=True)
        logger.exception("Saving extracted video frames failed")
        raise HTTPException(status_code=500, detail=f"Failed to save video frames: {exc}")

    # 7. Update job input_type and persist video report
    JobRepository.update_fields(job_id, input_type="video")
    report_file = settings.uploads_root / str(job_id) / "video_report.json"
    report_data = {
        "source_video": clean_name,
        "input_type": "video",
        "preset": preset,
        "budget": budget,
        **report,
    }
    report_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    background_tasks.add_task(_sync_to_cloud, job_id, cloud_items)

    updated_job = JobRepository.get_job(job_id)
    return {
        "job_id": str(job_id),
        "source_video": clean_name,
        "input_type": "video",
        "duration_s": report["duration_s"],
        "fps": report["fps"],
        "resolution": report["resolution"],
        "frames_extracted": report["frames_extracted"],
        "frames_kept": report["frames_kept"],
        "reasons_dropped": report["reasons_dropped"],
        "total_images": JobRepository.count_images(job_id),
        "status": updated_job["status"] if updated_job else job["status"],
        "message": f"Successfully extracted and selected {report['frames_kept']} keyframes from video",
    }
