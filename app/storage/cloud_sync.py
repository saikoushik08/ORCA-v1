"""Cloud storage synchronization for images and reconstruction outputs (F-41).

- StorageSync.ensure_local(job_id): Downloads missing images from 'orca-images' bucket
  and verifies SHA-256 (enables worker to run on an independent machine).
- StorageSync.publish_outputs(job_id, outputs_dir): Uploads final meshes (.glb, .ply, .obj, .stl),
  quality reports, and logs to the 'orca-models' bucket.
"""
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from app.config import settings
from app.jobs.job_repository import JobRepository
from app.storage.supabase_client import supabase

logger = logging.getLogger(__name__)


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class StorageSync:
    @staticmethod
    def ensure_local(job_id: UUID) -> Path:
        """Ensure all job images exist locally, downloading from Supabase storage if missing."""
        local_dir = settings.uploads_root / str(job_id) / "images"
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            db_images = JobRepository.list_images(job_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[StorageSync] Could not query database image list for job %s: %s", job_id, exc)
            return local_dir

        bucket = settings.storage_bucket_images
        for img in db_images:
            fname = img.get("filename")
            if not fname:
                continue
            local_path = local_dir / fname
            expected_sha = img.get("sha256")

            # Check if file exists and has correct hash
            if local_path.exists():
                if expected_sha and _compute_sha256(local_path) == expected_sha:
                    continue  # Valid local copy exists

            # Download missing or corrupted image from cloud bucket
            cloud_path = f"{job_id}/{fname}"
            try:
                data = supabase.storage.from_(bucket).download(cloud_path)
                if data:
                    local_path.write_bytes(data)
                    logger.info("[StorageSync] Downloaded missing image %s from bucket %s", fname, bucket)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[StorageSync] Could not download %s from cloud storage: %s", cloud_path, exc)

        return local_dir

    @staticmethod
    def publish_outputs(job_id: UUID, outputs_dir: Path) -> Dict[str, str]:
        """Upload completed meshes, reports, and logs to the models bucket."""
        outputs_dir = Path(outputs_dir)
        if not outputs_dir.exists():
            return {}

        bucket = getattr(settings, "storage_bucket_models", "orca-models")
        uploaded: Dict[str, str] = {}

        file_types = {
            "final_mesh.glb": "model/gltf-binary",
            "final_mesh.ply": "application/octet-stream",
            "final_mesh.obj": "text/plain",
            "final_mesh.stl": "model/stl",
            "fused_isolated.ply": "application/octet-stream",
            "fused.ply": "application/octet-stream",
            "quality.json": "application/json",
            "mesh_report.json": "application/json",
            "preflight_report.json": "application/json",
            "job.log": "text/plain",
            "colmap.log": "text/plain",
        }

        for filename, mime in file_types.items():
            local_file = outputs_dir / filename
            if not local_file.exists():
                continue

            # Skip large point clouds over 50MB to prevent Supabase 413 Payload Too Large
            size_bytes = local_file.stat().st_size
            if size_bytes > 50 * 1024 * 1024 and filename in ("fused.ply", "fused_isolated.ply"):
                logger.info("[StorageSync] Skipping cloud upload for large point cloud %s (%.1f MB > 50MB limit)",
                            filename, size_bytes / (1024 * 1024))
                continue

            cloud_path = f"{job_id}/{filename}"
            try:
                with open(local_file, "rb") as f:
                    file_data = f.read()

                supabase.storage.from_(bucket).upload(
                    cloud_path,
                    file_data,
                    file_options={"content-type": mime, "upsert": "true"},
                )
                uploaded[filename] = cloud_path
                logger.info("[StorageSync] Uploaded %s to bucket %s (%s)", filename, bucket, cloud_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[StorageSync] Failed to upload %s to cloud storage: %s", filename, exc)

        return uploaded
