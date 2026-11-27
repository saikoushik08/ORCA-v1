# app/workers/worker_executor.py
import inspect
import json
import logging
import shutil
import threading
from pathlib import Path
from uuid import UUID

from app.config import settings
from app.jobs.errors import PipelineError
from app.jobs.job_manager import JobManager
from app.jobs.job_repository import JobRepository
from app.logging_config import job_logging
from app.services.method1_photogrammetry.colmap_runner import IMAGE_EXTENSIONS
from app.services.method1_photogrammetry.pipeline import run_photogrammetry_pipeline
from app.services.power_utils import keep_awake
from app.services.preflight import run_preflight_check
from app.storage.cloud_sync import StorageSync

logger = logging.getLogger(__name__)

# One reconstruction at a time per process: two COLMAP runs on one GPU run out of memory.
_RUN_LOCK = threading.Lock()


class WorkerExecutor:
    """
    Runs one job that is ALREADY in PROCESSING (claimed through
    JobManager.claim_for_processing). Ends with the job COMPLETED or FAILED,
    and always stores the reason when it fails.
    """

    @staticmethod
    def run_job(job_id: UUID, method: str):
        job_workspace = settings.workspace_root / str(job_id)
        with _RUN_LOCK:
            with job_logging(job_id, job_workspace / "job.log"):
                with keep_awake(display=True):
                    WorkerExecutor._run_job(job_id, method)

    # ------------------------------------------------------------------
    @staticmethod
    def _fail(job_id: UUID, reason: str, stage: str | None, code: str):
        logger.error("[Worker] Job %s failed (%s/%s): %s", job_id, stage, code, reason.splitlines()[0])
        try:
            JobManager.mark_failed(job_id, reason=reason, stage=stage, code=code)
        except Exception as mark_error:  # noqa: BLE001
            logger.error("[Worker] Could not mark job %s FAILED: %s", job_id, mark_error)

    @staticmethod
    def _prepare_images(job_id: UUID, job_workspace: Path) -> int:
        # Ensure local images exist, syncing missing images from cloud if needed (F-41)
        uploads = StorageSync.ensure_local(job_id)
        if not uploads.exists():
            raise FileNotFoundError(f"Uploaded images directory not found: {uploads}")
        files = [p for p in uploads.iterdir()
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if not files:
            raise FileNotFoundError(f"No uploaded images found in: {uploads}")

        target = job_workspace / "colmap" / "images"
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for p in files:
            shutil.copy2(p, target / p.name)
        logger.info("[Worker] %d images copied to workspace", len(files))
        return len(files)

    @staticmethod
    def _publish_outputs(job_id: UUID, job_workspace: Path, outputs: dict) -> Path:
        out = settings.outputs_root / str(job_id)
        out.mkdir(parents=True, exist_ok=True)

        for key in ("final_mesh", "fused_ply", "dense_mesh", "fused_isolated_ply"):
            src = outputs.get(key)
            if src and Path(src).exists():
                shutil.copy2(src, out / Path(src).name)

        # Copy all mesh export formats (.glb, .obj, .stl) from mesh directory (F-40)
        final_mesh_path = outputs.get("final_mesh")
        if final_mesh_path:
            mesh_parent = Path(final_mesh_path).parent
            for ext in (".glb", ".obj", ".stl"):
                extra_mesh = mesh_parent / f"final_mesh{ext}"
                if extra_mesh.exists():
                    shutil.copy2(extra_mesh, out / extra_mesh.name)

        for name in ("mesh_report.json", "report.json", "preflight_report.json", "job.log", "colmap.log"):
            src = job_workspace / name
            if src.exists():
                shutil.copy2(src, out / name)

        sparse_model = outputs.get("sparse_model")
        if sparse_model and Path(sparse_model).exists():
            dest = out / "sparse_model"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(sparse_model, dest)

        (out / "quality.json").write_text(
            json.dumps(outputs["quality"], indent=2), encoding="utf-8")

        # Synchronize outputs to cloud storage (F-41)
        try:
            cloud_files = StorageSync.publish_outputs(job_id, out)
            if cloud_files:
                outputs["cloud_files"] = cloud_files
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Worker] Cloud output sync error: %s", exc)

        return out

    @staticmethod
    def _cleanup_intermediates(job_workspace: Path):
        """Depth/normal maps and undistorted images can be several GB per job."""
        if settings.keep_intermediates:
            return
        for sub in ("colmap/dense/stereo", "colmap/dense/images"):
            shutil.rmtree(job_workspace / sub, ignore_errors=True)

    @staticmethod
    def _summary(outputs: dict) -> dict:
        """Compact, JSON-ready result stored on the job row."""
        sparse = dict(outputs["sparse"])
        sparse.pop("model_dir", None)
        mesh_info = {k: outputs["mesh"][k] for k in
                     ("method_used", "fallback_used", "faces", "vertices",
                      "has_vertex_colors", "steps") if k in outputs["mesh"]}
        if "exported_formats" in outputs["mesh"]:
            mesh_info["exported_formats"] = outputs["mesh"]["exported_formats"]

        result = {
            "sparse": sparse,
            "dense": {"fused_points": outputs["dense"]["fused_points"],
                      "poisson_attempts": outputs["dense"]["poisson_attempts"]},
            "mesh": mesh_info,
            "quality": outputs["quality"],
            "timings": outputs["timings"],
            "files": ["final_mesh.glb", "final_mesh.ply", "final_mesh.obj", "final_mesh.stl",
                      "fused.ply", "quality.json", "mesh_report.json"],
        }
        if "cloud_files" in outputs:
            result["cloud_files"] = outputs["cloud_files"]
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _run_job(job_id: UUID, method: str):
        logger.info("[Worker] Started job %s (method=%s)", job_id, method)

        if method != "method_1":
            WorkerExecutor._fail(job_id, f"Unsupported method '{method}'", "start",
                                 "unsupported_method")
            return

        job_data = JobRepository.get_job(job_id) or {}
        preset = job_data.get("preset", "standard")
        is_video = (
            job_data.get("input_type") == "video"
            or (settings.uploads_root / str(job_id) / "video_report.json").exists()
            or (settings.uploads_root / str(job_id) / "video").exists()
            or any("frame_" in p.name for p in (settings.uploads_root / str(job_id) / "images").glob("*.jpg"))
        )

        job_workspace = settings.workspace_root / str(job_id)
        current = {"stage": "start"}

        def on_stage(name: str, progress: int):
            current["stage"] = name
            JobRepository.update_fields(job_id, current_stage=name, progress=progress)

        try:
            job_workspace.mkdir(parents=True, exist_ok=True)
            WorkerExecutor._prepare_images(job_id, job_workspace)

            # Stage 0: Pre-flight check (CPU)
            on_stage("preflight", 2)
            preflight = run_preflight_check(job_workspace / "colmap" / "images", is_video=is_video)
            (job_workspace / "preflight_report.json").write_text(
                json.dumps(preflight.to_dict(), indent=2), encoding="utf-8"
            )
            if preflight.verdict == "REJECT":
                WorkerExecutor._fail(job_id, preflight.message, "preflight", "preflight_rejected")
                return

            sig = inspect.signature(run_photogrammetry_pipeline)
            pipe_kwargs = {"on_stage": on_stage}
            if "preset" in sig.parameters:
                pipe_kwargs["preset"] = preset
            if "is_video" in sig.parameters:
                pipe_kwargs["is_video"] = is_video
            outputs = run_photogrammetry_pipeline(job_workspace, **pipe_kwargs)
            WorkerExecutor._publish_outputs(job_id, job_workspace, outputs)
            WorkerExecutor._cleanup_intermediates(job_workspace)
        except PipelineError as exc:
            WorkerExecutor._fail(job_id, str(exc), exc.stage or current["stage"], exc.code)
            return
        except FileNotFoundError as exc:
            WorkerExecutor._fail(job_id, str(exc), current["stage"], "missing_file")
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Worker] Unexpected error in job %s", job_id)
            WorkerExecutor._fail(job_id, f"{type(exc).__name__}: {exc}", current["stage"],
                                 "internal_error")
            return

        # The mesh exists from here on. A database hiccup must not turn a finished
        # job into a failed one, so completion is retried and never marks FAILED.
        quality = outputs["quality"]
        if quality["verdict"] == "FAIL":
            WorkerExecutor._fail(
                job_id, "Mesh quality check failed: " + "; ".join(quality["errors"]),
                "quality", "quality_failed")
            return

        summary = WorkerExecutor._summary(outputs)
        for attempt in range(3):
            try:
                JobManager.mark_completed(job_id, result=summary)
                logger.info("[Worker] Job %s COMPLETED (verdict=%s)", job_id, quality["verdict"])
                return
            except Exception as exc:  # noqa: BLE001
                logger.error("[Worker] mark_completed attempt %d failed: %s", attempt + 1, exc)
        logger.error("[Worker] Job %s finished but could not be marked COMPLETED; "
                     "outputs are in %s", job_id, settings.outputs_root / str(job_id))
