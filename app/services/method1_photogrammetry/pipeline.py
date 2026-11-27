# app/services/method1_photogrammetry/pipeline.py
import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

from app.services.method1_photogrammetry.colmap_runner import IMAGE_EXTENSIONS, run_colmap_sfm
from app.services.method1_photogrammetry.dense_reconstruction import run_dense_reconstruction
from app.services.method1_photogrammetry.mesh_builder import build_mesh
from app.services.method1_photogrammetry.point_cloud_isolation import isolate_object_point_cloud
from app.services.method1_photogrammetry.quality_checker import check_mesh_quality
from app.services.presets import Preset, get_preset

logger = logging.getLogger(__name__)

StageCallback = Callable[[str, int], None]


def run_photogrammetry_pipeline(workspace_dir: Path,
                                on_stage: Optional[StageCallback] = None,
                                preset: Optional[str | Preset] = None,
                                is_video: bool = False) -> dict:
    """
    images -> sparse (SfM + gate) -> dense (+Poisson) -> mesh (+fallback, clean-up) -> quality

    Expects the job images in <workspace>/colmap/images (prepared by the worker).
    Returns a JSON-serialisable dict (all paths are strings). Raises PipelineError
    (with code + stage) when a stage fails.
    """
    p = preset if isinstance(preset, Preset) else get_preset(preset)
    workspace_dir = Path(workspace_dir)
    colmap_dir = workspace_dir / "colmap"
    images_dir = colmap_dir / "images"

    if not workspace_dir.exists():
        raise FileNotFoundError(f"Workspace directory not found: {workspace_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    image_files = [p for p in images_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_files:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    def stage(name: str, progress: int):
        logger.info("[Pipeline] === %s (%d%%) ===", name, progress)
        if on_stage:
            on_stage(name, progress)

    timings: dict = {}
    t_start = time.time()
    logger.info("[Pipeline] Starting with %d images in %s (preset=%s, is_video=%s)",
                len(image_files), workspace_dir, p.name, is_video)

    # ---- 1. Sparse reconstruction + registered-image gate ----
    stage("sfm", 5)
    t = time.time()
    sparse = run_colmap_sfm(images_dir=images_dir, workspace_dir=workspace_dir,
                            preset=p, is_video=is_video)
    timings["sfm_s"] = round(time.time() - t, 1)

    # ---- 2. Dense reconstruction ----
    stage("dense", 30)
    t = time.time()
    dense = run_dense_reconstruction(
        Path(sparse["model_dir"]), workspace_dir, preset=p, is_video=is_video
    )
    timings["dense_s"] = round(time.time() - t, 1)

    # ---- 2.5 Object isolation before meshing (F-22) ----
    stage("isolate", 70)
    t = time.time()
    fused_isolated = workspace_dir / "colmap" / "dense" / "fused_isolated.ply"
    isolation_report = isolate_object_point_cloud(
        Path(dense["fused_ply"]),
        fused_isolated,
        sparse_model_dir=Path(sparse["model_dir"]),
        mode="auto",
        fill_holes=True,
    )
    fused_for_mesh = fused_isolated if fused_isolated.exists() else Path(dense["fused_ply"])
    timings["isolate_s"] = round(time.time() - t, 1)

    # ---- 3. Mesh (Screened Poisson on isolated cloud with density trimming) and clean-up ----
    stage("mesh", 80)
    t = time.time()
    mesh_report = build_mesh(
        Path(dense["poisson_mesh"]) if dense["poisson_mesh"] else None,
        fused_for_mesh,
        workspace_dir,
        preset=p,
        surface_method="poisson",
    )
    timings["mesh_s"] = round(time.time() - t, 1)

    # ---- 4. Quality report ----
    stage("quality", 92)
    t = time.time()
    quality = check_mesh_quality(
        Path(mesh_report["final_mesh"]),
        fused_points=dense["fused_points"],
        sparse=sparse,
        mesh_report=mesh_report,
        fused_ply_path=fused_for_mesh,
    )
    timings["quality_s"] = round(time.time() - t, 1)
    timings["total_s"] = round(time.time() - t_start, 1)

    outputs = {
        "workspace_dir": str(workspace_dir),
        "images_dir": str(images_dir),
        "preset": p.name,
        "is_video": is_video,
        "sparse_model": sparse["model_dir"],
        "fused_ply": dense["fused_ply"],
        "fused_isolated_ply": str(fused_for_mesh),
        "dense_mesh": dense["poisson_mesh"],
        "final_mesh": mesh_report["final_mesh"],
        "sparse": sparse,
        "dense": dense,
        "isolation": isolation_report,
        "mesh": mesh_report,
        "quality": quality,
        "timings": timings,
    }
    (workspace_dir / "report.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    logger.info("[Pipeline] Finished in %.0f s, verdict=%s", timings["total_s"], quality["verdict"])
    return outputs
