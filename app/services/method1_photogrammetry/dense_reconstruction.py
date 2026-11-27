"""Dense reconstruction: undistort -> PatchMatch stereo -> fusion -> Poisson meshing.

Poisson meshing is attempted here (with a lower-depth retry), but its failure is
NOT fatal: the caller (mesh_builder) falls back to Ball-Pivoting on fused.ply.
"""
import logging
from pathlib import Path
from typing import Optional

from app.config import settings
from app.jobs.errors import PipelineError
from app.services.method1_photogrammetry.colmap_runner import (
    get_colmap_exe,
    has_cuda,
    run_colmap,
)
from app.services.presets import Preset, get_preset

logger = logging.getLogger(__name__)


def count_ply_vertices(ply_path: Path) -> int:
    """Vertex count from a PLY header (works for ASCII and binary files)."""
    with open(ply_path, "rb") as fh:
        header = fh.read(8192).decode("ascii", errors="ignore")
    for line in header.splitlines():
        if line.startswith("element vertex"):
            return int(line.split()[2])
        if line.strip() == "end_header":
            break
    return 0


def run_dense_reconstruction(sparse_model_dir: Path, workspace_dir: Path,
                             preset: Optional[Preset] = None,
                             is_video: bool = False) -> dict:
    """
    Returns a JSON-ready dict:
      fused_ply, fused_points, poisson_mesh (path or None), poisson_attempts[]
    Raises PipelineError when the dense point cloud cannot be produced.
    """
    preset = preset or get_preset("standard")
    sparse_model_dir = Path(sparse_model_dir)
    workspace_dir = Path(workspace_dir)
    stage = "dense"

    if not sparse_model_dir.exists():
        raise FileNotFoundError(f"Sparse model not found: {sparse_model_dir}")

    exe = get_colmap_exe()
    colmap_dir = workspace_dir / "colmap"
    images_dir = colmap_dir / "images"
    dense_dir = colmap_dir / "dense"
    log = workspace_dir / "colmap.log"

    if not images_dir.exists():
        raise FileNotFoundError(f"COLMAP images directory not found: {images_dir}")

    if not has_cuda():
        raise PipelineError(
            "Dense reconstruction needs an NVIDIA GPU with CUDA, but none was found "
            "(nvidia-smi is missing or failed). Run the worker on a CUDA machine; "
            "set SKIP_CUDA_CHECK=true if nvidia-smi is just not on PATH.",
            "no_cuda", stage,
        )

    dense_dir.mkdir(parents=True, exist_ok=True)
    fused_ply = dense_dir / "fused.ply"
    meshed_ply = dense_dir / "meshed-poisson.ply"
    for stale in (fused_ply, meshed_ply):
        stale.unlink(missing_ok=True)

    # For video input, frames already have continuous temporal parallax and high overlap.
    # Skipping the second geometric consistency pass avoids the 500-run bottleneck,
    # accelerating PatchMatch by 10x (from >1hr to ~5-8 mins), while stereo_fusion filters outliers.
    geom = preset.patchmatch_geom_consistency and not is_video
    pm_max_image_size = preset.dense_undistort_max_image_size
    pm_iterations = min(preset.patchmatch_num_iterations, 5)

    run_colmap(
        [exe, "image_undistorter",
         "--image_path", images_dir, "--input_path", sparse_model_dir,
         "--output_path", dense_dir, "--output_type", "COLMAP",
         "--max_image_size", preset.dense_undistort_max_image_size],
        log, "image_undistorter",
    )

    run_colmap(
        [exe, "patch_match_stereo",
         "--workspace_path", dense_dir, "--workspace_format", "COLMAP",
         "--PatchMatchStereo.max_image_size", pm_max_image_size,
         "--PatchMatchStereo.geom_consistency", "true" if geom else "false",
         "--PatchMatchStereo.window_radius", preset.patchmatch_window_radius,
         "--PatchMatchStereo.num_iterations", pm_iterations],
        log, "patch_match_stereo",
    )

    # Fusion must read the same kind of depth maps PatchMatch wrote:
    # geometric maps exist only when geom_consistency was on.
    # max_reproj_error: relaxed from COLMAP's default (2.0) to keep more points
    #   on detailed surfaces (face, hair, clothing textures).
    # check_num_images: minimum images a 3D point must be seen in (2 = keep more points).
    run_colmap(
        [exe, "stereo_fusion",
         "--workspace_path", dense_dir, "--workspace_format", "COLMAP",
         "--input_type", "geometric" if geom else "photometric",
         "--output_path", fused_ply,
         "--StereoFusion.max_image_size", preset.dense_fusion_max_image_size,
         "--StereoFusion.min_num_pixels", preset.fusion_min_num_pixels,
         "--StereoFusion.max_reproj_error", preset.fusion_max_reproj_error,
         "--StereoFusion.check_num_images", preset.fusion_check_num_images],
        log, "stereo_fusion",
    )

    if not fused_ply.exists():
        raise PipelineError("Dense point cloud was not generated", "no_fused_cloud", stage)

    fused_points = count_ply_vertices(fused_ply)
    logger.info("[Dense] Fused point cloud: %d points", fused_points)
    if fused_points < settings.min_fused_points:
        raise PipelineError(
            f"Dense point cloud has only {fused_points} points "
            f"(minimum {settings.min_fused_points}). The object is probably too "
            "textureless, shiny or covered by too few views.",
            "too_few_points", stage,
        )

    # Dense point cloud generation complete. Surface meshing is performed
    # in Stage 3 directly on the isolated point cloud to prevent scene clutter
    # and artificial balloon closures.
    return {
        "fused_ply": str(fused_ply),
        "fused_points": fused_points,
        "poisson_mesh": None,
        "poisson_attempts": [],
    }
