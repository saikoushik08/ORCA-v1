"""Quality presets (draft / standard / high) replacing hardcoded debug settings (F-31).

A single Preset dataclass feeds every COLMAP, mesh, and video call:
- Draft: fast feedback, smaller image size, lower depth, fewer iterations.
- Standard: balanced quality and speed, geometric consistency enabled.
- High: maximum fidelity, high image resolution, deep Poisson reconstruction.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Preset:
    name: str
    # SfM settings
    sfm_max_image_size: int
    max_sift_features: int
    vocab_tree_threshold: int
    # Video settings
    video_frame_budget: int
    # Dense reconstruction settings
    dense_undistort_max_image_size: int
    dense_fusion_max_image_size: int
    patchmatch_window_radius: int
    patchmatch_num_iterations: int
    patchmatch_geom_consistency: bool
    fusion_min_num_pixels: int
    fusion_max_reproj_error: float       # relax to keep more fused points (COLMAP default 2.0)
    fusion_check_num_images: int         # minimum images a 3D point must be seen in
    poisson_depth: int
    poisson_fallback_depth: int
    # Mesh post-processing settings
    mesh_target_faces: int
    taubin_iterations: int


PRESETS: Dict[str, Preset] = {
    "draft": Preset(
        name="draft",
        sfm_max_image_size=1600,
        max_sift_features=4096,
        vocab_tree_threshold=80,
        video_frame_budget=80,
        dense_undistort_max_image_size=1000,
        dense_fusion_max_image_size=1000,
        patchmatch_window_radius=4,
        patchmatch_num_iterations=3,
        patchmatch_geom_consistency=False,
        fusion_min_num_pixels=2,
        fusion_max_reproj_error=2.0,
        fusion_check_num_images=2,
        poisson_depth=9,
        poisson_fallback_depth=7,
        mesh_target_faces=80_000,
        taubin_iterations=2,
    ),
    "standard": Preset(
        name="standard",
        sfm_max_image_size=2400,
        max_sift_features=8192,
        vocab_tree_threshold=80,
        video_frame_budget=200,
        dense_undistort_max_image_size=1600,
        dense_fusion_max_image_size=1600,
        patchmatch_window_radius=5,
        patchmatch_num_iterations=5,
        patchmatch_geom_consistency=True,
        fusion_min_num_pixels=2,
        fusion_max_reproj_error=2.5,
        fusion_check_num_images=2,
        poisson_depth=11,
        poisson_fallback_depth=8,
        mesh_target_faces=400_000,
        taubin_iterations=4,
    ),
    "high": Preset(
        name="high",
        sfm_max_image_size=3200,
        max_sift_features=8192,
        vocab_tree_threshold=80,
        video_frame_budget=350,
        dense_undistort_max_image_size=2400,
        dense_fusion_max_image_size=2400,
        patchmatch_window_radius=5,
        patchmatch_num_iterations=5,
        patchmatch_geom_consistency=True,
        fusion_min_num_pixels=2,
        fusion_max_reproj_error=3.0,
        fusion_check_num_images=2,
        poisson_depth=11,
        poisson_fallback_depth=8,
        mesh_target_faces=800_000,
        taubin_iterations=4,
    ),
}


def get_preset(name: str | None = None) -> Preset:
    """Resolve preset by name ('draft', 'standard', 'high'). Defaults to 'standard'."""
    key = (name or "standard").strip().lower()
    return PRESETS.get(key, PRESETS["standard"])
