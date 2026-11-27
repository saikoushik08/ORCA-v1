"""Single source of configuration for ORCA.

Every setting can be overridden with an environment variable or a line in the
.env file in the project root (see .env.example). Nothing else in the code base
should hard-code paths, limits or origins.
"""
import shutil
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the folder that contains app/. Independent of the launch directory.
BASE_DIR = Path(__file__).resolve().parents[1]

# Common COLMAP install locations on Windows. Used only if COLMAP_BIN is not set
# and colmap is not on PATH.
_WINDOWS_COLMAP_DEFAULTS = [
    Path(r"C:\Program Files\colmap-x64-windows-cuda\bin\colmap.exe"),
    Path(r"C:\Program Files\COLMAP\bin\colmap.exe"),
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # ---- COLMAP ----
    colmap_bin: Optional[Path] = None
    # Seconds a single COLMAP stage may run before it is killed.
    colmap_stage_timeout_s: int = 4 * 3600
    # Set true if nvidia-smi is not on PATH but CUDA works anyway.
    skip_cuda_check: bool = False

    # ---- Folders (absolute; default to <project root>/...) ----
    uploads_root: Path = BASE_DIR / "uploads"
    workspace_root: Path = BASE_DIR / "workspace"
    outputs_root: Path = BASE_DIR / "outputs"

    # ---- Image rules (single definition of the limits) ----
    min_images_per_job: int = 5
    max_images_per_job: int = 200
    max_file_mb: int = 50
    max_job_gb: float = 3.0
    min_image_short_side: int = 360          # reject smaller images (SfM cannot use them)
    allowed_image_formats: str = "JPEG,PNG,TIFF"   # Pillow format names

    # ---- Video rules (F-47) ----
    video_max_duration_s: int = 600          # 10 minutes maximum
    video_max_file_mb: int = 200             # 200 MB maximum upload
    video_candidate_fps: float = 10.0        # candidate frame extraction rate (5-10 fps)
    video_min_blur_ratio: float = 0.45       # drop frames with Laplacian var < 0.45 * median
    video_min_parallax_ratio: float = 0.01   # require median displacement >= 1.0% diagonal
    video_frame_budget_draft: int = 100
    video_frame_budget_standard: int = 200
    video_frame_budget_high: int = 350
    video_allowed_formats: str = "MP4,MOV,WEBM,MKV,AVI"

    # ---- Sparse reconstruction gate ----
    min_registered_ratio_fail: float = 0.30   # below this the job fails before dense work
    min_registered_ratio_warn: float = 0.60   # below this the quality report warns
    sfm_max_image_size: int = 2400
    sfm_max_num_features: int = 8192
    reuse_features: bool = True               # cache colmap database.db (F-35)
    enable_preflight: bool = True             # Stage 0 pre-flight check
    skip_preflight_overlap: bool = False      # skip SIFT overlap (for tests/mock images)

    # ---- Dense reconstruction (standard-quality starting values) ----
    dense_undistort_max_image_size: int = 1600
    dense_fusion_max_image_size: int = 1600
    patchmatch_window_radius: int = 5
    patchmatch_num_iterations: int = 5
    patchmatch_geom_consistency: bool = True
    fusion_min_num_pixels: int = 5
    min_fused_points: int = 5000
    poisson_depth: int = 10
    poisson_fallback_depth: int = 8

    # ---- Mesh post-processing ----
    mesh_target_faces: int = 300_000          # decimate down to this many faces
    mesh_max_faces_before_cleanup: int = 1_200_000
    taubin_iterations: int = 3
    bpa_max_points: int = 2_000_000

    # ---- Quality verdict ----
    quality_min_faces_fail: int = 1000
    quality_max_boundary_ratio_warn: float = 0.02
    quality_max_nonmanifold_ratio_warn: float = 0.001

    # ---- Jobs / worker ----
    embedded_worker: bool = True              # run the worker loop inside the API process
    worker_poll_seconds: float = 2.0
    max_job_attempts: int = 3                 # after this many interrupted runs a job is failed
    keep_intermediates: bool = False          # keep dense depth maps / undistorted images

    # ---- API ----
    # Comma separated list. Set CORS_ORIGINS in .env to your frontend URL(s).
    cors_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5173,http://127.0.0.1:5173"
    )
    # Serving /outputs publicly is kept ON so existing frontends keep working.
    # Switch to false once the frontend uses GET /jobs/{id}/result/download.
    public_outputs: bool = True
    storage_bucket_images: str = "orca-images"
    storage_bucket_models: str = "orca-models"
    app_title: str = "ORCA Backend"
    app_description: str = "Object Reconstruction via Photogrammetry"
    app_version: str = "0.2.0"

    # ---- Logging ----
    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_formats(self) -> set:
        return {f.strip().upper() for f in self.allowed_image_formats.split(",") if f.strip()}

    @property
    def allowed_video_exts(self) -> set:
        return {f.strip().lower() for f in self.video_allowed_formats.split(",") if f.strip()}

    def resolve_colmap(self) -> Optional[Path]:
        """COLMAP_BIN env var -> colmap on PATH -> usual Windows install folders."""
        if self.colmap_bin:
            return Path(self.colmap_bin)
        found = shutil.which("colmap")
        if found:
            return Path(found)
        for candidate in _WINDOWS_COLMAP_DEFAULTS:
            if candidate.exists():
                return candidate
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
