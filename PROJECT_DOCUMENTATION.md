# ORCA Backend — Comprehensive Project Documentation

**ORCA (Object Reconstruction via Photogrammetry)** is a FastAPI + COLMAP + Open3D backend service that converts overlapping photographs or revolving scan videos into cleaned, vertex-colored 3D meshes (`.glb`, `.ply`, `.obj`, `.stl`) with an automated 100-point quality report.

- **Version:** `0.2.0` (`app/config.py`)
- **Runtime Stack:** Python 3.12+, FastAPI, Uvicorn, Pydantic v2 (`pydantic-settings`), Supabase (`supabase-py`), COLMAP (CUDA), Open3D, Trimesh, SciPy, NumPy, OpenCV, Pillow, NetworkX.

---

## 1. How the Project Is Built (Architecture & Components)

### 1.1 Directory Layout

```text
orca-backend/
├── app/
│   ├── main.py                       # FastAPI app, lifespan startup checks, CORS, /outputs static mount
│   ├── config.py                     # Centralized Pydantic Settings (.env overrides)
│   ├── logging_config.py             # Structured logger with ContextVar [job=...] & per-job job.log
│   ├── api/routes/
│   │   ├── health.py                 # GET /health, GET /health/supabase
│   │   ├── upload.py                 # POST /upload/image/{job_id}, POST /upload/video/{job_id}
│   │   └── jobs.py                   # Job CRUD, /preflight, /start, /rerun, /logs, /events, /quality, /result
│   ├── jobs/
│   │   ├── job_states.py             # JobStatus enum & state-transition rule table
│   │   ├── job_manager.py            # State transition & lifecycle coordinator
│   │   ├── job_repository.py         # All Supabase CRUD (jobs, images, job_events) with schema fallback
│   │   └── errors.py                 # JobNotFound, InvalidTransition, TransitionConflict, PipelineError
│   ├── workers/
│   │   ├── local_worker.py           # Polling worker loop + startup crash recovery
│   │   ├── task_dispatcher.py        # Atomic job claiming (PENDING -> PROCESSING)
│   │   └── worker_executor.py        # End-to-end job execution, GPU lock, cloud sync, output publishing
│   ├── storage/
│   │   ├── supabase_client.py        # Supabase client initialized from .env
│   │   └── cloud_sync.py             # StorageSync.ensure_local & publish_outputs (orca-images / orca-models)
│   └── services/
│       ├── presets.py                # Preset profiles: draft, standard, high
│       ├── image_utils.py            # Pillow image decoding, EXIF baking, SHA-256 & camera key checks
│       ├── video_utils.py            # Video probing, blur/parallax/dHash keyframe selection
│       ├── preflight.py              # Stage 0 CPU check: blur, exposure, dHash dupes, SIFT+RANSAC graph
│       ├── power_utils.py            # Windows SetThreadExecutionState keep_awake context manager
│       └── method1_photogrammetry/
│           ├── pipeline.py           # Orchestrates Stages 1 -> 4
│           ├── colmap_runner.py      # Sparse SfM (feature_extractor, matcher, mapper, analyzer)
│           ├── dense_reconstruction.py # Dense MVS (image_undistorter, patch_match_stereo, stereo_fusion)
│           ├── point_cloud_isolation.py # Subject cropping, RANSAC plane, glare/yellow & glass filter, normals
│           ├── mesh_builder.py       # Screened Poisson / BPA fallback, air pruning, cleaning, color transfer
│           └── quality_checker.py    # 100-point explainable quality score & PASS/WARN/FAIL verdict
├── scripts/
│   ├── run_video_job.py              # End-to-end video CLI (API mode or --local direct mode)
│   ├── run_pipeline_local.py         # End-to-end photo folder CLI (local execution)
│   ├── run_test_job.py               # HTTP API smoke-test client for uploads/images
│   └── queue_smoke_test.py           # Queueing verification script
├── tests/                            # Pytest suite (fake Supabase client for offline/CPU testing)
├── schema.sql                        # PostgreSQL/Supabase migration (jobs, images, job_events, claim_next_job)
└── requirements.txt                  # Core dependencies
```

---

### 1.2 Job State Machine & Worker Model

Defined in `app/jobs/job_states.py` and enforced atomically by `JobRepository.transition()`:

```text
CREATED ──► UPLOADING ──► PENDING ──► PROCESSING ──► COMPLETED
   │            │            │            │
   └────────────┴────────────┴────────────┴──► FAILED
                                          │
                       (worker recovery)  └──► PENDING
```

- **State Enforcement:** `JobRepository.transition` issues a conditional `UPDATE ... WHERE job_id = ? AND status = from_status`. If two workers race, only one succeeds and the other receives `TransitionConflict`.
- **Audit Trail:** Every transition appends an event row to `job_events`.
- **Execution Modes:**
  - **Embedded Worker** (`EMBEDDED_WORKER=true`, default): `app/main.py` launches `LocalWorker.run_forever()` in a background daemon thread during FastAPI lifespan startup.
  - **Standalone Worker** (`EMBEDDED_WORKER=false`): Run `python -m app.workers.local_worker` separately (recommended: one worker per GPU).
  - **GPU Serialization:** `WorkerExecutor._RUN_LOCK` guarantees only one COLMAP reconstruction runs on the GPU at a time per process.

---

### 1.3 API Endpoints

| Method & Path | Purpose |
|---|---|
| `GET /` | Basic health banner (`ORCA backend running`) |
| `GET /health` | Component readiness check: COLMAP binary, CUDA (`nvidia-smi`), disk free space ($\ge 20\text{ GB}$), Supabase connectivity (returns `503` if unhealthy) |
| `GET /health/supabase` | Supabase table connectivity check |
| `POST /jobs/` | Create a new job (`method=method_1`, `preset=draft\|standard\|high`); `method_2` returns `501` |
| `GET /jobs/{job_id}` | Current status, stage, progress percentage, image count, failure details, and quality summary |
| `POST /upload/image/{job_id}` | All-or-nothing batch photo upload (JPEG/PNG/TIFF, $\ge 360\text{px}$ short side, EXIF transpose, SHA-256 deduplication, max 200 images / 3 GB) |
| `POST /upload/video/{job_id}` | Video upload (`.mp4`, `.mov`, `.webm`, `.mkv`, `.avi`, max 200 MB) with automatic keyframe extraction & selection |
| `POST /jobs/{job_id}/preflight` / `GET /jobs/{job_id}/preflight` | Run or inspect Stage 0 CPU pre-flight report |
| `GET /jobs/{job_id}/frames` | Inspect video keyframe extraction & drop breakdown report |
| `POST /jobs/{job_id}/start` | Transition `UPLOADING -> PENDING` and return `202` immediately |
| `POST /jobs/{job_id}/rerun` | Create a child job reusing parent images and cached COLMAP `database.db` features |
| `GET /jobs/{job_id}/logs?tail=200` | Tail per-job log lines from `workspace/<job_id>/job.log` |
| `GET /jobs/{job_id}/events` | List state-change audit events |
| `GET /jobs/{job_id}/quality` | Retrieve `quality.json` (100-point rubric) |
| `GET /jobs/{job_id}/result` | List available downloadable output files & prioritized `glb`/`ply`/`obj`/`stl` URLs |
| `GET /jobs/{job_id}/result/download` | Download a whitelisted artifact (`?file=...` or `?format=...`) |

---

### 1.4 Quality Presets (`app/services/presets.py`)

| Parameter | `draft` | `standard` | `high` |
|---|---|---|---|
| `sfm_max_image_size` | 1600 | 2400 | 3200 |
| `max_sift_features` | 4096 | 8192 | 8192 |
| `video_frame_budget` | 80 | 200 | 350 |
| `dense_undistort_max_image_size` | 1000 | 1600 | 2400 |
| `dense_fusion_max_image_size` | 1000 | 1600 | 2400 |
| `patchmatch_window_radius` | 4 | 5 | 5 |
| `patchmatch_num_iterations` | 3 | 5 | 5 |
| `patchmatch_geom_consistency` | `False` | `True` | `True` |
| `fusion_max_reproj_error` | 2.0 | 2.5 | 3.0 |
| `poisson_depth` | 9 | 11 | 11 |
| `mesh_target_faces` | 80,000 | 400,000 | 800,000 |
| `taubin_iterations` | 2 | 4 | 4 |

---

### 1.5 End-to-End Reconstruction Pipeline

1. **Stage 0 — Pre-flight (`app/services/preflight.py`):**
   - Validates image readability, resolution, Laplacian blur variance ($< 40\%$ of median flagged), exposure clipping ($> 25\%$ saturated/dark), and dHash duplicates (Hamming $\le 3$).
   - Builds a view-connectivity graph on up to 60 sampled images using OpenCV SIFT + BFMatcher + RANSAC Fundamental Matrix; rejects sets where the largest connected component is $< 60\%$.
2. **Stage 1 — Sparse SfM (`app/services/method1_photogrammetry/colmap_runner.py`):**
   - Supports cached feature reuse via `database.db` + `features_manifest.json` (hashes + parameters).
   - Runs `feature_extractor` $\rightarrow$ `sequential_matcher` (with `loop_detection=1` for video) or `exhaustive_matcher` (photos) $\rightarrow$ `mapper`.
   - Evaluates all resulting sparse sub-models with `model_analyzer`, selects the model with the most registered images (tie-breaking on lowest reprojection error), and enforces `min_registered_ratio_fail` (`0.30`).
3. **Stage 2 — Dense MVS (`app/services/method1_photogrammetry/dense_reconstruction.py`):**
   - Verifies CUDA availability via `nvidia-smi`.
   - Runs `image_undistorter` $\rightarrow$ `patch_match_stereo` $\rightarrow$ `stereo_fusion` to produce `fused.ply`.
4. **Stage 2.5 — Point Cloud Isolation (`app/services/method1_photogrammetry/point_cloud_isolation.py`):**
   - Estimates the camera ray intersection (`FocusPoint`), refines to the subject centroid, verifies upright orientation, performs radial subject cropping, rejects bisecting RANSAC planes while removing external support planes, runs statistical/radius outlier removal, filters yellow/glare/spectacle-refraction artifacts, selects the core subject cluster via DBSCAN, and re-orients all point normals outward.
5. **Stage 3 — Surface Meshing & Post-Processing (`app/services/method1_photogrammetry/mesh_builder.py`):**
   - Runs Screened Poisson reconstruction (`depth=11`, `trim_percentile=0.0`, `scale=1.1`, `linear_fit=True`) directly on `fused_isolated.ply` (falling back to Ball-Pivoting only if Poisson fails).
   - Executes pre-decimation (`1.2M` faces), basic clean, empty-air triangle pruning, dominant component filtering, Taubin smoothing, target decimation, manifold/boundary cleanup, inverse-distance-weighted vertex color transfer with facial/clothing glare polish, and multi-format export (`.ply`, `.obj`, `.stl`, `.glb`).
6. **Stage 4 — Quality Scoring (`app/services/method1_photogrammetry/quality_checker.py`):**
   - Scores the output out of 100 across Sparse (30 pts), Dense (10 pts), Mesh topology (45 pts), and Point-Cloud Fidelity (15 pts: completeness + trimmed relative accuracy), yielding `PASS`, `WARN`, or `FAIL`.

---

## 2. Fixes & Changes Implemented So Far

### 2.1 Video Ingestion & Photogrammetry Performance
- **Longer Video Support (`app/config.py`, `app/services/video_utils.py`):** Raised `video_max_duration_s` to `600s` (10 minutes) and `video_max_file_mb` to `200 MB`; tuned `video_min_blur_ratio` (`0.45`) and continuous parallax accumulation (`video_min_parallax_ratio = 0.01`) with a time-gap fallback (`>= 0.8s`) so slow revolving video scans keep dense coverage without stalling.
- **PatchMatch Stereo Bottleneck Fix (`app/services/method1_photogrammetry/dense_reconstruction.py`):** Disabled the second geometric consistency pass automatically for video (`geom = preset.patchmatch_geom_consistency and not is_video`) because video keyframes already have continuous temporal parallax and high overlap, cutting PatchMatch execution from $> 1\text{ hr}$ stalls down to normal runtime while relying on `stereo_fusion` for outlier filtering.
- **Windows Power & Screen Sleep Prevention (`app/services/power_utils.py`, `app/workers/worker_executor.py`):** Added `keep_awake(display=True)` using Windows `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)` during job execution so Modern Standby/screen timeout does not throttle GPU execution.

### 2.2 Point Cloud Isolation & Reconstruction Artifact Fixes
- **Subject Centering & Envelope Tightening (`app/services/method1_photogrammetry/point_cloud_isolation.py`):**
  - Replaced the fixed `max(1.5, radius_thresh)` crop (which kept 3m room walls/floors) with camera-focus refinement toward the subject's authentic skin/core centroid and bounded the crop to `0.70m–0.88m`.
  - Fixed DBSCAN cluster selection to pick the **largest cluster** within `0.70m` of the focus point instead of whichever tiny noise speck happened to sit closer.
- **Automatic Upright Orientation Fix (`point_cloud_isolation.py`):**
  - Fixed an inverted `up_vector` bug caused by COLMAP world coordinates pointing toward the floor in portrait/rotated captures. Now verifies upright direction by comparing the lateral cross-sectional spread of the upper vs. lower halves of the cropped subject (torso/shoulders are consistently wider than the head crown), ensuring head, eye-level, and chin zones map to the true anatomy.
- **Non-Bisecting RANSAC Plane Check (`point_cloud_isolation.py`):**
  - Added signed-distance two-sided split validation (`min_split < 0.04`): prevents RANSAC from accidentally deleting planes that cut through the subject's face or torso while still removing external tables/floors.
- **Yellow Wall Bleed, Chin Flakes & Spectacle Glass Refraction Removal (`point_cloud_isolation.py`):**
  - Added `filter_yellow_and_specular_noise`: identifies authentic human skin ($R - G \ge 0.15$, $G/R \le 0.70$, $B < G$) to protect it while deleting spurious glass refraction points in the eye-level aperture (preventing Poisson from building opaque triangular sheets over spectacles lenses), deleting yellow/cream noise crusts on the chin/beard/neck, and recoloring remaining minor surface glare from clean neighboring donors.
- **100% Outward Normal Alignment (`point_cloud_isolation.py`, `mesh_builder.py`):**
  - Enforced $\vec{n} \cdot \vec{v}_{\text{out}} > 0$ (relative to focus/centroid) across the isolated cloud and pre-Poisson cloud so inward-facing normals cannot force Poisson to generate double-layer surfaces or crossing feature flaps on the nose and cheeks.
- **Hole Preservation & Empty-Air / Shoulder Shard Elimination (`mesh_builder.py`):**
  - Moved Poisson meshing to Stage 3 directly on `fused_isolated.ply` with `trim_percentile=0.0` (and adaptive normal radius estimation based on median NN spacing) to eliminate black holes/gouges on cheeks, temples, and throat.
  - Added `_prune_empty_air_faces(max_dist_mm=16.0)` to remove Poisson balloon triangles and hanging shoulder wings extending $> 16\text{mm}$ into empty air.
  - Removed `trimesh.repair.fill_holes(tm)` from `_repair_watertight` (which triangulated open torso boundaries into giant fan-blade shards), retaining non-manifold edge cleanup, stretched boundary pruning, and winding repair.
- **Performance & Windows Threading Fixes (`mesh_builder.py`, `point_cloud_isolation.py`):**
  - Lowered `mesh_max_faces_before_cleanup` from `3,000,000` to `1,200,000` (`app/config.py`) and subsampled NN distance checks to 20k–500k points for sub-2s execution.
  - Switched SciPy `cKDTree.query(..., workers=1)` in `_prune_empty_air_faces`, `_transfer_vertex_colors`, `filter_yellow_and_specular_noise`, and DBSCAN proxy lookup to avoid Windows Python 3.12 multiprocessing contention.

### 2.3 Exports, Cloud Sync, Job Lifecycle & Error Resilience
- **Multi-Format Exports & Web Viewer Support (`mesh_builder.py`, `app/api/routes/jobs.py`):** Exports `final_mesh.ply`, `final_mesh.obj`, `final_mesh.stl`, and `final_mesh.glb` (prioritized in `/jobs/{id}/result`), plus K=7 inverse-distance-weighted vertex color transfer with facial/clothing glare polish.
- **Cloud Sync Size Guard (`app/storage/cloud_sync.py`):** Skips uploading `fused.ply` and `fused_isolated.ply` when $> 50\text{ MB}$ to prevent Supabase `413 Payload Too Large` errors and multi-minute HTTP timeouts while still uploading meshes, reports, and logs.
- **Status Conflict & Windows File-Lock Resilience:**
  - `app/jobs/job_manager.py`: `mark_completed` catches `TransitionConflict` and calls `JobRepository.update_status(..., JobStatus.COMPLETED, ...)` so completed reconstructions are never stranded as `failed`/`processing` due to a status race.
  - `app/services/method1_photogrammetry/colmap_runner.py`: Wrapped `database_path.unlink()` in `try...except PermissionError` for Windows lock safety (`[WinError 32]`).
  - `scripts/run_video_job.py`: Checks whether `final_mesh.glb` exists on disk before treating a database status conflict as a failed run.
- **Quality Checker & Rerun Capabilities:** Added F-30 100-point rubric with trimmed surface fidelity evaluation (`quality_checker.py`) and child-job rerun endpoint with `database.db` reuse (`POST /jobs/{id}/rerun`).

---

## 3. Changes & Work Left To Do

### 3.1 Bugs / Edge Cases Observed in Code & Runtime
1. **Local CLI vs. Embedded/Standalone Worker Race Condition (`scripts/run_video_job.py`, `scripts/run_pipeline_local.py`):**
   - `run_video_job.py --local` transitions the job to `PENDING` (`JobManager.start_job`) before calling `LocalWorker.execute(job_id)`. If a FastAPI server (`EMBEDDED_WORKER=true`) or `python -m app.workers.local_worker` is running simultaneously, both processes can claim/collide on the same `workspace/<job_id>/colmap/database.db`.
   - **Action:** Add a direct local execution claim or check for an already active worker before calling `start_job`, or separate `run_locally` into a distinct worker-lock/non-queueing path.
2. **Supabase Schema Migration Not Applied on Active Database (`schema.sql`):**
   - Runtime logs show `PGRST204` warnings (`Could not find the 'input_type' column of 'jobs' in the schema cache` and `images table lacks optional columns`). As a result, `JobRepository.create_job()` falls back to inserting only core columns — silently dropping `preset` and `input_type` so `--preset high` fell back to `standard` on job `7bb1c890`.
   - **Action:** Execute `schema.sql` in the Supabase SQL Editor (and reload PostgREST schema cache) so `preset`, `input_type`, `current_stage`, `progress`, `result`, and `job_events` persist properly.
3. **API Video Frame Budget Cap Mismatch (`app/api/routes/upload.py` vs. `app/config.py` / `app/services/presets.py`):**
   - `upload_video` line 236 caps `max_frames` using `settings.max_images_per_job` (`200`), even though `presets.py` defines `video_frame_budget_high = 350` and CLI users pass `--max-frames 300`.
   - **Action:** Align `max_images_per_job` (e.g., raise to `400` or separate `max_video_keyframes`) with the `high` preset budget.
4. **Remaining `workers=-1` & Full-Cloud NN Calls (`point_cloud_isolation.py`):**
   - In `densify_and_fill_holes_knn()` (`point_cloud_isolation.py` lines 313, 329, 338) and `isolate_object_point_cloud` lines 545, 670, `cKDTree.query(..., workers=-1)` and un-subsampled `compute_nearest_neighbor_distance()` still remain when `fill_holes=True` is enabled (as it is in `pipeline.py` line 79).
   - **Action:** Update `densify_and_fill_holes_knn` to use `workers=1` and subsampled NN distance estimates for speed and Windows stability.
5. **Strict Anatomical Isolation Generalization (`point_cloud_isolation.py`):**
   - Current color/height rules in `filter_yellow_and_specular_noise` are tuned for indoor portrait/upper-body busts (head/spectacles/chin/shirt). For general non-human objects (shoes, figurines, yellow/white products), aggressive yellow/bleached filtering could misclassify genuine bright-yellow or cream object textures as glare.
   - **Action:** Gate skin/eye/chin heuristics behind an object-mode check or make glare removal strictly spatial/outlier-based (`min_dists > floating_thresh` + silhouette boundary check) unless portrait mode is selected.

### 3.2 Failing Tests & Stale Documentation
6. **Fix 2 Failing Tests (`pytest -q` currently reports `99 passed, 2 failed, 5 skipped`):**
   - `tests/test_presets.py::test_presets_table_values`: Asserts outdated values (`standard.video_frame_budget == 120`, `poisson_depth == 10`, `mesh_target_faces == 300_000`, etc.) instead of current `app/services/presets.py` values (`200`, `11`, `400_000`, etc.).
   - `tests/test_quality_isolation.py::test_point_cloud_isolation_filters_table_plane_and_outliers`: Tightened radial crop (`0.88m`) removes most of the synthetic ground plane before RANSAC runs (`ratio=0.028 < 0.15`), causing `assert report["plane_points_removed"] > 0` to fail. Either preserve ground-plane removal testing with a closer plane / adjusted test geometry, or verify combined removal (`initial_points - final_points`).
7. **Update `README.md`:**
   - `README.md` still lists implemented features (presets, video input, pre-flight check, object isolation, multi-format `.glb`/`.obj`/`.stl` exports, Supabase output upload, 100-point quality score) under "Known limits / not implemented" and cites the old `480px` minimum short side instead of `360px`.

### 3.3 Unimplemented Features & Production Hardening
8. **Authentication & Authorization (`app/main.py`, `app/api/routes/jobs.py`):**
   - `PUBLIC_OUTPUTS=true` serves `/outputs` without authentication, and `owner_id` (present in `schema.sql` and `JobRepository.create_job`) is not wired to auth/JWT middleware or ownership checks on API routes.
   - **Action:** Add Supabase Auth JWT verification, enforce per-user `owner_id` scoping on `/jobs/*` and `/upload/*`, and set `PUBLIC_OUTPUTS=false`.
9. **Job Heartbeat & Stale Worker Recovery (`schema.sql`, `app/workers/worker_executor.py`):**
   - `heartbeat_at` exists in `schema.sql` and `claim_next_job()` exists as a SQL RPC, but `WorkerExecutor` does not update `heartbeat_at` periodically during long COLMAP runs, and `TaskDispatcher` still polls via Python select + `transition()` rather than `claim_next_job()`.
10. **Method 2 (AI / NeRF Reconstruction):**
    - `ReconstructionMethod.method_2` in `app/api/routes/jobs.py` currently returns `HTTP 501` (`Method 'method_2' is not implemented`). Implement if neural reconstruction (e.g., 3D Gaussian Splatting / Nerfstudio) is in project scope, or remove from the API enum if out of scope.
11. **Repository & Workspace Housekeeping:**
    - Clean up root-level experimental `.ply`/`.png` artifacts (`bpa_cleaned_full.ply`, `test_bpa.ply`, `test_poisson_trimmed.ply`, `tight_poisson.ply`, `new_xy.png`, `old_xy.png`, etc.) and consider an automated retention policy for `workspace/` and `uploads/` folders after job completion.
