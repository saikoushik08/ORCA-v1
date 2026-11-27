"""COLMAP helpers and the sparse reconstruction (Structure-from-Motion) step."""
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from app.config import settings
from app.jobs.errors import PipelineError
from app.services.image_utils import read_image_size
from app.services.presets import Preset, get_preset

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Locating COLMAP and the GPU
# ---------------------------------------------------------------------------
def get_colmap_exe() -> Path:
    """Resolve the COLMAP executable from COLMAP_BIN / PATH / usual install folders."""
    exe = settings.resolve_colmap()
    if exe is None or not exe.exists():
        raise FileNotFoundError(
            "COLMAP executable not found. Set COLMAP_BIN in .env to the full path "
            "of colmap.exe (or colmap) or add COLMAP to PATH."
        )
    return exe


def has_cuda() -> bool:
    """True when an NVIDIA GPU answers to nvidia-smi (or the check is disabled)."""
    if settings.skip_cuda_check:
        return True
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "-L"], capture_output=True, timeout=20).returncode == 0
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize=32)
def _help_text(exe: str, command: str) -> str:
    """Output of `colmap <command> -h`; used to pick option names that exist in the
    installed COLMAP version (they were renamed between releases)."""
    try:
        out = subprocess.run([exe, command, "-h"], capture_output=True, text=True,
                             errors="replace", timeout=60)
        return (out.stdout or "") + (out.stderr or "")
    except Exception:  # noqa: BLE001
        return ""


def pick_option(exe: Path, command: str, candidates: List[str], value) -> List[str]:
    """Return ['--<name>', value] for the first candidate option the installed COLMAP
    knows, or [] if none does. If the help text cannot be read, the first (newest)
    name is used."""
    text = _help_text(str(exe), command)
    for name in candidates:
        if not text or name in text:
            return [f"--{name}", str(value)]
    logger.warning("COLMAP %s has none of the options %s; skipped", command, candidates)
    return []


# ---------------------------------------------------------------------------
# Running COLMAP safely
# ---------------------------------------------------------------------------
def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        proc.kill()


def run_colmap(cmd: list, log_path: Path, stage: str, timeout: Optional[int] = None) -> List[str]:
    """Run one COLMAP command.

    Output goes to `log_path` (not the main log, it can be huge). On a non-zero
    exit or timeout a PipelineError carries the last output lines, so the user
    sees the real reason instead of 'CalledProcessError'. A hung process (and its
    children) is killed.
    """
    timeout = timeout or settings.colmap_stage_timeout_s
    cmd = [str(c) for c in cmd]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: deque = deque(maxlen=3000)

    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    logger.info("[COLMAP] %s: %s", stage, " ".join(cmd[1:3]))
    with open(log_path, "a", encoding="utf-8", errors="replace") as logfile:
        logfile.write(f"\n===== {stage}: {' '.join(cmd)}\n")
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", bufsize=1, **kwargs,
            )
        except OSError as exc:
            raise PipelineError(f"Could not start COLMAP: {exc}", "colmap_start", stage)

        def pump():
            for line in proc.stdout:
                line = line.rstrip()
                lines.append(line)
                logfile.write(line + "\n")
                logfile.flush()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            reader.join(5)
            raise PipelineError(
                f"COLMAP {stage} timed out after {timeout} s and was stopped",
                "timeout", stage,
            )
        reader.join(10)

    if returncode != 0:
        tail = "\n".join(list(lines)[-20:])
        raise PipelineError(
            f"COLMAP {stage} failed (exit code {returncode}). Last output:\n{tail}",
            "colmap_failed", stage,
        )
    return list(lines)


# ---------------------------------------------------------------------------
# Reading a sparse model
# ---------------------------------------------------------------------------
def _parse_analyzer(lines: List[str]) -> dict:
    text = "\n".join(lines)

    def num(pattern, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    return {
        "registered_images": num(r"Registered images:\s*(\d+)", int),
        "points": num(r"Points:\s*(\d+)", int),
        "mean_track_length": num(r"Mean track length:\s*([\d.]+)"),
        "mean_reprojection_error": num(r"Mean reprojection error:\s*([\d.]+)"),
    }


def _model_dirs(sparse_dir: Path) -> List[Path]:
    dirs = [p for p in sparse_dir.iterdir() if p.is_dir()]
    # numeric order: 0, 1, 2 ... 10 (a plain string sort would put 10 before 2)
    return sorted(dirs, key=lambda p: (0, int(p.name)) if p.name.isdigit() else (1, p.name))


def _bin_size(model_dir: Path) -> int:
    f = model_dir / "images.bin"
    return f.stat().st_size if f.exists() else 0


def _registered_names(exe: Path, model_dir: Path, scratch: Path, log: Path) -> Optional[set]:
    """Names of registered images (via a TXT export of the model); None if unavailable."""
    try:
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True)
        run_colmap([exe, "model_converter", "--input_path", model_dir,
                    "--output_path", scratch, "--output_type", "TXT"], log, "model_converter", 300)
        images_txt = scratch / "images.txt"
        # images.txt alternates a header line and a (possibly EMPTY) points line, so
        # empty lines must be kept to keep the pairing right.
        rows = [l for l in images_txt.read_text(encoding="utf-8").splitlines()
                if not l.startswith("#")]
        return {r.split()[-1] for r in rows[::2] if r.strip()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list registered images: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Sparse reconstruction
# ---------------------------------------------------------------------------
def run_colmap_sfm(images_dir: Path, workspace_dir: Path,
                   preset: Optional[Preset] = None, is_video: bool = False) -> dict:
    """
    images -> feature extraction -> matching -> mapping -> pick the best model.

    Returns a JSON-ready dict:
      model_dir, registered_images, total_images, registered_ratio,
      mean_reprojection_error, points, mean_track_length, num_models,
      unregistered_images, warnings
    Raises PipelineError if no model exists or too few images registered.
    """
    preset = preset or get_preset("standard")
    images_dir = Path(images_dir)
    workspace_dir = Path(workspace_dir)
    stage = "sfm"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    image_files = sorted(p for p in images_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_files:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    exe = get_colmap_exe()
    colmap_dir = workspace_dir / "colmap"
    database_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    colmap_images_dir = colmap_dir / "images"
    log = workspace_dir / "colmap.log"

    colmap_dir.mkdir(parents=True, exist_ok=True)

    # Incremental feature reuse (F-35)
    manifest_path = colmap_dir / "features_manifest.json"
    current_hashes = {p.name: _compute_sha256(p) for p in image_files}
    current_params = {
        "sfm_max_image_size": preset.sfm_max_image_size,
        "max_sift_features": preset.max_sift_features,
        "is_video": is_video,
    }

    features_reused = 0
    features_extracted = len(image_files)

    can_reuse = False
    if settings.reuse_features and database_path.exists() and manifest_path.exists():
        try:
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            saved_hashes = saved_manifest.get("hashes", {})
            saved_params = saved_manifest.get("params", {})
            # Parameters must match and all previous images must still exist and be unchanged
            if saved_params == current_params and all(
                saved_hashes.get(name) == current_hashes.get(name) for name in saved_hashes
            ):
                can_reuse = True
                new_images = [p for p in image_files if p.name not in saved_hashes]
                features_reused = len(saved_hashes)
                features_extracted = len(new_images)
                logger.info(
                    "[COLMAP] Reusing cached feature database (%d existing images, %d new images to extract)",
                    features_reused, features_extracted,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[COLMAP] Could not read feature manifest: %s", exc)
            can_reuse = False

    if not can_reuse:
        if database_path.exists():
            try:
                database_path.unlink()
            except PermissionError:
                logger.warning("[COLMAP] database.db is locked by another process; skipping unlink")
        features_reused = 0
        features_extracted = len(image_files)

    # Always ensure clean sparse output dir for the mapper
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True)

    if images_dir.resolve() != colmap_images_dir.resolve():
        if not can_reuse and colmap_images_dir.exists():
            shutil.rmtree(colmap_images_dir)
        colmap_images_dir.mkdir(parents=True, exist_ok=True)
        for p in image_files:
            dst = colmap_images_dir / p.name
            if not dst.exists():
                shutil.copy2(p, dst)

    use_gpu = has_cuda()
    if not use_gpu:
        logger.warning("[COLMAP] No NVIDIA GPU found: using CPU for features/matching (slow).")
    gpu_flag = "1" if use_gpu else "0"

    # Camera model: all frames of one video share single_camera=1.
    # For photos, one camera if all dimensions match; otherwise one per image.
    if is_video:
        camera_args = ["--ImageReader.single_camera", "1"]
    else:
        sizes = {read_image_size(p) for p in image_files}
        camera_args = (["--ImageReader.single_camera", "1"] if len(sizes) == 1
                       else ["--ImageReader.single_camera_per_image", "1"])
        if len(sizes) > 1:
            logger.info("[COLMAP] %d different image sizes: using one camera per image", len(sizes))

    # 1) Features
    run_colmap(
        [exe, "feature_extractor",
         "--database_path", database_path, "--image_path", colmap_images_dir,
         *camera_args,
         *pick_option(exe, "feature_extractor",
                      ["FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"], gpu_flag),
         *pick_option(exe, "feature_extractor",
                      ["FeatureExtraction.max_image_size", "SiftExtraction.max_image_size"],
                      preset.sfm_max_image_size),
         *pick_option(exe, "feature_extractor",
                      ["SiftExtraction.max_num_features", "FeatureExtraction.max_num_features"],
                      preset.max_sift_features)],
        log, "feature_extractor",
    )

    # 2) Matching: sequential for video frames (D4); exhaustive for photos
    if is_video:
        run_colmap(
            [exe, "sequential_matcher", "--database_path", database_path,
             "--SequentialMatching.loop_detection", "1",
             *pick_option(exe, "sequential_matcher",
                          ["FeatureMatching.use_gpu", "SiftMatching.use_gpu"], gpu_flag)],
            log, "sequential_matcher",
        )
    else:
        run_colmap(
            [exe, "exhaustive_matcher", "--database_path", database_path,
             *pick_option(exe, "exhaustive_matcher",
                          ["FeatureMatching.use_gpu", "SiftMatching.use_gpu"], gpu_flag)],
            log, "exhaustive_matcher",
        )

    # 3) Mapping
    run_colmap(
        [exe, "mapper", "--database_path", database_path,
         "--image_path", colmap_images_dir, "--output_path", sparse_dir],
        log, "mapper",
    )

    models = _model_dirs(sparse_dir)
    if not models:
        raise PipelineError(
            "COLMAP finished but built no sparse model. The photos probably do not "
            "overlap enough or are too blurry/featureless. Take 60-70% overlap between "
            "neighbouring shots on a textured object.",
            "no_sparse_model", stage,
        )

    # Choose the model that registered the most images (not just 'the first folder').
    scored = []
    for m in models:
        try:
            stats = _parse_analyzer(run_colmap(
                [exe, "model_analyzer", "--path", m], log, "model_analyzer", 300))
        except PipelineError as exc:
            logger.warning("model_analyzer failed for %s: %s", m.name, exc)
            stats = {}
        reg = stats.get("registered_images")
        err = stats.get("mean_reprojection_error")
        scored.append((m, stats, reg if reg is not None else -1, err if err is not None else 99.0))

    if all(s[2] < 0 for s in scored):          # analyzer output unreadable: use file size
        best = max(scored, key=lambda s: _bin_size(s[0]))
    else:
        best = max(scored, key=lambda s: (s[2], -s[3]))
    best_dir, stats, registered, _ = best

    total = len(image_files)
    names = _registered_names(exe, best_dir, workspace_dir / "sparse_txt", log)
    if registered < 0:
        registered = len(names) if names is not None else 0
    unregistered = sorted(p.name for p in image_files if names is not None and p.name not in names)
    ratio = registered / total if total else 0.0

    try:
        manifest_path.write_text(
            json.dumps({"hashes": current_hashes, "params": current_params}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[COLMAP] Could not write features manifest: %s", exc)

    result = {
        "model_dir": str(best_dir),
        "registered_images": registered,
        "total_images": total,
        "registered_ratio": round(ratio, 4),
        "mean_reprojection_error": stats.get("mean_reprojection_error"),
        "points": stats.get("points"),
        "mean_track_length": stats.get("mean_track_length"),
        "num_models": len(models),
        "unregistered_images": unregistered,
        "features_reused": features_reused,
        "features_extracted": features_extracted,
        "warnings": [],
    }
    logger.info("[COLMAP] Best sparse model %s: %d/%d images registered (%.0f%%)",
                best_dir.name, registered, total, ratio * 100)

    if ratio < settings.min_registered_ratio_fail:
        raise PipelineError(
            f"Only {registered} of {total} photos could be placed in 3D "
            f"({ratio:.0%}; minimum {settings.min_registered_ratio_fail:.0%}). "
            "Take photos with 60-70% overlap, sharp and well lit, all around the object.",
            "sparse_gate", stage,
        )
    if ratio < settings.min_registered_ratio_warn:
        result["warnings"].append(
            f"Only {ratio:.0%} of photos registered; the model may be incomplete."
        )
    return result
