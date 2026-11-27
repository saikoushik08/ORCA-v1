"""Video extraction and keyframe selection (F-47).

End-to-end pipeline:
1. Validate video format, size, and duration.
2. Read video metadata (fps, duration, resolution, rotation).
3. Sample candidate frames at candidate fps (default 5-10 fps).
4. Normalise rotation metadata (no sideways portrait video).
5. Filter blurry frames (Laplacian variance < 0.6 * set median).
6. Filter by parallax baseline (median feature displacement >= 3.5% of diagonal).
7. Drop near-duplicates (perceptual hash).
8. Cap to frame budget via uniform temporal thinning.
"""
import hashlib
import io
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


class VideoRejected(ValueError):
    """The video cannot be used; the message is safe to show to the user."""


@dataclass
class VideoMetadata:
    duration_s: float
    fps: float
    frame_count: int
    width: int
    height: int
    rotation: int  # 0, 90, 180, 270
    size_bytes: int


@dataclass
class ExtractedFrame:
    frame_index: int
    timestamp_s: float
    image_bytes: bytes
    width: int
    height: int
    sha256: str
    size_bytes: int
    blur_score: float
    dhash: int


def _compute_dhash(gray: np.ndarray) -> int:
    """Compute 64-bit difference hash for fast duplicate detection."""
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    val = 0
    for b in diff.flatten():
        val = (val << 1) | int(b)
    return val


def _hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of Laplacian at fixed size (or native size) as blur metric."""
    # Scale to ~1024px wide for consistent thresholding across resolutions
    h, w = gray.shape[:2]
    target_w = 1024
    if w > target_w:
        scale = target_w / w
        gray_scaled = cv2.resize(gray, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        gray_scaled = gray
    lap = cv2.Laplacian(gray_scaled, cv2.CV_64F)
    return float(lap.var())


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def probe_video(video_path: Path) -> VideoMetadata:
    """Read metadata from a video file."""
    if not video_path.exists():
        raise VideoRejected(f"Video file not found: {video_path}")
    size_bytes = video_path.stat().st_size

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoRejected(f"Cannot decode video {video_path.name}. Unsupported or corrupt format.")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        rotation = 0
        try:
            rot_meta = cap.get(cv2.CAP_PROP_ORIENTATION_META)
            if rot_meta in (90, 180, 270):
                rotation = int(rot_meta)
        except Exception:  # noqa: BLE001
            rotation = 0

        if rotation in (90, 270):
            width, height = height, width

        duration_s = frame_count / fps if fps > 0 else 0.0
        return VideoMetadata(
            duration_s=duration_s,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            rotation=rotation,
            size_bytes=size_bytes,
        )
    finally:
        cap.release()


def _calculate_feature_displacement(gray1: np.ndarray, gray2: np.ndarray) -> float:
    """Calculate median feature displacement (optical flow / ORB) between two grayscale frames."""
    # Detect Shi-Tomasi corners or ORB points on gray1
    p0 = cv2.goodFeaturesToTrack(
        gray1, maxCorners=500, qualityLevel=0.01, minDistance=10, blockSize=7
    )
    if p0 is None or len(p0) < 10:
        # Fallback to ORB
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
            return 0.0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if not matches:
            return 0.0
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        displacements = np.linalg.norm(pts2 - pts1, axis=1)
        return float(np.median(displacements))

    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, p0, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    if p1 is None or status is None:
        return 0.0

    valid = status.flatten() == 1
    if np.sum(valid) < 8:
        return 0.0

    dists = np.linalg.norm(p1[valid] - p0[valid], axis=1)
    return float(np.median(dists))


def extract_and_select_keyframes(
    video_path: Path,
    frame_budget: int = 120,
    candidate_fps: Optional[float] = None,
    min_parallax_ratio: Optional[float] = None,
    min_blur_ratio: Optional[float] = None,
    max_duration_s: Optional[int] = None,
) -> Tuple[List[ExtractedFrame], dict]:
    """Extract and select keyframes from a video file.

    Returns:
        (keyframes, report_dict)
    """
    candidate_fps = candidate_fps or settings.video_candidate_fps
    min_parallax_ratio = min_parallax_ratio or settings.video_min_parallax_ratio
    min_blur_ratio = min_blur_ratio or settings.video_min_blur_ratio
    max_duration = max_duration_s or settings.video_max_duration_s

    meta = probe_video(video_path)
    if meta.duration_s > max_duration:
        raise VideoRejected(
            f"Video duration is {meta.duration_s:.1f}s, which exceeds maximum "
            f"allowed {max_duration}s ({max_duration // 60} mins)."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoRejected("Failed to open video stream")

    # Step 1: sample candidate frames
    raw_candidates = []
    fps = meta.fps if meta.fps > 0 else 30.0
    frame_step = max(1, int(round(fps / candidate_fps)))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            if meta.rotation != 0:
                frame = _apply_rotation(frame, meta.rotation)
            timestamp_s = frame_idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = _laplacian_variance(gray)
            dhash = _compute_dhash(gray)
            raw_candidates.append({
                "frame_index": frame_idx,
                "timestamp_s": timestamp_s,
                "frame": frame,
                "gray": gray,
                "blur_score": blur_score,
                "dhash": dhash,
            })
        frame_idx += 1
    cap.release()

    total_extracted = len(raw_candidates)
    if total_extracted == 0:
        raise VideoRejected("No frames could be extracted from video")

    # Step 2: Blur filtering (drop below min_blur_ratio * median)
    all_blur_scores = [c["blur_score"] for c in raw_candidates]
    median_blur = float(np.median(all_blur_scores)) if all_blur_scores else 0.0
    blur_threshold = median_blur * min_blur_ratio

    sharp_candidates = []
    dropped_blurry = 0
    for c in raw_candidates:
        if c["blur_score"] < blur_threshold:
            dropped_blurry += 1
        else:
            sharp_candidates.append(c)

    if not sharp_candidates:
        # Fallback to top candidates if all were rejected
        sharp_candidates = sorted(raw_candidates, key=lambda x: x["blur_score"], reverse=True)[:5]
        dropped_blurry = total_extracted - len(sharp_candidates)

    # Step 3: Parallax selection & near-duplicate drop with continuous motion accumulation
    h, w = sharp_candidates[0]["gray"].shape[:2]
    diagonal = math.hypot(w, h)
    # Parallax threshold: use the configured ratio directly
    # (values < 0.03 are for video revolving scans that need more frames)
    min_dist_px = diagonal * min_parallax_ratio

    parallax_selected = [sharp_candidates[0]]
    dropped_parallax = 0
    dropped_duplicates = 0

    last_keyframe = sharp_candidates[0]
    accumulated_disp = 0.0

    for cand in sharp_candidates[1:]:
        # Check duplicate via dhash
        if _hamming_distance(cand["dhash"], last_keyframe["dhash"]) <= 2:
            dropped_duplicates += 1
            continue

        step_disp = _calculate_feature_displacement(last_keyframe["gray"], cand["gray"])
        accumulated_disp += step_disp
        time_gap = cand["timestamp_s"] - last_keyframe["timestamp_s"]

        # Select if accumulated displacement >= min_dist_px OR if steady motion spanned >= 0.8s
        if accumulated_disp >= min_dist_px or (time_gap >= 0.8 and accumulated_disp >= min_dist_px * 0.15):
            parallax_selected.append(cand)
            last_keyframe = cand
            accumulated_disp = 0.0
        else:
            dropped_parallax += 1

    # If parallax was too strict and gave fewer than minimum images, relax threshold
    if len(parallax_selected) < settings.min_images_per_job and len(sharp_candidates) >= settings.min_images_per_job:
        # Uniform sample from sharp candidates
        take_indices = np.linspace(0, len(sharp_candidates) - 1, min(len(sharp_candidates), frame_budget), dtype=int)
        parallax_selected = [sharp_candidates[i] for i in np.unique(take_indices)]
        dropped_parallax = len(sharp_candidates) - len(parallax_selected)

    # Step 4: Uniform thinning down to frame budget
    dropped_thinning = 0
    if len(parallax_selected) > frame_budget:
        indices = np.round(np.linspace(0, len(parallax_selected) - 1, frame_budget)).astype(int)
        indices = np.unique(indices)
        dropped_thinning = len(parallax_selected) - len(indices)
        final_candidates = [parallax_selected[i] for i in indices]
    else:
        final_candidates = parallax_selected

    # Step 5: Encode final frames to JPEG bytes
    final_keyframes: List[ExtractedFrame] = []
    for c in final_candidates:
        # Encode as JPEG
        success, buf = cv2.imencode(".jpg", c["frame"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            continue
        data = buf.tobytes()
        final_keyframes.append(
            ExtractedFrame(
                frame_index=c["frame_index"],
                timestamp_s=round(c["timestamp_s"], 3),
                image_bytes=data,
                width=w,
                height=h,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                blur_score=round(c["blur_score"], 2),
                dhash=c["dhash"],
            )
        )

    report = {
        "duration_s": round(meta.duration_s, 2),
        "fps": round(meta.fps, 2),
        "resolution": f"{meta.width}x{meta.height}",
        "frames_extracted": total_extracted,
        "frames_kept": len(final_keyframes),
        "reasons_dropped": {
            "blurry": dropped_blurry,
            "insufficient_parallax": dropped_parallax,
            "duplicate": dropped_duplicates,
            "budget_thinning": dropped_thinning,
        },
    }

    logger.info(
        "[Video] %s: extracted %d candidates, kept %d keyframes (blurry=%d, parallax=%d, dupes=%d, thinned=%d)",
        video_path.name, total_extracted, len(final_keyframes),
        dropped_blurry, dropped_parallax, dropped_duplicates, dropped_thinning
    )
    return final_keyframes, report
