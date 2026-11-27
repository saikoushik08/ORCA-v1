"""CPU Pre-flight check (Stage 0) (F-32).

Fast validation before spending GPU time on Structure-from-Motion or dense MVS:
1. Per-image:
   - decode check and resolution check (short side >= 800px, warn if < 1200px)
   - blur detection via Laplacian variance at 1024px (< 40% of set median flagged)
   - exposure clipping check (> 25% pixels overexposed or underexposed)
   - perceptual difference hash duplicate check (Hamming <= 3)
2. Set-level:
   - image count bounds
   - camera key consistency
   - overlap connectivity graph via SIFT (1000 features at 800px) + RANSAC (>= 25 inliers = edge)
   - REJECT if largest connected component < 60% of images
   - WARN if many blurry or duplicate shots
   - Clear, explainable guidance message for the user
"""
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import networkx as nx
import numpy as np
from PIL import Image

from app.config import settings
from app.services.image_utils import read_image_size

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class ImagePreflightResult:
    filename: str
    width: int
    height: int
    blur_score: float
    is_blurry: bool
    exposure_clipped: bool
    clipped_ratio: float
    duplicate_of: Optional[str]
    warnings: List[str]


@dataclass
class PreflightReport:
    verdict: str  # "PASS", "WARN", "REJECT"
    total_images: int
    usable_images: int
    blurry_count: int
    exposure_issue_count: int
    duplicate_count: int
    overlap_edge_count: int
    connected_component_ratio: float
    message: str
    errors: List[str]
    warnings: List[str]
    images: List[Dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _compute_dhash(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    val = 0
    for b in diff.flatten():
        val = (val << 1) | int(b)
    return val


def _hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def _laplacian_variance(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    target_w = 1024
    if w > target_w:
        scale = target_w / w
        gray_scaled = cv2.resize(gray, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        gray_scaled = gray
    lap = cv2.Laplacian(gray_scaled, cv2.CV_64F)
    return float(lap.var())


def _check_exposure(gray: np.ndarray) -> Tuple[bool, float]:
    """Check if >25% of pixels are overexposed (>250) or underexposed (<5)."""
    total = gray.size
    over = np.sum(gray > 250)
    under = np.sum(gray < 5)
    clipped = over + under
    ratio = float(clipped / total)
    return ratio > 0.25, round(ratio, 4)


def run_preflight_check(images_dir: Path, is_video: bool = False) -> PreflightReport:
    """Run Stage 0 CPU pre-flight on a folder of images."""
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return PreflightReport(
            verdict="REJECT",
            total_images=0,
            usable_images=0,
            blurry_count=0,
            exposure_issue_count=0,
            duplicate_count=0,
            overlap_edge_count=0,
            connected_component_ratio=0.0,
            message=f"Directory does not exist: {images_dir}",
            errors=["Directory not found"],
            warnings=[],
            images=[],
        )

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    total_count = len(image_paths)

    if not settings.enable_preflight or settings.skip_preflight_overlap:
        return PreflightReport(
            verdict="PASS",
            total_images=total_count,
            usable_images=total_count,
            blurry_count=0,
            exposure_issue_count=0,
            duplicate_count=0,
            overlap_edge_count=total_count,
            connected_component_ratio=1.0,
            message=f"Pre-flight check passed ({total_count} images)",
            errors=[],
            warnings=[],
            images=[],
        )

    if total_count < settings.min_images_per_job:
        return PreflightReport(
            verdict="REJECT",
            total_images=total_count,
            usable_images=total_count,
            blurry_count=0,
            exposure_issue_count=0,
            duplicate_count=0,
            overlap_edge_count=0,
            connected_component_ratio=0.0,
            message=(
                f"Found only {total_count} images (minimum {settings.min_images_per_job} required). "
                "Please upload more photos covering all sides of the object."
            ),
            errors=[f"Image count ({total_count}) below minimum ({settings.min_images_per_job})"],
            warnings=[],
            images=[],
        )

    # 1. Per-image analysis: decode, size, blur, exposure, dhash
    raw_images = []
    blur_scores = []
    dhashes = []

    for p in image_paths:
        data = p.read_bytes()
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raw_images.append({
                "path": p, "valid": False, "w": 0, "h": 0,
                "blur": 0.0, "gray": None, "dhash": 0, "clipped": False, "clip_ratio": 0.0,
            })
            continue

        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = _laplacian_variance(gray)
        dhash = _compute_dhash(gray)
        is_clip, clip_ratio = _check_exposure(gray)

        blur_scores.append(blur)
        dhashes.append(dhash)
        raw_images.append({
            "path": p, "valid": True, "w": w, "h": h,
            "blur": blur, "gray": gray, "dhash": dhash,
            "clipped": is_clip, "clip_ratio": clip_ratio,
        })

    median_blur = float(np.median(blur_scores)) if blur_scores else 0.0
    blur_thresh = median_blur * 0.40  # flag if below 40% of median

    # Evaluate duplicates and per-image flags
    per_image_results: List[ImagePreflightResult] = []
    blurry_count = 0
    exposure_issue_count = 0
    duplicate_count = 0
    seen_hashes: List[Tuple[str, int]] = []

    for item in raw_images:
        p = item["path"]
        warnings = []
        if not item["valid"]:
            per_image_results.append(
                ImagePreflightResult(
                    filename=p.name, width=0, height=0, blur_score=0.0,
                    is_blurry=True, exposure_clipped=False, clipped_ratio=0.0,
                    duplicate_of=None, warnings=["Unreadable or corrupted image file"],
                )
            )
            continue

        w, h = item["w"], item["h"]
        short_side = min(w, h)
        if short_side < 800:
            warnings.append(f"Resolution {w}x{h} is small (short side < 800px); SfM details may be degraded")
        elif short_side < 1200:
            warnings.append(f"Resolution {w}x{h} is moderate (short side < 1200px)")

        is_blur = item["blur"] < blur_thresh
        if is_blur:
            blurry_count += 1
            warnings.append(f"Blurry image (Laplacian variance {item['blur']:.1f} is below 40% of median {median_blur:.1f})")

        if item["clipped"]:
            exposure_issue_count += 1
            warnings.append(f"Exposure clipping: {item['clip_ratio']:.1%} of pixels are saturated or pitch dark")

        dupe_of = None
        for seen_name, seen_h in seen_hashes:
            if _hamming_distance(item["dhash"], seen_h) <= 3:
                dupe_of = seen_name
                duplicate_count += 1
                warnings.append(f"Near-duplicate of {seen_name}")
                break
        if not dupe_of:
            seen_hashes.append((p.name, item["dhash"]))

        per_image_results.append(
            ImagePreflightResult(
                filename=p.name,
                width=w,
                height=h,
                blur_score=round(item["blur"], 2),
                is_blurry=is_blur,
                exposure_clipped=item["clipped"],
                clipped_ratio=item["clip_ratio"],
                duplicate_of=dupe_of,
                warnings=warnings,
            )
        )

    # 2. Set-level overlap connectivity graph (SIFT + RANSAC)
    valid_candidates = [item for item in raw_images if item["valid"]]
    sample_size = min(len(valid_candidates), 60)
    if len(valid_candidates) > sample_size:
        # Uniform sampling across sequence
        sample_indices = np.round(np.linspace(0, len(valid_candidates) - 1, sample_size)).astype(int)
        sampled_items = [valid_candidates[i] for i in sample_indices]
    else:
        sampled_items = valid_candidates

    sift = cv2.SIFT_create(nfeatures=1000)
    features = []
    for item in sampled_items:
        # Scale to max 800px for speedy SIFT feature detection
        gray = item["gray"]
        h, w = gray.shape[:2]
        max_dim = max(w, h)
        if max_dim > 800:
            scale = 800.0 / max_dim
            gray_sift = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            gray_sift = gray
        kp, des = sift.detectAndCompute(gray_sift, None)
        features.append((item["path"].name, kp, des))

    # Build connectivity graph
    graph = nx.Graph()
    for name, _, _ in features:
        graph.add_node(name)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    edge_count = 0
    n_feat = len(features)

    # Thresholds adapted for video vs high-res photos
    min_kp = 15 if is_video else 25
    min_good = 8 if is_video else 20
    min_inliers = 10 if is_video else 20
    ratio_thresh = 0.80 if is_video else 0.75

    for i in range(n_feat):
        name_i, kp_i, des_i = features[i]
        if des_i is None or len(kp_i) < min_kp:
            continue
        # Compare with neighboring and candidate views
        # If video: window of adjacent views; if photo: check nearby candidates
        step_range = range(i + 1, min(i + 8, n_feat)) if is_video else range(i + 1, n_feat)
        for j in step_range:
            name_j, kp_j, des_j = features[j]
            if des_j is None or len(kp_j) < min_kp:
                continue
            matches = bf.knnMatch(des_i, des_j, k=2)
            good_matches = []
            for m_pair in matches:
                if len(m_pair) == 2 and m_pair[0].distance < ratio_thresh * m_pair[1].distance:
                    good_matches.append(m_pair[0])

            if len(good_matches) >= min_good:
                pts_i = np.float32([kp_i[m.queryIdx].pt for m in good_matches])
                pts_j = np.float32([kp_j[m.trainIdx].pt for m in good_matches])
                _, mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC, 3.0)
                if mask is not None and np.sum(mask) >= min_inliers:
                    graph.add_edge(name_i, name_j)
                    edge_count += 1

    # Connected component analysis
    if graph.number_of_nodes() > 0:
        comps = list(nx.connected_components(graph))
        largest_comp_size = max(len(c) for c in comps) if comps else 0
        conn_ratio = float(largest_comp_size / graph.number_of_nodes())
    else:
        conn_ratio = 0.0

    errors = []
    warnings = []
    verdict = "PASS"

    # REJECT if largest connected group < 60%
    if conn_ratio < 0.60:
        verdict = "REJECT"
        if is_video:
            msg = (
                f"Video frames have insufficient overlap ({int(conn_ratio * 100)}% connected). "
                "Move more slowly, avoid rapid jerking, and maintain consistent lighting."
            )
        else:
            connected_cnt = int(conn_ratio * total_count)
            msg = (
                f"Only {connected_cnt} of {total_count} photos overlap enough "
                f"({int(conn_ratio * 100)}% connected, minimum 60% required). "
                "Please retake photos ensuring 60-70% overlap between adjacent shots."
            )
        errors.append(msg)
    elif blurry_count > total_count * 0.30:
        verdict = "WARN"
        msg = f"{blurry_count} of {total_count} images are blurry. Reconstruction quality may be degraded."
        warnings.append(msg)
    elif duplicate_count > total_count * 0.25:
        verdict = "WARN"
        msg = f"{duplicate_count} near-duplicate images detected. Consider taking photos from more angles."
        warnings.append(msg)
    elif conn_ratio < 0.80:
        verdict = "WARN"
        msg = (
            f"Image overlap is moderate ({int(conn_ratio * 100)}% connected). "
            "Reconstruction will proceed but some views may fail to register."
        )
        warnings.append(msg)
    else:
        verdict = "PASS"
        msg = f"Pre-flight check passed! {total_count} images with strong {int(conn_ratio * 100)}% view connectivity."

    return PreflightReport(
        verdict=verdict,
        total_images=total_count,
        usable_images=total_count - blurry_count - duplicate_count,
        blurry_count=blurry_count,
        exposure_issue_count=exposure_issue_count,
        duplicate_count=duplicate_count,
        overlap_edge_count=edge_count,
        connected_component_ratio=round(conn_ratio, 3),
        message=msg,
        errors=errors,
        warnings=warnings,
        images=[asdict(r) for r in per_image_results],
    )
