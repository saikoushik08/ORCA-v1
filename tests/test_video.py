"""Tests for Video-to-Keyframes Feature (F-47)."""
import io
import math
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest

from fake_supabase import DB


def create_synthetic_video(
    file_path: Path,
    num_frames: int = 40,
    fps: float = 10.0,
    width: int = 320,
    height: int = 240,
    add_blur: bool = True,
    motion: bool = True,
) -> Path:
    """Creates an MP4/AVI synthetic video with orbiting textured circle."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))

    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 5

    for i in range(num_frames):
        # Blank canvas with gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (30, 40, 50)

        # Draw a moving high-contrast circle or textured box
        angle = 2.0 * math.pi * (i / num_frames) if motion else 0.0
        orbit_r = radius if motion else 0
        cx = int(center_x + orbit_r * math.cos(angle))
        cy = int(center_y + orbit_r * math.sin(angle))

        # Distinct textured pattern
        cv2.circle(frame, (cx, cy), radius, (200, 180, 50), -1)
        cv2.rectangle(frame, (cx - 15, cy - 15), (cx + 15, cy + 15), (20, 20, 240), -1)
        cv2.putText(frame, f"F{i}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Intentionally inject severe blur on a few frames
        if add_blur and i in (5, 6, 15, 16):
            frame = cv2.GaussianBlur(frame, (25, 25), 10.0)

        out.write(frame)

    out.release()
    return file_path


def test_probe_video(tmp_path):
    from app.services.video_utils import probe_video
    vpath = tmp_path / "test.mp4"
    create_synthetic_video(vpath, num_frames=30, fps=10.0, width=320, height=240)

    meta = probe_video(vpath)
    assert meta.width == 320
    assert meta.height == 240
    assert meta.fps == 10.0
    assert meta.frame_count == 30
    assert 2.9 <= meta.duration_s <= 3.1
    assert meta.size_bytes > 0


def test_video_duration_exceeded_raises(tmp_path, monkeypatch):
    from app.config import settings
    from app.services.video_utils import VideoRejected, extract_and_select_keyframes

    monkeypatch.setattr(settings, "video_max_duration_s", 2)
    vpath = tmp_path / "long.mp4"
    create_synthetic_video(vpath, num_frames=40, fps=10.0)  # 4 seconds

    with pytest.raises(VideoRejected, match="duration"):
        extract_and_select_keyframes(vpath)


def test_keyframes_blur_and_parallax_filtering(tmp_path):
    from app.services.video_utils import extract_and_select_keyframes
    vpath = tmp_path / "orbit.mp4"
    create_synthetic_video(vpath, num_frames=50, fps=10.0, add_blur=True, motion=True)

    keyframes, report = extract_and_select_keyframes(
        vpath,
        frame_budget=15,
        candidate_fps=10.0,
        min_parallax_ratio=0.03,
        min_blur_ratio=0.6,
    )
    assert len(keyframes) >= 5
    assert len(keyframes) <= 15
    assert report["frames_extracted"] == 50
    assert report["reasons_dropped"]["blurry"] >= 2
    assert report["frames_kept"] == len(keyframes)
    for kf in keyframes:
        assert kf.width == 320
        assert kf.height == 240
        assert len(kf.image_bytes) > 0
        assert len(kf.sha256) == 64


def test_keyframes_budget_thinning(tmp_path):
    from app.services.video_utils import extract_and_select_keyframes
    vpath = tmp_path / "orbit.mp4"
    create_synthetic_video(vpath, num_frames=60, fps=10.0, add_blur=False, motion=True)

    budget = 8
    keyframes, report = extract_and_select_keyframes(
        vpath,
        frame_budget=budget,
        candidate_fps=10.0,
        min_parallax_ratio=0.01,
    )
    assert len(keyframes) == budget
    assert report["reasons_dropped"]["budget_thinning"] > 0


def test_upload_video_api_end_to_end(client, tmp_path):
    from app.config import settings
    # 1. Create job
    res = client.post("/jobs/", params={"method": "method_1", "preset": "draft"})
    assert res.status_code == 200
    jid = res.json()["job_id"]

    # 2. Upload video
    vpath = tmp_path / "clip.mp4"
    create_synthetic_video(vpath, num_frames=40, fps=10.0)

    with open(vpath, "rb") as fh:
        r = client.post(
            f"/upload/video/{jid}",
            files={"file": ("orbit_scan.mp4", fh.read(), "video/mp4")},
        )

    assert r.status_code == 200
    data = r.json()
    assert data["job_id"] == jid
    assert data["input_type"] == "video"
    assert data["source_video"] == "orbit_scan.mp4"
    assert data["frames_kept"] >= 5
    assert data["total_images"] == data["frames_kept"]
    assert data["status"] == "uploading"

    # 3. Check status
    status_res = client.get(f"/jobs/{jid}").json()
    assert status_res["input_type"] == "video"
    assert status_res["images"] == data["frames_kept"]

    # 4. Check frames endpoint
    frames_res = client.get(f"/jobs/{jid}/frames")
    assert frames_res.status_code == 200
    fdata = frames_res.json()
    assert fdata["source_video"] == "orbit_scan.mp4"
    assert fdata["frames_kept"] == data["frames_kept"]
    assert "reasons_dropped" in fdata

    # Verify images saved on disk
    stored_images = list((settings.uploads_root / jid / "images").glob("*.jpg"))
    assert len(stored_images) == data["frames_kept"]


def test_upload_video_rejects_unsupported_format(client):
    res = client.post("/jobs/")
    jid = res.json()["job_id"]

    r = client.post(
        f"/upload/video/{jid}",
        files={"file": ("test.exe", b"binary content", "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "not supported" in r.text


def test_upload_video_rejects_empty_file(client):
    res = client.post("/jobs/")
    jid = res.json()["job_id"]

    r = client.post(
        f"/upload/video/{jid}",
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    assert r.status_code == 400
    assert "empty" in r.text


def test_reject_mixed_photo_and_video(client, tmp_path):
    from helpers import make_png

    # Create job and upload photos first
    res = client.post("/jobs/")
    jid = res.json()["job_id"]
    client.post(
        f"/upload/image/{jid}",
        files=[("files", ("img1.png", make_png(1), "image/png"))],
    )

    # Attempting to upload video to photo job should be rejected
    vpath = tmp_path / "clip.mp4"
    create_synthetic_video(vpath, num_frames=30)
    with open(vpath, "rb") as fh:
        r_vid = client.post(
            f"/upload/video/{jid}",
            files={"file": ("clip.mp4", fh.read(), "video/mp4")},
        )
    assert r_vid.status_code == 400
    assert "Mixed photos and video" in r_vid.text

    # Now create a fresh job and upload video
    res2 = client.post("/jobs/")
    jid2 = res2.json()["job_id"]
    with open(vpath, "rb") as fh:
        client.post(
            f"/upload/video/{jid2}",
            files={"file": ("clip.mp4", fh.read(), "video/mp4")},
        )

    # Attempting to upload photos to video job should be rejected
    r_img = client.post(
        f"/upload/image/{jid2}",
        files=[("files", ("img2.png", make_png(2), "image/png"))],
    )
    assert r_img.status_code == 400
    assert "Mixed photos and video" in r_img.text
