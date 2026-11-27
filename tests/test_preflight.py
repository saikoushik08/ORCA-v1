"""Tests for CPU Pre-flight Image Check (F-32)."""
import cv2
import numpy as np
import pytest
from pathlib import Path
from uuid import uuid4

from fake_supabase import DB
from helpers import make_png


@pytest.fixture(autouse=True)
def force_preflight(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "skip_preflight_overlap", False)


def _create_synthetic_overlap_set(folder: Path, count: int = 10, disjoint: bool = False):
    folder.mkdir(parents=True, exist_ok=True)
    w, h = 800, 600

    for i in range(count):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = (40, 50, 60)

        if disjoint:
            # Random unrelated noise per image = zero matching SIFT features
            np.random.seed(i)
            noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            img = noise
        else:
            # Overlapping pattern: grid of distinct features translating gradually
            shift_x = int(i * 25)
            for gy in range(80, h - 80, 80):
                for gx in range(80, w - 80, 80):
                    px = (gx + shift_x) % (w - 100) + 50
                    cv2.rectangle(img, (px - 10, gy - 10), (px + 10, gy + 10), (200, 150, 20), -1)
                    cv2.circle(img, (px, gy), 5, (20, 220, 240), -1)
                    cv2.putText(img, f"{gx}_{gy}", (px - 15, gy + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imwrite(str(folder / f"img_{i:02d}.png"), img)


def test_preflight_too_few_images(tmp_path):
    from app.services.preflight import run_preflight_check
    folder = tmp_path / "few"
    folder.mkdir()
    for i in range(3):
        (folder / f"{i}.png").write_bytes(make_png(i))

    report = run_preflight_check(folder)
    assert report.verdict == "REJECT"
    assert "below minimum" in report.errors[0]
    assert report.total_images == 3


def test_preflight_blur_detection(tmp_path):
    from app.services.preflight import run_preflight_check
    folder = tmp_path / "blur_test"
    _create_synthetic_overlap_set(folder, count=10)

    # Intentionally blur 4 images
    for i in (2, 3, 4, 5):
        p = folder / f"img_{i:02d}.png"
        img = cv2.imread(str(p))
        blurred = cv2.GaussianBlur(img, (51, 51), 20.0)
        cv2.imwrite(str(p), blurred)

    report = run_preflight_check(folder)
    assert report.blurry_count >= 3
    blurry_flags = [im for im in report.images if im["is_blurry"]]
    assert len(blurry_flags) >= 3


def test_preflight_duplicate_detection(tmp_path):
    from app.services.preflight import run_preflight_check
    folder = tmp_path / "dupe_test"
    _create_synthetic_overlap_set(folder, count=10)

    # Overwrite two images with identical copies
    base_data = (folder / "img_00.png").read_bytes()
    (folder / "img_01.png").write_bytes(base_data)
    (folder / "img_02.png").write_bytes(base_data)

    report = run_preflight_check(folder)
    assert report.duplicate_count >= 2


def test_preflight_disjoint_set_rejected(tmp_path):
    from app.services.preflight import run_preflight_check
    folder = tmp_path / "disjoint"
    _create_synthetic_overlap_set(folder, count=8, disjoint=True)

    report = run_preflight_check(folder)
    assert report.verdict == "REJECT"
    assert report.connected_component_ratio < 0.60
    assert any("overlap enough" in e for e in report.errors)


def test_preflight_overlapping_set_passes(tmp_path):
    from app.services.preflight import run_preflight_check
    folder = tmp_path / "overlap"
    _create_synthetic_overlap_set(folder, count=8, disjoint=False)

    report = run_preflight_check(folder)
    assert report.verdict in ("PASS", "WARN")
    assert report.connected_component_ratio >= 0.60
    assert report.overlap_edge_count > 0


def test_preflight_api_endpoint(client, tmp_path):
    # 1. Create job
    jid = client.post("/jobs/").json()["job_id"]

    # Preflight before upload should be 400
    r_empty = client.post(f"/jobs/{jid}/preflight")
    assert r_empty.status_code == 400

    # 2. Upload images
    files = [("files", (f"im_{i}.png", make_png(i), "image/png")) for i in range(6)]
    client.post(f"/upload/image/{jid}", files=files)

    # 3. Call preflight endpoint
    r_pref = client.post(f"/jobs/{jid}/preflight")
    assert r_pref.status_code == 200
    data = r_pref.json()
    assert "verdict" in data
    assert "connected_component_ratio" in data
    assert data["total_images"] == 6

    # 4. GET preflight report
    r_get = client.get(f"/jobs/{jid}/preflight")
    assert r_get.status_code == 200
    assert r_get.json()["total_images"] == 6
