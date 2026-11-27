"""Tests for Quality Presets (F-31)."""
from uuid import uuid4
from app.services.presets import PRESETS, Preset, get_preset


def test_presets_table_values():
    draft = get_preset("draft")
    std = get_preset("standard")
    high = get_preset("high")

    assert draft.sfm_max_image_size == 1600
    assert std.sfm_max_image_size == 2400
    assert high.sfm_max_image_size == 3200

    assert draft.max_sift_features == 4096
    assert std.max_sift_features == 8192
    assert high.max_sift_features == 8192

    assert draft.video_frame_budget == 80
    assert std.video_frame_budget == 120
    assert high.video_frame_budget == 200

    assert draft.patchmatch_num_iterations == 3
    assert std.patchmatch_num_iterations == 5
    assert high.patchmatch_num_iterations == 7

    assert draft.patchmatch_geom_consistency is False
    assert std.patchmatch_geom_consistency is True
    assert high.patchmatch_geom_consistency is True

    assert draft.poisson_depth == 9
    assert std.poisson_depth == 11
    assert high.poisson_depth == 12

    assert draft.mesh_target_faces == 80_000
    assert std.mesh_target_faces == 300_000
    assert high.mesh_target_faces == 800_000


def test_case_insensitive_and_fallback():
    assert get_preset("DRAFT").name == "draft"
    assert get_preset("Standard").name == "standard"
    assert get_preset("HIGH").name == "high"
    assert get_preset("unknown_val").name == "standard"
    assert get_preset(None).name == "standard"


def test_create_job_with_preset(client):
    r_draft = client.post("/jobs/", params={"preset": "draft"})
    assert r_draft.status_code == 200
    assert r_draft.json()["preset"] == "draft"

    r_high = client.post("/jobs/", params={"preset": "high"})
    assert r_high.status_code == 200
    assert r_high.json()["preset"] == "high"

    # Status route reflects preset
    jid = r_high.json()["job_id"]
    status_data = client.get(f"/jobs/{jid}").json()
    assert status_data["preset"] == "high"
