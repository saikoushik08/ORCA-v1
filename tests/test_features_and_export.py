"""Tests for Multi-Format 3D Export (F-40), Feature Database Reuse (F-35), Re-run (F-44), and Cloud Storage Sync (F-41)."""
import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

o3d = pytest.importorskip("open3d")
trimesh = pytest.importorskip("trimesh")

from app.config import settings
from app.jobs.job_manager import JobManager
from app.jobs.job_repository import JobRepository
from app.services.method1_photogrammetry.mesh_builder import build_mesh
from app.storage.cloud_sync import StorageSync
from fake_supabase import Bucket
from helpers import make_png


def test_multiformat_export_in_mesh_builder(tmp_path):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    sphere.compute_vertex_normals()
    src = tmp_path / "poisson.ply"
    o3d.io.write_triangle_mesh(str(src), sphere)

    out_dir = tmp_path / "out"
    rep = build_mesh(src, tmp_path / "missing_fused.ply", out_dir)

    assert rep["method_used"] == "poisson"
    assert (out_dir / "final_mesh.ply").exists()
    assert (out_dir / "final_mesh.glb").exists()
    assert (out_dir / "final_mesh.obj").exists()
    assert (out_dir / "final_mesh.stl").exists()
    assert set(rep["exported_formats"]) >= {"ply", "glb", "obj", "stl"}


def test_download_by_format_endpoint(client):
    res_create = client.post("/jobs?method=method_1&preset=standard")
    jid = res_create.json()["job_id"]

    # Upload 5 images
    files = [("files", (f"img_{i}.png", make_png(i), "image/png")) for i in range(5)]
    res_up = client.post(f"/upload/image/{jid}", files=files)
    assert res_up.status_code == 200

    JobManager.start_job(jid)
    JobManager.claim_for_processing(jid)

    out = settings.outputs_root / jid
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_mesh.ply").write_bytes(b"ply-data")
    (out / "final_mesh.glb").write_bytes(b"glb-data")
    (out / "final_mesh.obj").write_bytes(b"obj-data")
    (out / "final_mesh.stl").write_bytes(b"stl-data")
    (out / "quality.json").write_text('{"score": 90, "verdict": "PASS"}')

    JobManager.mark_completed(jid, result={"quality": {"verdict": "PASS"}})

    # Check result summary
    res = client.get(f"/jobs/{jid}/result").json()
    assert res["mesh_url"].endswith("file=final_mesh.glb")
    assert "model_urls" in res
    assert "glb" in res["model_urls"]
    assert "ply" in res["model_urls"]
    assert "obj" in res["model_urls"]
    assert "stl" in res["model_urls"]

    # Download by format query param
    res_glb = client.get(f"/jobs/{jid}/result/download?format=glb")
    assert res_glb.status_code == 200
    assert res_glb.content == b"glb-data"

    res_obj = client.get(f"/jobs/{jid}/result/download?format=obj")
    assert res_obj.status_code == 200
    assert res_obj.content == b"obj-data"

    res_stl = client.get(f"/jobs/{jid}/result/download?format=stl")
    assert res_stl.status_code == 200
    assert res_stl.content == b"stl-data"


def test_rerun_job_reuses_images_and_creates_linked_child(client):
    res_create = client.post("/jobs?method=method_1&preset=standard")
    parent_jid = res_create.json()["job_id"]

    # Upload 5 images to parent
    files = [("files", (f"img_{i}.png", make_png(i), "image/png")) for i in range(5)]
    res_up = client.post(f"/upload/image/{parent_jid}", files=files)
    assert res_up.status_code == 200

    # Simulate cached feature database in parent's workspace
    parent_colmap = settings.workspace_root / parent_jid / "colmap"
    parent_colmap.mkdir(parents=True, exist_ok=True)
    (parent_colmap / "database.db").write_bytes(b"fake-colmap-db")
    (parent_colmap / "features_manifest.json").write_text('{"hashes": {}, "params": {}}')

    # Trigger re-run with preset=high
    res_rerun = client.post(f"/jobs/{parent_jid}/rerun?preset=high")
    assert res_rerun.status_code == 201
    child_data = res_rerun.json()

    child_jid = child_data["job_id"]
    assert child_data["parent_job_id"] == parent_jid
    assert child_data["preset"] == "high"
    assert child_data["status"] == "created"

    # Verify images were cloned
    assert JobRepository.count_images(child_jid) == 5
    child_uploads = settings.uploads_root / child_jid / "images"
    assert len(list(child_uploads.glob("*.png"))) == 5

    # Verify database.db and manifest were copied to child
    child_colmap = settings.workspace_root / child_jid / "colmap"
    assert (child_colmap / "database.db").exists()
    assert (child_colmap / "database.db").read_bytes() == b"fake-colmap-db"
    assert (child_colmap / "features_manifest.json").exists()


def test_storage_sync_publish_outputs(tmp_path):
    jid = uuid4()
    out = tmp_path / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    (out / "final_mesh.glb").write_bytes(b"glb-bytes")
    (out / "final_mesh.ply").write_bytes(b"ply-bytes")
    (out / "quality.json").write_text('{"score": 95}')
    (out / "job.log").write_text("All steps completed successfully.")

    uploaded = StorageSync.publish_outputs(jid, out)
    assert "final_mesh.glb" in uploaded
    assert "final_mesh.ply" in uploaded
    assert "quality.json" in uploaded
    assert "job.log" in uploaded

    # Verify records in fake supabase bucket
    assert f"{jid}/final_mesh.glb" in Bucket.uploads
    assert f"{jid}/final_mesh.ply" in Bucket.uploads
    assert f"{jid}/quality.json" in Bucket.uploads
