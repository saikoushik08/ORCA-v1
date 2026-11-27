"""Mesh clean-up, Poisson -> Ball-Pivoting fallback and quality verdicts.
Need the real Open3D and trimesh; skipped when they are not installed."""
import json

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")
trimesh = pytest.importorskip("trimesh")


def _sphere_with_blobs(path, resolution=40):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    blob = o3d.geometry.TriangleMesh.create_sphere(radius=0.05, resolution=6)
    blob.translate((3.0, 0.0, 0.0))
    mesh = sphere + blob
    o3d.io.write_triangle_mesh(str(path), mesh)
    return len(sphere.triangles)


def test_cleanup_removes_floating_blob_and_reports_every_step(tmp_path):
    from app.services.method1_photogrammetry.mesh_builder import build_mesh
    src = tmp_path / "poisson.ply"
    sphere_faces = _sphere_with_blobs(src)
    rep = build_mesh(src, tmp_path / "missing_fused.ply", tmp_path / "out")
    assert rep["method_used"] == "poisson" and not rep["fallback_used"]
    assert rep["initial_components"] == 2
    assert abs(rep["faces"] - sphere_faces) < 0.05 * sphere_faces      # blob gone
    names = {s["step"]: s for s in rep["steps"]}
    assert names["keep_largest_component"]["applied"] is True
    assert (tmp_path / "out" / "final_mesh.ply").exists()
    assert json.loads((tmp_path / "out" / "mesh_report.json").read_text())["faces"] == rep["faces"]


def test_large_mesh_is_decimated_not_skipped(tmp_path, monkeypatch):
    from app.config import settings
    from app.services.method1_photogrammetry.mesh_builder import build_mesh
    src = tmp_path / "poisson.ply"
    _sphere_with_blobs(src, resolution=60)
    monkeypatch.setattr(settings, "mesh_max_faces_before_cleanup", 5000)
    monkeypatch.setattr(settings, "mesh_target_faces", 3000)
    rep = build_mesh(src, tmp_path / "none.ply", tmp_path / "out")
    assert rep["faces"] <= 3100
    assert {s["step"] for s in rep["steps"] if s["applied"]} >= {"pre_decimation", "target_decimation"}


def test_ball_pivoting_fallback_when_poisson_is_missing(tmp_path):
    from app.services.method1_photogrammetry.mesh_builder import build_mesh
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=30)
    sphere.compute_vertex_normals()
    pcd = sphere.sample_points_poisson_disk(6000)
    fused = tmp_path / "fused.ply"
    o3d.io.write_point_cloud(str(fused), pcd)
    rep = build_mesh(None, fused, tmp_path / "out")          # no Poisson mesh at all
    assert rep["method_used"] == "ball_pivoting" and rep["fallback_used"] is True
    assert rep["faces"] > 500


def test_no_mesh_at_all_raises_a_clear_error(tmp_path):
    from app.jobs.errors import PipelineError
    from app.services.method1_photogrammetry.mesh_builder import build_mesh
    with pytest.raises(PipelineError, match="usable mesh"):
        build_mesh(None, tmp_path / "none.ply", tmp_path / "out")


def _export(mesh, path):
    mesh.export(path)
    return path


def test_closed_sphere_passes_quality(tmp_path):
    from app.services.method1_photogrammetry.quality_checker import check_mesh_quality
    p = _export(trimesh.creation.icosphere(subdivisions=4), tmp_path / "s.ply")
    q = check_mesh_quality(p)
    assert q["verdict"] == "PASS" and q["watertight"] and q["euler_number"] == 2
    assert q["components"] == 1 and q["boundary_edge_ratio"] == 0


def test_open_mesh_gets_warning_not_failure(tmp_path):
    from app.services.method1_photogrammetry.quality_checker import check_mesh_quality
    m = trimesh.creation.icosphere(subdivisions=4)
    m.update_faces(np.arange(len(m.faces)) % 20 != 0)        # punch holes
    q = check_mesh_quality(_export(m, tmp_path / "o.ply"))
    assert q["verdict"] == "WARN" and not q["watertight"] and q["boundary_edge_ratio"] > 0


def test_tiny_mesh_fails_quality_without_raising(tmp_path):
    from app.services.method1_photogrammetry.quality_checker import check_mesh_quality
    q = check_mesh_quality(_export(trimesh.creation.box(), tmp_path / "b.ply"))
    assert q["verdict"] == "FAIL" and any("faces" in e for e in q["errors"])


def test_fallback_and_low_registration_are_flagged_as_warnings(tmp_path):
    from app.services.method1_photogrammetry.quality_checker import check_mesh_quality
    p = _export(trimesh.creation.icosphere(subdivisions=4), tmp_path / "s.ply")
    q = check_mesh_quality(p, sparse={"registered_ratio": 0.4, "mean_reprojection_error": 0.6},
                           mesh_report={"fallback_used": True, "method_used": "ball_pivoting"})
    assert q["verdict"] == "WARN"
    assert any("Ball-Pivoting" in w for w in q["warnings"])
    assert any("registered" in w for w in q["warnings"])
