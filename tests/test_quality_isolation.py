"""Tests for Object Point Cloud Isolation (F-22) and 100-Point Quality Score Matrix (F-30)."""
import json
from pathlib import Path

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")
trimesh = pytest.importorskip("trimesh")

from app.services.method1_photogrammetry.point_cloud_isolation import (
    _estimate_focus_point_from_cameras,
    isolate_object_point_cloud,
)
from app.services.method1_photogrammetry.quality_checker import (
    _compute_fidelity_metrics,
    _edge_stats,
    _triangle_quality,
    check_mesh_quality,
)


def test_estimate_focus_point_from_cameras(tmp_path):
    images_txt = tmp_path / "images.txt"
    # Create synthetic camera optical axes intersecting near (0, 0, 0)
    # Camera at (0, -2, 0) looking along +Y -> R maps world to cam
    # Let's create dummy COLMAP images.txt format
    content = """# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
1 1.0 0.0 0.0 0.0 0.0 0.0 -2.0 1 cam1.jpg
100 200 1
2 0.7071068 0.0 0.7071068 0.0 0.0 0.0 -2.0 1 cam2.jpg
100 200 2
3 0.7071068 0.0 -0.7071068 0.0 0.0 0.0 -2.0 1 cam3.jpg
100 200 3
"""
    images_txt.write_text(content, encoding="utf-8")
    focus = _estimate_focus_point_from_cameras(images_txt)
    assert focus is not None
    assert len(focus) == 3


def test_point_cloud_isolation_filters_table_plane_and_outliers(tmp_path):
    # Create synthetic point cloud:
    # 1. An object (dense sphere centered at (0, 0, 1) with 3000 points)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=20)
    pcd_object = sphere.sample_points_poisson_disk(2500)
    pcd_object.translate((0.0, 0.0, 1.0))

    # 2. A dominant horizontal ground plane at Z = 0 (1500 points)
    x = np.random.uniform(-2, 2, 1500)
    y = np.random.uniform(-2, 2, 1500)
    z = np.zeros(1500) + np.random.normal(0, 0.005, 1500)
    plane_pts = np.column_stack([x, y, z])
    pcd_plane = o3d.geometry.PointCloud()
    pcd_plane.points = o3d.utility.Vector3dVector(plane_pts)

    # 3. Outlier points far away
    outlier_pts = np.array([[10.0, 10.0, 10.0], [-10.0, -10.0, 5.0], [8.0, -8.0, 3.0]])
    pcd_outliers = o3d.geometry.PointCloud()
    pcd_outliers.points = o3d.utility.Vector3dVector(outlier_pts)

    combined = pcd_object + pcd_plane + pcd_outliers
    fused_ply = tmp_path / "fused.ply"
    isolated_ply = tmp_path / "isolated.ply"
    o3d.io.write_point_cloud(str(fused_ply), combined)

    report = isolate_object_point_cloud(
        fused_ply_path=fused_ply,
        output_isolated_ply=isolated_ply,
        sparse_model_dir=None,
        mode="auto",
    )

    assert report["applied"] is True
    assert report["final_points"] < report["initial_points"]
    assert report["plane_points_removed"] > 0
    assert isolated_ply.exists()

    pcd_res = o3d.io.read_point_cloud(str(isolated_ply))
    assert len(pcd_res.points) == report["final_points"]


def test_point_cloud_isolation_off_mode(tmp_path):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=10)
    pcd = sphere.sample_points_poisson_disk(1200)
    fused_ply = tmp_path / "fused.ply"
    isolated_ply = tmp_path / "isolated.ply"
    o3d.io.write_point_cloud(str(fused_ply), pcd)

    report = isolate_object_point_cloud(
        fused_ply_path=fused_ply,
        output_isolated_ply=isolated_ply,
        mode="off",
    )
    assert report["applied"] is False
    assert report["final_points"] == len(pcd.points)


def test_quality_score_100_point_matrix(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    mesh_path = tmp_path / "sphere.ply"
    mesh.export(mesh_path)

    # Create synthetic fused point cloud matching the mesh
    pts, _ = trimesh.sample.sample_surface(mesh, 4000)
    fused_path = tmp_path / "fused.ply"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(fused_path), pcd)

    sparse_meta = {
        "registered_ratio": 0.95,
        "mean_reprojection_error": 0.55,
        "mean_track_length": 4.5,
    }

    report = check_mesh_quality(
        mesh_path=mesh_path,
        fused_points=15000,
        sparse=sparse_meta,
        fused_ply_path=fused_path,
    )

    assert report["score"] >= 85
    assert report["verdict"] == "PASS"
    assert "groups" in report
    groups = report["groups"]
    assert "sparse" in groups
    assert "dense" in groups
    assert "mesh" in groups
    assert "fidelity" in groups
    assert report["watertight"] is True
    assert report["euler_number"] == 2
    assert len(report["checks"]) > 5


def test_fidelity_metrics_computation(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    pts, _ = trimesh.sample.sample_surface(mesh, 2000)
    fused_path = tmp_path / "fused.ply"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(fused_path), pcd)

    comp, acc = _compute_fidelity_metrics(mesh, fused_path)
    assert comp >= 0.9
    assert acc < 0.05


def test_yellow_glare_filter_removes_refraction_and_wall_noise(tmp_path):
    from app.services.method1_photogrammetry.point_cloud_isolation import filter_yellow_and_specular_noise

    # Create synthetic subject points (skin tone: R=0.5, G=0.32, B=0.22)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.4, resolution=15)
    pcd_skin = sphere.sample_points_poisson_disk(1500)
    skin_pts = np.asarray(pcd_skin.points)
    skin_cols = np.tile(np.array([0.52, 0.32, 0.22]), (len(skin_pts), 1))

    # Create synthetic yellow wall / glare noise points (R=0.75, G=0.65, B=0.50)
    # Some floating in front of face (distance 0.02m away)
    glare_pts = np.random.uniform(-0.3, 0.3, (300, 3)) + np.array([0.0, 0.0, 0.6])
    glare_cols = np.tile(np.array([0.76, 0.65, 0.50]), (len(glare_pts), 1))

    combined = o3d.geometry.PointCloud()
    combined.points = o3d.utility.Vector3dVector(np.vstack([skin_pts, glare_pts]))
    combined.colors = o3d.utility.Vector3dVector(np.vstack([skin_cols, glare_cols]))

    focus = np.array([0.0, 0.0, 0.0])
    cleaned, num_deleted, num_recolored = filter_yellow_and_specular_noise(combined, focus)

    assert num_deleted > 0
    assert len(cleaned.points) < len(combined.points)
    # Verify no bright yellow noise remains
    c_cols = np.asarray(cleaned.colors)
    yellow_remaining = (c_cols[:, 0] > 0.65) & (c_cols[:, 1] > 0.55) & (c_cols[:, 2] < 0.6)
    assert np.sum(yellow_remaining) == 0

