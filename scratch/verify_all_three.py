import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import json
import logging
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
import trimesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verifier")

from app.services.method1_photogrammetry.point_cloud_isolation import isolate_object_point_cloud
from app.services.method1_photogrammetry.mesh_builder import build_mesh
from app.services.presets import get_preset

preset = get_preset("standard")

jobs = [
    ("e8c0e70a-a1a5-4a04-870d-7f3036787f6a", "Glasses & red shirt"),
    ("234ed01f-8c1a-47b0-a518-531d941cc448", "Curly hair & white shirt"),
    ("374ffc25-150c-4c77-9493-2cd25cacbdbd", "Blue plaid shirt"),
]

def render_mesh_views(mesh_path: Path, out_front: Path, out_angle: Path):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    v = np.asarray(mesh.vertices)
    c = np.asarray(mesh.vertex_colors)
    f = np.asarray(mesh.triangles)
    if len(v) == 0 or len(f) == 0:
        return

    # Center and PCA/orientation
    center = np.median(v, axis=0)
    v_c = v - center

    # Vertical direction: along bounding box height or largest vertical component
    # We estimate upright by checking spread
    cov = np.cov(v_c.T)
    evals, evecs = np.linalg.eigh(cov)
    # Usually the body vertical axis has large variation or is near Z/Y
    # Check alignment with world axes
    axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    # Use bounding box principal axis
    bbox = v.max(axis=0) - v.min(axis=0)
    vert_axis_idx = int(np.argmax(bbox))
    up_dir = np.zeros(3)
    up_dir[vert_axis_idx] = 1.0

    # Test top vs bottom spread
    h = v_c @ up_dir
    h_low = v_c[h < np.percentile(h, 25)]
    h_high = v_c[h > np.percentile(h, 75)]
    spread_low = np.mean(np.linalg.norm(h_low - (h_low @ up_dir[:, None]) * up_dir, axis=1))
    spread_high = np.mean(np.linalg.norm(h_high - (h_high @ up_dir[:, None]) * up_dir, axis=1))
    if spread_high > spread_low:
        up_dir = -up_dir

    # Ref horizontal
    ref = np.array([1.0, 0.0, 0.0]) if abs(up_dir[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    right_dir = ref - np.dot(ref, up_dir) * up_dir
    right_dir /= np.linalg.norm(right_dir)
    front_dir = np.cross(right_dir, up_dir)

    def draw(yaw_deg=0, pitch_deg=0, zoom=0.85, w=800, h_px=800):
        rad_y = np.radians(yaw_deg)
        rad_p = np.radians(pitch_deg)
        c_y, s_y = np.cos(rad_y), np.sin(rad_y)
        curr_right = c_y * right_dir + s_y * front_dir
        curr_front = -s_y * right_dir + c_y * front_dir
        curr_up = up_dir

        c_p, s_p = np.cos(rad_p), np.sin(rad_p)
        cam_dir = c_p * curr_front + s_p * curr_up
        cam_up = -s_p * curr_front + c_p * curr_up
        cam_right = np.cross(cam_dir, cam_up)
        cam_right /= np.linalg.norm(cam_right)

        x_proj = v_c @ cam_right
        y_proj = -(v_c @ cam_up)
        z_proj = v_c @ cam_dir

        scale = (w * 0.40) * zoom / (np.max(np.abs(x_proj)) + 1e-4)
        u = (x_proj * scale + w / 2).astype(np.int32)
        v_px = (y_proj * scale + h_px / 2).astype(np.int32)

        img = np.zeros((h_px, w, 3), dtype=np.uint8)
        for y in range(h_px):
            frac = y / h_px
            img[y, :] = (int(50 + frac * 70), int(25 + frac * 35), int(25 + frac * 35))

        tri_z = np.mean(z_proj[f], axis=1)
        order = np.argsort(tri_z)

        for tri_idx in order:
            tri = f[tri_idx]
            pts_2d = np.column_stack((u[tri], v_px[tri]))
            if np.any(pts_2d[:, 0] < 0) or np.any(pts_2d[:, 0] >= w) or np.any(pts_2d[:, 1] < 0) or np.any(pts_2d[:, 1] >= h_px):
                continue
            avg_col = np.mean(c[tri], axis=0) * 255.0
            bgr = (int(avg_col[2]), int(avg_col[1]), int(avg_col[0]))
            cv2.fillPoly(img, [pts_2d], bgr)
        return img

    img_front = draw(yaw_deg=0, zoom=0.85)
    img_angle = draw(yaw_deg=35, pitch_deg=-5, zoom=0.85)
    cv2.imwrite(str(out_front), img_front)
    cv2.imwrite(str(out_angle), img_angle)

for jid, desc in jobs:
    logger.info("=== Processing %s (%s) ===", jid, desc)
    job_dir = Path(f"outputs/{jid}")
    fused_raw = job_dir / "fused.ply"
    sparse_model = job_dir / "sparse_model"
    
    out_dir = job_dir / "verified_clean"
    out_dir.mkdir(parents=True, exist_ok=True)
    fused_isolated = out_dir / "fused_isolated.ply"
    
    t0 = time.time()
    iso_report = isolate_object_point_cloud(
        fused_raw,
        fused_isolated,
        sparse_model_dir=sparse_model,
        mode="auto",
        fill_holes=False,
    )
    logger.info("Isolation: %d -> %d points in %.1fs", iso_report["initial_points"], iso_report["final_points"], time.time() - t0)
    
    t1 = time.time()
    mesh_report = build_mesh(
        None,
        fused_isolated,
        out_dir,
        preset=preset,
        surface_method="poisson",
    )
    logger.info("Mesh built: %d faces in %.1fs", mesh_report["faces"], time.time() - t1)
    
    # Also update the primary outputs/ folder so exports match
    for ext in ["ply", "obj", "stl", "glb"]:
        src = out_dir / f"final_mesh.{ext}"
        dst = job_dir / f"final_mesh.{ext}"
        if src.exists():
            import shutil
            shutil.copyfile(src, dst)
            logger.info("Copied clean %s to %s", src.name, dst)
            
    # Render views
    render_mesh_views(out_dir / "final_mesh.ply", out_dir / "render_front.png", out_dir / "render_angle.png")
    logger.info("Done %s", jid)
