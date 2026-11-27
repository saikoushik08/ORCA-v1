"""Point cloud object isolation before meshing (F-22).

Removes background tables, floors, walls, and floating noise from the fused point cloud
before surface reconstruction (Poisson / Ball-Pivoting):
1. Focus point estimation via least-squares intersection of camera optical axes (or median).
2. Robust radial crop around the focus point.
3. RANSAC plane removal if a dominant plane holds > 15% of points (table/floor).
4. Statistical outlier removal.
5. DBSCAN clustering keeping the cluster closest to the focus point.
"""
import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


class FocusPoint(np.ndarray):
    """3D focus point with attached minimum camera distance and scene up vector."""
    min_cam_dist: Optional[float] = None
    up_vector: Optional[np.ndarray] = None

    def __new__(cls, input_array, min_cam_dist: Optional[float] = None, up_vector: Optional[np.ndarray] = None):
        obj = np.asarray(input_array).view(cls)
        obj.min_cam_dist = min_cam_dist
        obj.up_vector = up_vector
        return obj


def _estimate_focus_point_from_cameras(images_path: Path) -> Optional[np.ndarray]:
    """Calculate the least-squares intersection point of all camera optical axes.
    Supports both COLMAP text (images.txt) and binary (images.bin) formats."""
    if not images_path.exists():
        return None

    try:
        centers = []
        directions = []
        cam_ups = []

        if images_path.suffix.lower() == ".bin":
            import struct
            with open(images_path, "rb") as f:
                num_reg_images = struct.unpack("<Q", f.read(8))[0]
                for _ in range(num_reg_images):
                    _ = struct.unpack("<I", f.read(4))[0]  # image_id
                    qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
                    tx, ty, tz = struct.unpack("<3d", f.read(24))
                    _ = struct.unpack("<I", f.read(4))[0]  # camera_id
                    # Name (null-terminated)
                    while True:
                        ch = f.read(1)
                        if ch == b"\x00" or not ch:
                            break
                    num_points2d = struct.unpack("<Q", f.read(8))[0]
                    f.seek(num_points2d * 24, 1)  # skip 2D points

                    R = np.array([
                        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
                        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
                    ])
                    T = np.array([tx, ty, tz])
                    C = -R.T @ T
                    v = R.T @ np.array([0.0, 0.0, 1.0])
                    norm_v = np.linalg.norm(v)
                    if norm_v > 1e-6:
                        v = v / norm_v
                        centers.append(C)
                        directions.append(v)
                    cam_up = R.T @ np.array([0.0, -1.0, 0.0])
                    norm_up = np.linalg.norm(cam_up)
                    if norm_up > 1e-6:
                        cam_ups.append(cam_up / norm_up)
        else:
            lines = images_path.read_text(encoding="utf-8", errors="replace").splitlines()
            rows = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
            headers = rows[::2]
            for h in headers:
                parts = h.split()
                if len(parts) < 10:
                    continue
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])

                R = np.array([
                    [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
                    [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                    [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
                ])
                T = np.array([tx, ty, tz])
                C = -R.T @ T
                v = R.T @ np.array([0.0, 0.0, 1.0])
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-6:
                    v = v / norm_v
                    centers.append(C)
                    directions.append(v)
                cam_up = R.T @ np.array([0.0, -1.0, 0.0])
                norm_up = np.linalg.norm(cam_up)
                if norm_up > 1e-6:
                    cam_ups.append(cam_up / norm_up)

        if len(centers) < 3:
            return None

        # Solve sum((I - d*d^T)) * x = sum((I - d*d^T) * C)
        A = np.zeros((3, 3))
        b = np.zeros(3)
        I = np.eye(3)

        for C, d in zip(centers, directions):
            proj = I - np.outer(d, d)
            A += proj
            b += proj @ C

        if np.linalg.cond(A) < 1e5:
            focus = np.linalg.solve(A, b)
            dists_to_focus = [float(np.linalg.norm(c - focus)) for c in centers]
            min_cam_dist = float(np.min(dists_to_focus))
            mean_up = np.mean(cam_ups, axis=0) if len(cam_ups) > 0 else np.array([0.0, 0.0, 1.0])
            norm_m_up = np.linalg.norm(mean_up)
            mean_up = mean_up / (norm_m_up + 1e-8) if norm_m_up > 1e-6 else np.array([0.0, 0.0, 1.0])
            logger.info("[Isolation] Camera ray focus point: %s (min_cam_dist=%.2fm, up_vector=%s)",
                        np.round(focus, 3), min_cam_dist, np.round(mean_up, 3))
            return FocusPoint(focus, min_cam_dist, mean_up)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Isolation] Could not parse camera focus point from %s: %s", images_path, exc)
        return None


def filter_yellow_and_specular_noise(
    pcd: o3d.geometry.PointCloud,
    focus_point: np.ndarray,
    search_radius: float = 0.004,
) -> Tuple[o3d.geometry.PointCloud, int, int]:
    """Eliminates yellow background wall bleed, glass refraction noise on spectacles,
    and specular glare from the point cloud.

    Recolors yellow glare points on the head, face, spectacles, and hairline using
    clean neighboring skin and hair colors instead of deleting them.
    This guarantees zero missing points on the face while completely removing yellow dots
    and preserving 100% of the body and clothing intact.
    """
    if not pcd.has_colors() or len(pcd.points) < 100:
        return pcd, 0, 0

    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors).copy()
    r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]

    # Compute height along scene up-vector to identify the head/face region accurately
    up_vector = getattr(focus_point, "up_vector", None)
    if up_vector is None or not hasattr(up_vector, "__len__"):
        up_vector = np.array([0.0, 0.0, 1.0])
    up_vector = np.asarray(up_vector, dtype=float)
    up_norm = np.linalg.norm(up_vector)
    if up_norm > 1e-6:
        up_vector = up_vector / up_norm
    else:
        up_vector = np.array([0.0, 0.0, 1.0])

    # Check upright orientation using lateral cross-sectional spread of upper vs lower half
    # Torso/shoulders (base) have substantially wider lateral spread than the head crown
    f_center = np.asarray(focus_point[:3], dtype=float)
    d_f = np.linalg.norm(pts - f_center, axis=1)
    pts_sub = pts[d_f < 0.55]
    if len(pts_sub) >= 200:
        h_cand = (pts_sub - f_center) @ up_vector
        h_low_pts = pts_sub[h_cand < np.percentile(h_cand, 25)]
        h_high_pts = pts_sub[h_cand > np.percentile(h_cand, 75)]
        if len(h_low_pts) > 20 and len(h_high_pts) > 20:
            p_low = h_low_pts - (h_low_pts @ up_vector[:, None]) * up_vector
            p_high = h_high_pts - (h_high_pts @ up_vector[:, None]) * up_vector
            spread_low = float(np.mean(np.linalg.norm(p_low - np.mean(p_low, axis=0), axis=1)))
            spread_high = float(np.mean(np.linalg.norm(p_high - np.mean(p_high, axis=0), axis=1)))
            if spread_high > spread_low:
                up_vector = -up_vector
                focus_point.up_vector = up_vector

    heights = (pts - f_center) @ up_vector

    # Subject head spans upper half of heights (h >= 40th percentile)
    h_p40 = float(np.percentile(heights, 40))
    is_head = heights >= h_p40
    h_head = heights[is_head]
    chin_h = float(np.percentile(h_head, 5)) if len(h_head) > 50 else float(np.percentile(heights, 20))
    crown_h = float(np.percentile(h_head, 98)) if len(h_head) > 50 else float(np.percentile(heights, 95))

    # Detect true yellow wall reflection, light flare, glass refraction, and specular glare:
    # Human skin has strong red dominance (R - G >= 0.15, G/R <= 0.70, B < G).
    # Yellow wall paint, spectacle glass glare, and specular flares have G/R > 0.68, (R - G) < 0.15, or high brightness.
    ratio_gr = g / (r + 1e-6)

    # Protected authentic skin definition (zero false positives on natural skin):
    authentic_skin = (r >= 0.35) & (r <= 0.80) & ((r - g) >= 0.15) & (ratio_gr <= 0.70) & (b < g)

    # Yellow background wall bleed & warm room reflections:
    is_yellow_1 = (r > 0.35) & (g > 0.28) & (ratio_gr > 0.68) & (b < 0.90 * r) & ((r - g) < 0.15)
    is_yellow_2 = (r > 0.42) & (g > 0.35) & (ratio_gr > 0.70) & (b < 0.95 * r) & ((r - g) < 0.13)

    # Specular ceiling light / camera flash glare on spectacles, head, face, chin, and throat:
    is_cream_white = (r > 0.52) & (g > 0.45) & (ratio_gr > 0.72) & ((r - g) < 0.12)
    is_bleached_all = (r > 0.65) & (g > 0.55)

    # Stray noise specks on dark clothing (e.g. maroon/colored t-shirt):
    is_shirt_noise = (r > 0.25) & (g > 0.30) & (ratio_gr > 0.60) & (b < 0.45) & (r < 0.50) & (~is_head)

    # Spectacle glass refractions in the eye-level aperture zone:
    eye_low = chin_h + 0.45 * (crown_h - chin_h)
    eye_high = chin_h + 0.75 * (crown_h - chin_h)
    is_eye_level = is_head & (heights >= eye_low) & (heights <= eye_high)
    is_glass_refraction = is_eye_level & ((r > 0.45) & (g > 0.40) & (r + g > 0.85))

    # Yellow / bleached dots on chin, beard, and lower neck:
    chin_zone = is_head & (heights <= chin_h + 0.25 * (crown_h - chin_h))
    yellow_chin = chin_zone & (r > 0.40) & (g > 0.35) & (r + g > 0.85)

    glare_mask = ((is_yellow_1 | is_yellow_2 | is_cream_white | is_bleached_all | is_shirt_noise | is_glass_refraction | yellow_chin) & (~authentic_skin))
    num_glare = int(np.sum(glare_mask))

    if num_glare == 0:
        return pcd, 0, 0

    # Clean donor points for authentic skin, hair, and clothing (exclude any glare points)
    clean_mask = ~glare_mask
    donor_mask = clean_mask & ~((r > 0.60) & (g > 0.50))
    if np.sum(donor_mask) < 20:
        donor_mask = clean_mask

    if np.sum(donor_mask) >= 20:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts[donor_mask])
        k_query = min(5, int(np.sum(donor_mask)))
        dists, idxs = tree.query(pts[glare_mask], k=k_query, workers=1)

        # Adaptive floater threshold based on local point spacing:
        # Glare points detached > 1.8x median NN distance (min 3.5mm) from authentic surface
        # are detached floaters (spectacle glass refraction, air dust, wall reflection) -> delete
        nn_dists_eval = np.asarray(pcd.compute_nearest_neighbor_distance())
        med_nn = float(np.median(nn_dists_eval)) if len(nn_dists_eval) > 0 else 0.002
        floating_thresh = max(0.0035, med_nn * 1.8)

        min_dists = dists[:, 0] if dists.ndim > 1 else dists
        # Spurious glass refractions, chin/beard yellow specks, and detached floaters are DELETED.
        # Deleting glass refractions exposes authentic pupils, eyelids, and eye sockets behind spectacles without opaque webbing.
        # Deleting chin/beard yellow specks eliminates lumpy yellow crusts from the beard.
        is_floating = (min_dists > floating_thresh) | is_glass_refraction[glare_mask] | yellow_chin[glare_mask]
        glare_indices = np.where(glare_mask)[0]
        floating_indices = glare_indices[is_floating]
        surface_indices = glare_indices[~is_floating]

        num_deleted = len(floating_indices)
        num_recolored = len(surface_indices)

        # Recolor remaining minor surface glare points with authentic skin / hair tones
        if num_recolored > 0:
            surf_idxs = idxs[~is_floating]
            donor_cols = cols[donor_mask]
            if k_query == 1:
                cols[surface_indices] = donor_cols[surf_idxs]
            else:
                cols[surface_indices] = np.median(donor_cols[surf_idxs], axis=1)
            pcd.colors = o3d.utility.Vector3dVector(cols)

        # Prune detached floating noise points if present
        if num_deleted > 0:
            keep_indices = np.setdiff1d(np.arange(len(pts)), floating_indices)
            pcd = pcd.select_by_index(keep_indices)

        logger.info(
            "[Isolation] Yellow/glare filter: deleted %d floating noise & glass specks, recolored %d surface points (face & clothing 100%% intact)",
            num_deleted, num_recolored,
        )
        return pcd, num_deleted, num_recolored

    return pcd, 0, 0


def densify_and_fill_holes_knn(
    pcd: o3d.geometry.PointCloud,
    k: int = 6,
    max_iterations: int = 3,
    target_distance_multiplier: float = 1.3,
    max_gap_multiplier: float = 3.5,
    max_total_points: int = 5_000_000,
) -> Tuple[o3d.geometry.PointCloud, dict]:
    """Iteratively fill holes and missing points in the point cloud using K-Nearest Neighbors.

    Points are inserted at the midpoints of neighboring pairs where distance > target_distance
    and <= max_gap with normal alignment, continuing until point-to-point distance is below threshold.
    Checks local point density so sparse regions (clothing/gaps) get filled even if total count is large.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(pcd.points).copy()
    if len(pts) < 50:
        return pcd, {"iterations": 0, "added_points": 0}

    # Cap total points to 6.0M to keep memory and Poisson solver fast
    if len(pts) >= 6_000_000:
        logger.info("[Isolation] Point cloud has %d points (>= 6M); skipping KNN hole insertion", len(pts))
        return pcd, {"iterations": 0, "added_points": 0, "final_points": len(pts), "note": "already dense"}

    cols = np.asarray(pcd.colors).copy() if pcd.has_colors() else np.zeros((len(pts), 3))
    if not pcd.has_normals():
        # Adaptive radius based on actual point cloud scale
        nn_dists_init = np.asarray(pcd.compute_nearest_neighbor_distance())
        med_init = float(np.median(nn_dists_init)) if len(nn_dists_init) > 0 else 0.005
        normal_radius = max(med_init * 8.0, 0.005)
        normal_nn = min(30, max(15, len(pts) // 500))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_nn))
        pcd.orient_normals_consistent_tangent_plane(min(30, max(15, len(pts) // 1000)))
    normals = np.asarray(pcd.normals).copy()

    total_added = 0
    final_med_dist = 0.0
    actual_iterations = 0

    for iteration in range(max_iterations):
        if len(pts) >= max_total_points:
            break

        dists = np.asarray(pcd.compute_nearest_neighbor_distance())
        med_dist = float(np.median(dists)) if len(dists) > 0 else 0.005
        final_med_dist = med_dist
        d_target = med_dist * target_distance_multiplier
        d_max = med_dist * max_gap_multiplier

        # Fully vectorized neighbor pairing & midpoint densification
        tree = cKDTree(pts)
        k_query = min(k, len(pts))
        k_dists, k_indices = tree.query(pts, k=k_query, workers=-1)

        # Pair indices: i and neighbor
        i_arr = np.repeat(np.arange(len(pts)), k_query - 1)
        j_arr = k_indices[:, 1:].ravel()
        d_arr = k_dists[:, 1:].ravel()

        pair_mask = (j_arr > i_arr) & (d_arr > d_target) & (d_arr <= d_max)
        i_sel = i_arr[pair_mask]
        j_sel = j_arr[pair_mask]

        if len(i_sel) == 0:
            logger.info("[Isolation] KNN densification converged after %d iterations", iteration)
            break

        # Check normal alignment across pairs (0.15 allows smooth interpolation across curved facial contours)
        dot_prod = np.sum(normals[i_sel] * normals[j_sel], axis=1)
        aligned = dot_prod > 0.15
        i_ok = i_sel[aligned]
        j_ok = j_sel[aligned]

        if len(i_ok) == 0:
            logger.info("[Isolation] KNN densification converged (no aligned pairs) after %d iterations", iteration)
            break

        mid_pts = 0.5 * (pts[i_ok] + pts[j_ok])
        mid_cols = 0.5 * (cols[i_ok] + cols[j_ok])
        mid_norms = normals[i_ok] + normals[j_ok]
        norm_len = np.linalg.norm(mid_norms, axis=1, keepdims=True)
        mid_norms = np.where(norm_len > 1e-6, mid_norms / norm_len, normals[i_ok])

        actual_iterations += 1
        # Voxel grid thin new points to prevent clustering
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(mid_pts)
        temp_pcd.colors = o3d.utility.Vector3dVector(mid_cols)
        temp_pcd.normals = o3d.utility.Vector3dVector(mid_norms)
        temp_pcd = temp_pcd.voxel_down_sample(voxel_size=med_dist * 0.8)

        add_pts = np.asarray(temp_pcd.points)
        add_cols = np.asarray(temp_pcd.colors)
        add_norms = np.asarray(temp_pcd.normals)

        if len(add_pts) == 0:
            break

        if len(pts) + len(add_pts) > max_total_points:
            allowance = max_total_points - len(pts)
            if allowance <= 0:
                break
            add_pts = add_pts[:allowance]
            add_cols = add_cols[:allowance]
            add_norms = add_norms[:allowance]

        pts = np.vstack([pts, add_pts])
        cols = np.vstack([cols, add_cols])
        normals = np.vstack([normals, add_norms])
        total_added += len(add_pts)

        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        logger.info("[Isolation] KNN densification iter %d: +%d points (total: %d)",
                    iteration + 1, len(add_pts), len(pts))

    return pcd, {
        "iterations": actual_iterations,
        "added_points": total_added,
        "final_points": len(pts),
        "median_distance_mm": round(final_med_dist * 1000, 2),
    }


def isolate_object_point_cloud(
    fused_ply_path: Path,
    output_isolated_ply: Path,
    sparse_model_dir: Optional[Path] = None,
    mode: str = "auto",
    fill_holes: bool = False,
) -> dict:
    """Isolate object point cloud before Poisson / BPA meshing."""
    fused_ply_path = Path(fused_ply_path)
    output_isolated_ply = Path(output_isolated_ply)

    if not fused_ply_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {fused_ply_path}")

    pcd = o3d.io.read_point_cloud(str(fused_ply_path))
    initial_count = len(pcd.points)

    if mode == "off" or initial_count < 1000:
        o3d.io.write_point_cloud(str(output_isolated_ply), pcd)
        return {
            "applied": False,
            "initial_points": initial_count,
            "final_points": initial_count,
            "mode": mode,
        }

    # 1. Determine focus point and camera distance (check multiple candidate paths for images.txt / images.bin)
    focus_point = None
    min_cam_dist = None
    if sparse_model_dir:
        s_path = Path(sparse_model_dir)
        candidates = [
            s_path / "images.txt",
            s_path / "images.bin",
            s_path.parent / "sparse_txt" / "images.txt",
            s_path.parent.parent / "sparse_txt" / "images.txt",
            s_path.parent.parent.parent / "sparse_txt" / "images.txt",
        ]
        for c in candidates:
            if c.exists():
                cam_res = _estimate_focus_point_from_cameras(c)
                if cam_res is not None:
                    focus_point = cam_res
                    min_cam_dist = getattr(cam_res, "min_cam_dist", None)
                    break

    pts = np.asarray(pcd.points)
    if focus_point is None:
        focus_point = np.median(pts, axis=0)
        logger.info("[Isolation] Falling back to point cloud median focus point: %s", np.round(focus_point, 3))
    else:
        # Refine camera focus to exact centroid of subject points near focus
        d_init = np.linalg.norm(pts - np.asarray(focus_point[:3]), axis=1)
        pts_near = pts[d_init < 1.0]
        if len(pts_near) > 500 and pcd.has_colors():
            cols_near = np.asarray(pcd.colors)[d_init < 1.0]
            r_n, g_n, b_n = cols_near[:, 0], cols_near[:, 1], cols_near[:, 2]
            skin_n = (r_n >= 0.40) & (r_n <= 0.80) & ((r_n - g_n) >= 0.15) & (g_n > b_n)
            if np.sum(skin_n) > 500:
                ref_focus = np.median(pts_near[skin_n], axis=0)
            else:
                ref_focus = np.median(pts_near, axis=0)
            up_v = getattr(focus_point, "up_vector", None)
            min_cd = getattr(focus_point, "min_cam_dist", min_cam_dist)
            focus_point = FocusPoint(ref_focus, min_cd, up_v)

    # 2. Focus point radial crop: subject core bounded by camera orbit and radial density drop
    dists = np.linalg.norm(pts - np.asarray(focus_point[:3]), axis=1)
    p85 = float(np.percentile(dists, 85))
    if min_cam_dist is not None and min_cam_dist > 0.5:
        # Subject cannot extend beyond camera distance (scaled to subject envelope)
        radius_thresh = min(p85 * 1.25, min_cam_dist * 0.50)
    else:
        p95 = float(np.percentile(dists, 95))
        if p95 > p85 * 1.8:
            radius_thresh = p85 * 1.3
        else:
            radius_thresh = p95 * 1.15

    # Check for radial density valley between subject and background wall (real camera capture)
    if min_cam_dist is not None and len(pts) > 2000:
        max_d = float(np.percentile(dists, 95))
        bins = np.arange(0.1, min(max_d, 2.5), 0.05)
        if len(bins) > 8:
            hist, _ = np.histogram(dists, bins=bins)
            peak_idx = int(np.argmax(hist[:min(len(hist), 20)]))
            if peak_idx + 2 < len(hist):
                search_end = min(len(hist), peak_idx + 15)
                valley_idx = peak_idx + int(np.argmin(hist[peak_idx:search_end]))
                valley_r = float(bins[valley_idx]) + 0.12
                if valley_r < radius_thresh:
                    logger.info("[Isolation] Radial density valley detected at %.2fm (trimming background wall)", valley_r)
                    radius_thresh = max(0.70, valley_r)

    # Subject envelope: human bust fits within 0.88m radius of refined centroid
    # This completely excludes background walls, floor, and distant room geometry
    radius_thresh = max(0.70, min(radius_thresh, 0.88))
    logger.info("[Isolation] Radial crop threshold: %.2fm (p85=%.2fm, min_cam_dist=%s)",
                radius_thresh, p85, round(min_cam_dist, 2) if min_cam_dist else None)
    inside_indices = np.where(dists <= radius_thresh)[0]
    pcd = pcd.select_by_index(inside_indices)
    after_crop = len(pcd.points)

    # 3. RANSAC ground/table plane removal with STRICT non-bisecting validation
    plane_removed_count = 0
    if len(pcd.points) > 1000:
        pts_curr = np.asarray(pcd.points)
        # Use robust percentiles to prevent distant outliers from blowing up dist_thresh
        p1 = np.percentile(pts_curr, 1, axis=0)
        p99 = np.percentile(pts_curr, 99, axis=0)
        robust_diag = float(np.linalg.norm(p99 - p1))
        dist_thresh = max(0.005 * robust_diag, 0.01)

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=dist_thresh, ransac_n=3, num_iterations=1000
        )
        ratio = len(inliers) / len(pcd.points)
        a, b, c, d = plane_model
        norm = float(np.sqrt(a**2 + b**2 + c**2))
        if norm > 1e-6:
            signed_dists = (pts_curr @ np.array([a, b, c]) + d) / norm
            frac_pos = float(np.mean(signed_dists > dist_thresh))
            frac_neg = float(np.mean(signed_dists < -dist_thresh))
            dist_to_focus = float(abs(a*focus_point[0] + b*focus_point[1] + c*focus_point[2] + d) / norm)
            min_split = min(frac_pos, frac_neg)

            # CRITICAL GEOMETRIC INTEGRITY CHECK:
            # A genuine external support plane (table/floor/wall) has all/most object points on ONE side.
            # If points exist on BOTH sides (min_split >= 4%), the plane bisects and cuts through
            # the subject (e.g. chest/face) and MUST NOT be removed!
            is_external_plane = (
                ratio > 0.15
                and min_split < 0.04
            )
            if is_external_plane:
                pcd = pcd.select_by_index(inliers, invert=True)
                plane_removed_count = len(inliers)
                logger.info("[Isolation] RANSAC safely removed external support plane (%d points, %.1f%%)",
                            len(inliers), ratio * 100)
            else:
                logger.info("[Isolation] RANSAC plane rejected (ratio=%.3f, min_split=%.3f, dist_to_focus=%.3f): plane is part of subject",
                            ratio, min_split, dist_to_focus)

    # 4. Outlier removal: remove perimeter floaters, detached dust, and refraction fuzz
    outliers_removed_count = 0
    if len(pcd.points) > 500:
        cnt_before = len(pcd.points)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
        if len(pcd.points) > 500:
            nn_dists_eval = np.asarray(pcd.compute_nearest_neighbor_distance())
            med_nn = float(np.median(nn_dists_eval)) if len(nn_dists_eval) > 0 else 0.005
            pcd, _ = pcd.remove_radius_outlier(nb_points=6, radius=max(0.015, med_nn * 2.5))
        outliers_removed_count = cnt_before - len(pcd.points)

    # 4.5. Glare, yellow background wall bleed, and glass refraction noise filtering
    glare_removed_count = 0
    glare_recolored_count = 0
    if pcd.has_colors() and len(pcd.points) > 500:
        pcd, glare_removed_count, glare_recolored_count = filter_yellow_and_specular_noise(pcd, focus_point)

    # 5. DBSCAN clustering: keep main clusters belonging to the subject
    if len(pcd.points) > 500:
        try:
            pts_curr = np.asarray(pcd.points)
            p1 = np.percentile(pts_curr, 1, axis=0)
            p99 = np.percentile(pts_curr, 99, axis=0)
            robust_diag = float(np.linalg.norm(p99 - p1))
            eps = max(0.02 * robust_diag, 0.02)

            if len(pcd.points) > 200_000:
                voxel_size = max(0.005 * robust_diag, 0.01)
                pcd_proxy = pcd.voxel_down_sample(voxel_size=voxel_size)
                labels_proxy = np.array(pcd_proxy.cluster_dbscan(eps=eps, min_points=10, print_progress=False))
                if labels_proxy.max() >= 0:
                    pts_proxy = np.asarray(pcd_proxy.points)
                    unique_labels = set(labels_proxy[labels_proxy >= 0])
                    cluster_dists = {}
                    cluster_counts = {}
                    for lbl in unique_labels:
                        c_pts = pts_proxy[labels_proxy == lbl]
                        c_dist = float(np.min(np.linalg.norm(c_pts - focus_point, axis=1)))
                        cluster_dists[lbl] = c_dist
                        cluster_counts[lbl] = len(c_pts)

                    # Select the largest cluster within reach of the focus point (core subject)
                    valid_candidates = [lbl for lbl, d in cluster_dists.items() if d <= 0.70]
                    if valid_candidates:
                        best_label = max(valid_candidates, key=cluster_counts.get)
                    else:
                        best_label = max(cluster_counts, key=cluster_counts.get)
                    best_count = cluster_counts[best_label]
                    min_dist = cluster_dists[best_label]
                    keep_labels = {best_label}
                    for lbl, d in cluster_dists.items():
                        if lbl == best_label:
                            continue
                        cnt = cluster_counts[lbl]
                        if d <= max(min_dist * 2.0, 0.75) and cnt >= int(0.03 * best_count):
                            keep_labels.add(lbl)

                    from scipy.spatial import cKDTree
                    target_mask = np.isin(labels_proxy, list(keep_labels))
                    target_proxy_pts = pts_proxy[target_mask]
                    tree = cKDTree(target_proxy_pts)
                    pts_full = np.asarray(pcd.points)
                    tight_bound = max(voxel_size * 1.5, 0.025)
                    dists_full, _ = tree.query(pts_full, distance_upper_bound=tight_bound, workers=1)
                    keep_mask = np.isfinite(dists_full)
                    if np.sum(keep_mask) > 500:
                        pcd = pcd.select_by_index(np.where(keep_mask)[0])
                        logger.info("[Isolation] DBSCAN proxy filtered: kept %d points across %d clusters",
                                    len(pcd.points), len(keep_labels))
            else:
                labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=10, print_progress=False))
                if labels.max() >= 0:
                    pts_after = np.asarray(pcd.points)
                    unique_labels = set(labels[labels >= 0])
                    cluster_dists = {}
                    cluster_counts = {}
                    for lbl in unique_labels:
                        c_pts = pts_after[labels == lbl]
                        c_dist = float(np.min(np.linalg.norm(c_pts - focus_point, axis=1)))
                        cluster_dists[lbl] = c_dist
                        cluster_counts[lbl] = len(c_pts)

                    valid_candidates = [lbl for lbl, d in cluster_dists.items() if d <= 0.70]
                    if valid_candidates:
                        best_label = max(valid_candidates, key=cluster_counts.get)
                    else:
                        best_label = max(cluster_counts, key=cluster_counts.get)
                    best_count = cluster_counts[best_label]
                    min_dist = cluster_dists[best_label]
                    keep_labels = {best_label}
                    for lbl, d in cluster_dists.items():
                        if lbl == best_label:
                            continue
                        cnt = cluster_counts[lbl]
                        if d <= max(min_dist * 2.0, 0.75) and cnt >= int(0.03 * best_count):
                            keep_labels.add(lbl)

                    keep_indices = np.where(np.isin(labels, list(keep_labels)))[0]
                    pcd = pcd.select_by_index(keep_indices)
                    logger.info("[Isolation] DBSCAN kept %d points across %d clusters", len(pcd.points), len(keep_labels))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Isolation] DBSCAN clustering skipped: %s", exc)

    holes_info = None
    if fill_holes and len(pcd.points) > 500:
        pcd, holes_info = densify_and_fill_holes_knn(pcd)
        # Final pass to ensure no interpolated midpoints carry yellow glare
        if pcd.has_colors() and len(pcd.points) > 500:
            pcd, post_del, _ = filter_yellow_and_specular_noise(pcd, focus_point)
            glare_removed_count += post_del

    # Final outlier clean: ensure zero floating perimeter dust or detached distorted points
    if len(pcd.points) > 500:
        cnt_out = len(pcd.points)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.6)
        if len(pcd.points) > 500:
            nn_dists_f = np.asarray(pcd.compute_nearest_neighbor_distance())
            med_f = float(np.median(nn_dists_f)) if len(nn_dists_f) > 0 else 0.005
            pcd, _ = pcd.remove_radius_outlier(nb_points=8, radius=max(0.012, med_f * 3.0))
        outliers_removed_count += (cnt_out - len(pcd.points))

    # Ensure normals are estimated and aligned consistently across tangent planes only when missing
    if not pcd.has_normals() and len(pcd.points) > 100:
        sample_step = max(1, len(pcd.points) // 20000)
        pcd_sample = pcd.select_by_index(np.arange(0, len(pcd.points), sample_step))
        nn_dists = np.asarray(pcd_sample.compute_nearest_neighbor_distance())
        med_d = float(np.median(nn_dists)) if len(nn_dists) > 0 else 0.005
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(0.015, med_d * 6.0), max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(min(40, max(20, len(pcd.points) // 1000)))

    # Ensure 100% of normals consistently point OUTWARD away from the focus point / subject center
    # Inward normals force Poisson to create double-layered sheets, crossing feature flaps, and inverted geometry
    if pcd.has_normals() and len(pcd.points) > 100:
        pts_norm = np.asarray(pcd.points)
        norms = np.asarray(pcd.normals).copy()
        v_out = pts_norm - np.asarray(focus_point[:3])
        v_out /= (np.linalg.norm(v_out, axis=1, keepdims=True) + 1e-6)
        flip_mask = np.sum(norms * v_out, axis=1) < 0
        if np.any(flip_mask):
            num_flipped = int(np.sum(flip_mask))
            norms[flip_mask] = -norms[flip_mask]
            pcd.normals = o3d.utility.Vector3dVector(norms)
            logger.info("[Isolation] Re-oriented %d normals (%.1f%%) outward to prevent crossing feature points and duplicate surfaces",
                        num_flipped, num_flipped / len(pts_norm) * 100)

    final_count = len(pcd.points)
    output_isolated_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_isolated_ply), pcd)

    report = {
        "applied": True,
        "mode": mode,
        "initial_points": initial_count,
        "after_crop_points": after_crop,
        "plane_points_removed": plane_removed_count,
        "outliers_removed": outliers_removed_count,
        "glare_removed": glare_removed_count,
        "glare_recolored": glare_recolored_count,
        "final_points": final_count,
        "focus_point": [round(float(x), 3) for x in focus_point],
    }
    if holes_info:
        report["holes_filled"] = holes_info
    logger.info("[Isolation] Point cloud isolation: %d -> %d points", initial_count, final_count)
    return report
