"""Mesh generation and clean-up.

Chain: COLMAP Poisson mesh (made in the dense step)
       -> if missing/unusable: Ball-Pivoting on fused.ply.
All clean-up is done with Open3D (keeps vertex colours, no version-fragile
trimesh calls). Every step records {applied, count} so nothing is skipped silently.
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import trimesh

from app.config import settings
from app.jobs.errors import PipelineError
from app.services.presets import Preset, get_preset

logger = logging.getLogger(__name__)

MIN_USABLE_FACES = 500   # a Poisson/BPA mesh with fewer faces than this is not usable


# ---------------------------------------------------------------------------
# Loading / fallback generation
# ---------------------------------------------------------------------------
def _load_mesh(path: Optional[Path]):
    if not path or not Path(path).exists():
        return None
    try:
        mesh = o3d.io.read_triangle_mesh(str(path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Mesh] Could not read %s: %s", path, exc)
        return None
    if len(mesh.vertices) == 0 or len(mesh.triangles) < MIN_USABLE_FACES:
        logger.warning("[Mesh] %s is empty or too small (%d faces)", path, len(mesh.triangles))
        return None
    return mesh


def _poisson_from_fused(fused_ply: Path, depth: int = 11, trim_percentile: float = 0.0):
    """High-depth Screened Poisson reconstruction directly on the isolated point cloud.
    Uses adaptive normal estimation based on actual point cloud scale."""
    if not fused_ply.exists():
        return None
    pcd = o3d.io.read_point_cloud(str(fused_ply))
    n = len(pcd.points)
    if n < 50:
        return None

    # Cap depth to prevent C++ memory exhaustion on dense point clouds
    if n > 2_000_000:
        depth = min(depth, 10)
    else:
        depth = min(depth, 11)

    # If point cloud is large, sample to ~1.2M points for Poisson solver
    # This generates ~2-3M Poisson faces, which preserves all fine facial features
    # (eyes, nose, spectacles, lips, hairline) while keeping decimation fast and stable
    pcd_poisson = pcd
    if n > 1_200_000:
        ratio = 1_200_000 / n
        pcd_poisson = pcd.random_down_sample(ratio)
        logger.info("[Mesh] Downsampled point cloud for Poisson: %d -> %d points", n, len(pcd_poisson.points))

    # Adaptive normal estimation — radius based on actual point cloud scale
    # This is CRITICAL: a fixed radius=0.03 fails on real-world-scale scenes
    # where points may be meters apart.
    pts = np.asarray(pcd_poisson.points)
    nn_dists = np.asarray(pcd_poisson.compute_nearest_neighbor_distance())
    median_dist = float(np.median(nn_dists)) if len(nn_dists) > 0 else 0.005
    bbox = pcd_poisson.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(bbox.get_extent()))

    if not pcd_poisson.has_normals():
        # Use 8x median nn distance as radius — captures local surface geometry
        # without over-smoothing fine features (eyes, nose, lips)
        normal_radius = max(median_dist * 8.0, diag * 0.005)
        normal_nn = min(50, max(20, len(pcd_poisson.points) // 500))
        logger.info("[Mesh] Adaptive normal estimation: radius=%.4f, max_nn=%d (median_dist=%.4f, diag=%.3f)",
                    normal_radius, normal_nn, median_dist, diag)
        pcd_poisson.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_nn)
        )
        pcd_poisson.orient_normals_consistent_tangent_plane(min(60, max(30, len(pcd_poisson.points) // 1000)))

    # Ensure 100% outward-pointing normals: prevents double-layer surfaces, inverted faces, and crossing flaps
    if pcd_poisson.has_normals():
        pts_p = np.asarray(pcd_poisson.points)
        norms = np.asarray(pcd_poisson.normals).copy()
        v_out = pts_p - np.median(pts_p, axis=0)
        v_out /= (np.linalg.norm(v_out, axis=1, keepdims=True) + 1e-6)
        flip_mask = np.sum(norms * v_out, axis=1) < 0
        if np.any(flip_mask):
            norms[flip_mask] = -norms[flip_mask]
            pcd_poisson.normals = o3d.utility.Vector3dVector(norms)

    try:
        # scale=1.1 gives Poisson slightly more room to fit concave features
        # (eye sockets, nostrils) without over-closing them.
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_poisson, depth=depth, linear_fit=True, scale=1.1
        )
        densities = np.asarray(densities)
        if len(densities) > 0 and trim_percentile > 0:
            # Conservative trim: trim ONLY the outer ballooning hull in empty air
            # (bottom 1.5% of densities) without eating into lower-coverage facial or body features.
            effective_trim = min(1.5, max(0.5, trim_percentile))
            thresh = np.percentile(densities, effective_trim)
            before_trim = len(mesh.triangles)
            mesh.remove_vertices_by_mask(densities < thresh)
            logger.info("[Mesh] Poisson density trim: %.1f%% -> %d -> %d faces",
                        effective_trim, before_trim, len(mesh.triangles))
        if len(mesh.triangles) < MIN_USABLE_FACES:
            return None
        return mesh
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Mesh] Screened Poisson on fused cloud failed: %s", exc)
        return None


def _ball_pivoting_from_fused(fused_ply: Path):
    """Ball-Pivoting on the dense cloud. Needs no watertight closure and does not
    invent surface over gaps, but the result is usually not watertight."""
    if not fused_ply.exists():
        return None
    pcd = o3d.io.read_point_cloud(str(fused_ply))
    n = len(pcd.points)
    if n == 0:
        return None

    if n > settings.bpa_max_points:                       # keep memory/time sane
        pcd = pcd.random_down_sample(settings.bpa_max_points / n)
        logger.info("[Mesh] BPA: downsampled %d -> %d points", n, len(pcd.points))

    if not pcd.has_normals():                             # COLMAP's fused.ply normally has them
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
        pcd.orient_normals_consistent_tangent_plane(30)

    mean_dist = float(np.mean(np.asarray(pcd.compute_nearest_neighbor_distance())))
    if not np.isfinite(mean_dist) or mean_dist <= 0:
        return None
    radii = o3d.utility.DoubleVector([mean_dist * f for f in (1.0, 1.8, 3.2, 5.5)])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    if len(mesh.triangles) < MIN_USABLE_FACES:
        return None
    return mesh


# ---------------------------------------------------------------------------
# Clean-up steps (each returns the mesh and appends to `steps`)
# ---------------------------------------------------------------------------
def _record(steps: list, name: str, applied: bool, before: int, after: int, note: str = ""):
    steps.append({"step": name, "applied": applied, "faces_before": before,
                  "faces_after": after, "note": note})
    logger.info("[Mesh] %-26s %s faces %d -> %d %s",
                name, "applied" if applied else "SKIPPED", before, after, note)


def _basic_clean(mesh, steps: list):
    before = len(mesh.triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    _record(steps, "basic_clean", True, before, len(mesh.triangles))
    return mesh


def _decimate(mesh, target: int, steps: list, name: str):
    before = len(mesh.triangles)
    if before <= target:
        _record(steps, name, False, before, before, "already below target")
        return mesh
    out = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target))
    out.remove_unreferenced_vertices()
    _record(steps, name, True, before, len(out.triangles))
    return out


def _keep_main_components(mesh, steps: list, name: str = "keep_largest_component"):
    """Keep the largest connected piece and any significant components (head, torso, limbs),
    while removing small floating fragments/specks."""
    before = len(mesh.triangles)
    cluster_idx, cluster_sizes, _ = mesh.cluster_connected_triangles()
    cluster_idx = np.asarray(cluster_idx)
    sizes = np.asarray(cluster_sizes)
    if sizes.size <= 1:
        _record(steps, name, False, before, before, "single component")
        return mesh, 1

    largest_size = int(np.max(sizes))
    largest_idx = int(np.argmax(sizes))

    # If the largest component holds >= 80% of all triangles, it is the complete subject.
    # Keep strictly the main component to purge all floating background dust/specks.
    # Otherwise, keep only substantial components that have at least 5% of the largest component.
    if largest_size >= int(0.80 * before):
        keep_mask_clusters = np.zeros(sizes.size, dtype=bool)
        keep_mask_clusters[largest_idx] = True
    else:
        min_triangles = max(500, int(0.05 * largest_size))
        keep_mask_clusters = sizes >= min_triangles
        keep_mask_clusters[largest_idx] = True

    remove = ~keep_mask_clusters[cluster_idx]
    num_kept = int(np.sum(keep_mask_clusters))
    mesh.remove_triangles_by_mask(remove.tolist())
    mesh.remove_unreferenced_vertices()
    _record(steps, name, True, before, len(mesh.triangles),
            f"{sizes.size} components -> {num_kept} kept")
    return mesh, int(sizes.size)


def _smooth(mesh, iterations: int, steps: list):
    before = len(mesh.triangles)
    if iterations <= 0:
        _record(steps, "taubin_smoothing", False, before, before, "disabled")
        return mesh
    out = mesh.filter_smooth_taubin(number_of_iterations=iterations)
    _record(steps, "taubin_smoothing", True, before, len(out.triangles), f"{iterations} iterations")
    return out


def _prune_empty_air_faces(mesh, pcd_source, steps: list, max_dist_mm: float = 16.0):
    """Prune any Poisson mesh faces that extend into empty air beyond the point cloud.
    This cleanly eliminates hanging shoulder wings, fins, and bottom balloons without touching
    authentic facial or body features (which are within 1-2mm of the cloud)."""
    before = len(mesh.triangles)
    if before == 0 or pcd_source is None or len(pcd_source.points) == 0:
        return mesh
    try:
        from scipy.spatial import cKDTree
        pcd_pts = np.asarray(pcd_source.points)
        if len(pcd_pts) > 500_000:
            step = max(1, len(pcd_pts) // 500_000)
            pcd_pts = pcd_pts[::step]
        tree = cKDTree(pcd_pts)

        mv = np.asarray(mesh.vertices)
        mf = np.asarray(mesh.triangles)
        dists, _ = tree.query(mv, k=1, workers=1)

        dist_thresh = max_dist_mm / 1000.0

        # A face is in empty air if at least 2 of its 3 vertices exceed dist_thresh
        bad_v = dists > dist_thresh
        bad_faces = np.sum(bad_v[mf], axis=1) >= 2
        num_bad = int(np.sum(bad_faces))

        if num_bad > 0:
            mesh.remove_triangles_by_mask(bad_faces.tolist())
            mesh.remove_unreferenced_vertices()
            _record(steps, "prune_empty_air_shards", True, before, len(mesh.triangles),
                    f"{num_bad} empty-air faces pruned (dist > {dist_thresh*1000:.1f}mm)")
        else:
            _record(steps, "prune_empty_air_shards", False, before, before, "no empty air faces")
        return mesh
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Mesh] Empty air pruning skipped: %s", exc)
        _record(steps, "prune_empty_air_shards", False, before, before, str(exc))
        return mesh


def _repair_watertight(mesh, steps: list):
    """Fill open boundary loops and ensure watertight manifold closure."""
    before = len(mesh.triangles)
    try:
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.triangles)
        c = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
        tm = trimesh.Trimesh(vertices=v, faces=f, process=True)

        # 1. Clean non-manifold edges so boundary edges can close cleanly
        edges_sorted = tm.edges_sorted
        eu, inv, c_edges = np.unique(edges_sorted, return_inverse=True, return_counts=True, axis=0)
        is_nm = c_edges > 2
        if np.any(is_nm):
            edge_is_nm = is_nm[inv]
            nm_faces = np.unique(np.where(edge_is_nm)[0] // 3)
            nm_mask = np.zeros(len(tm.faces), dtype=bool)
            nm_mask[nm_faces] = True
            tm.update_faces(~nm_mask)
            tm.remove_unreferenced_vertices()

        # 2. Prune any abnormally stretched boundary faces (preventing fan blades / wings on shoulders)
        tm_v = tm.vertices
        tm_f = tm.faces
        if len(tm_f) > 0:
            e0 = np.linalg.norm(tm_v[tm_f[:, 0]] - tm_v[tm_f[:, 1]], axis=1)
            e1 = np.linalg.norm(tm_v[tm_f[:, 1]] - tm_v[tm_f[:, 2]], axis=1)
            e2 = np.linalg.norm(tm_v[tm_f[:, 2]] - tm_v[tm_f[:, 0]], axis=1)
            max_e = np.maximum(np.maximum(e0, e1), e2)
            med_e = float(np.median(max_e)) if len(max_e) > 0 else 0.012
            bad_faces = max_e > max(0.025, med_e * 2.5)
            if np.any(bad_faces):
                tm.update_faces(~bad_faces)
                tm.remove_unreferenced_vertices()

        trimesh.repair.fix_winding(tm)
        trimesh.repair.fix_normals(tm)

        out = o3d.geometry.TriangleMesh()
        out.vertices = o3d.utility.Vector3dVector(tm.vertices)
        out.triangles = o3d.utility.Vector3iVector(tm.faces)
        if c is not None and len(c) > 0:
            if len(out.vertices) > len(c):
                new_c = np.zeros((len(out.vertices), 3))
                new_c[:len(c)] = c
                new_c[len(c):] = np.mean(c, axis=0)
                out.vertex_colors = o3d.utility.Vector3dVector(new_c)
            else:
                out.vertex_colors = o3d.utility.Vector3dVector(c[:len(out.vertices)])
        _record(steps, "watertight_repair", True, before, len(out.triangles),
                f"watertight={tm.is_watertight}")
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Mesh] Watertight repair skipped: %s", exc)
        _record(steps, "watertight_repair", False, before, before, str(exc))
        return mesh


def _transfer_vertex_colors(mesh, pcd):
    """Transfer K-nearest-neighbor weighted-average colors from dense cloud to mesh vertices.
    Uses K=5 with vectorized inverse-distance weighting for smoother, accurate skin and texture colors."""
    if not pcd.has_colors() or len(mesh.vertices) == 0:
        return mesh
    try:
        from scipy.spatial import cKDTree
        pcd_pts = np.asarray(pcd.points)
        pcd_cols = np.asarray(pcd.colors)
        v_arr = np.asarray(mesh.vertices)
        if len(pcd_pts) == 0 or len(pcd_cols) == 0 or len(v_arr) == 0:
            return mesh

        # Subsample donor points if very dense to keep KDTree query fast (< 1s)
        if len(pcd_pts) > 2_500_000:
            step = max(1, len(pcd_pts) // 2_000_000)
            pcd_pts = pcd_pts[::step]
            pcd_cols = pcd_cols[::step]

        tree = cKDTree(pcd_pts)
        k_neighbors = min(7, len(pcd_pts))
        dists, idxs = tree.query(v_arr, k=k_neighbors, workers=1)
        if k_neighbors == 1:
            mesh_colors = pcd_cols[idxs].copy()
        else:
            weights = 1.0 / (dists + 1e-8)
            weights /= np.sum(weights, axis=1, keepdims=True)
            mesh_colors = np.sum(weights[:, :, None] * pcd_cols[idxs], axis=1)

        # Facial & Clothing Polish: purge any remaining yellow wall glare and bleached specular flare
        # Human skin has strong red dominance (mr - mg >= 0.15, ratio_gr <= 0.70, mb < mg).
        mr, mg, mb = mesh_colors[:, 0], mesh_colors[:, 1], mesh_colors[:, 2]
        ratio_gr = mg / (mr + 1e-6)

        authentic_skin_mesh = (mr >= 0.35) & (mr <= 0.75) & ((mr - mg) >= 0.15) & (ratio_gr <= 0.70) & (mb < mg)

        is_yellow = (mr > 0.35) & (mg > 0.25) & (ratio_gr > 0.68) & (mb < 0.90 * mr) & ((mr - mg) < 0.15)
        is_yellow_2 = (mr > 0.42) & (mg > 0.32) & (ratio_gr > 0.70) & (mb < 0.95 * mr) & ((mr - mg) < 0.13)
        is_cream_white = (mr > 0.52) & (mg > 0.42) & (ratio_gr > 0.72) & ((mr - mg) < 0.12)
        is_bleached_all = (mr > 0.65) & (mg > 0.55)
        is_shirt_noise = (mr > 0.25) & (mg > 0.30) & (ratio_gr > 0.60) & (mb < 0.45) & (mr < 0.50)

        mesh_glare = (is_yellow | is_yellow_2 | is_cream_white | is_bleached_all | is_shirt_noise) & (~authentic_skin_mesh)

        clean_mask = ~mesh_glare & ~((mr > 0.55) & (mg > 0.45))
        if np.sum(clean_mask) < 20:
            clean_mask = ~mesh_glare

        if np.sum(mesh_glare) > 0 and np.sum(clean_mask) > 20:
            head_tree = cKDTree(v_arr[clean_mask])
            _, h_idxs = head_tree.query(v_arr[mesh_glare], k=min(5, int(np.sum(clean_mask))), workers=1)
            clean_cols = mesh_colors[clean_mask]
            if h_idxs.ndim == 1:
                mesh_colors[mesh_glare] = clean_cols[h_idxs]
            else:
                mesh_colors[mesh_glare] = np.median(clean_cols[h_idxs], axis=1)

        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(mesh_colors, 0.0, 1.0))
        logger.info("[Mesh] Transferred vertex colors to %d vertices", len(mesh.vertices))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Mesh] Color transfer failed: %s", exc)
    return mesh


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_mesh(poisson_mesh_path: Optional[Path], fused_ply_path: Path, output_dir: Path,
               preset: Optional[Preset] = None, surface_method: Optional[str] = None) -> dict:
    """
    Produce final_mesh.ply and mesh_report.json in output_dir.

    Returns a JSON-ready report:
      final_mesh, method_used ('poisson'|'ball_pivoting'), fallback_used,
      faces, vertices, steps[], seconds
    Raises PipelineError if neither Poisson nor Ball-Pivoting gives a usable mesh.
    """
    target_faces = preset.mesh_target_faces if preset is not None else settings.mesh_target_faces
    taubin_iterations = preset.taubin_iterations if preset is not None else settings.taubin_iterations

    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fused_ply_path = Path(fused_ply_path)
    steps: list = []

    mesh = _load_mesh(poisson_mesh_path)
    method = "poisson"
    if mesh is None and surface_method == "poisson":
        depth = preset.poisson_depth if preset is not None else 11
        logger.info("[Mesh] Running Screened Poisson (depth=%d) directly on isolated cloud", depth)
        # trim_percentile=0.0 ensures Poisson smoothly closes all facial and body gaps (cheeks, throat, temple) without carving holes
        mesh = _poisson_from_fused(fused_ply_path, depth=depth, trim_percentile=0.0)

    if mesh is None:
        logger.warning("[Mesh] No usable Poisson mesh: falling back to Ball-Pivoting")
        try:
            mesh = _ball_pivoting_from_fused(fused_ply_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Mesh] Ball-Pivoting failed: %s", exc)
            mesh = None
        method = "ball_pivoting"
    if mesh is None:
        raise PipelineError(
            "Neither Poisson meshing nor Ball-Pivoting produced a usable mesh.",
            "no_mesh", "mesh",
        )
    fallback_used = method != "poisson"
    initial_faces = len(mesh.triangles)
    logger.info("[Mesh] Start: %s mesh with %d faces", method, initial_faces)

    pcd_source = None
    if fused_ply_path.exists():
        try:
            pcd_source = o3d.io.read_point_cloud(str(fused_ply_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Mesh] Could not read fused ply for reference: %s", exc)

    # Very large meshes are decimated FIRST so every later step stays fast.
    mesh = _decimate(mesh, settings.mesh_max_faces_before_cleanup, steps, "pre_decimation")
    mesh = _basic_clean(mesh, steps)

    # Prune empty-air shards, hanging shoulder wings, and balloon hulls (fast, < 2s on decimated mesh)
    if pcd_source is not None and len(pcd_source.points) > 0:
        mesh = _prune_empty_air_faces(mesh, pcd_source, steps, max_dist_mm=16.0)

    mesh, initial_components = _keep_main_components(mesh, steps)
    mesh = _smooth(mesh, taubin_iterations, steps)
    mesh = _decimate(mesh, target_faces, steps, "target_decimation")
    mesh = _basic_clean(mesh, steps)
    mesh = _repair_watertight(mesh, steps)
    mesh, final_components = _keep_main_components(mesh, steps, name="final_component_filter")
    mesh = _smooth(mesh, min(3, taubin_iterations), steps)

    if pcd_source is not None and len(pcd_source.points) > 0:
        try:
            mesh = _transfer_vertex_colors(mesh, pcd_source)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Mesh] Could not transfer colors: %s", exc)

    mesh.compute_vertex_normals()

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise PipelineError("Mesh was empty after clean-up", "empty_after_cleanup", "mesh")

    final_path = output_dir / "final_mesh.ply"
    if not o3d.io.write_triangle_mesh(str(final_path), mesh, write_ascii=False):
        raise PipelineError("Could not write the final mesh", "export_failed", "mesh")
    if not final_path.exists():
        raise PipelineError("Final mesh export failed", "export_failed", "mesh")

    exported_files = ["final_mesh.ply"]
    try:
        obj_path = output_dir / "final_mesh.obj"
        o3d.io.write_triangle_mesh(str(obj_path), mesh)
        if obj_path.exists():
            exported_files.append("final_mesh.obj")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Mesh] OBJ export failed: %s", exc)

    try:
        stl_path = output_dir / "final_mesh.stl"
        o3d.io.write_triangle_mesh(str(stl_path), mesh)
        if stl_path.exists():
            exported_files.append("final_mesh.stl")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Mesh] STL export failed: %s", exc)

    try:
        glb_path = output_dir / "final_mesh.glb"
        tm = trimesh.load_mesh(str(final_path), process=False)
        tm.export(str(glb_path))
        if glb_path.exists():
            exported_files.append("final_mesh.glb")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Mesh] GLB export failed: %s", exc)

    report = {
        "final_mesh": str(final_path),
        "exported_formats": [p.split(".")[-1] for p in exported_files],
        "exported_files": exported_files,
        "method_used": method,
        "fallback_used": fallback_used,
        "initial_faces": initial_faces,
        "initial_components": initial_components,
        "faces": len(mesh.triangles),
        "vertices": len(mesh.vertices),
        "has_vertex_colors": bool(mesh.has_vertex_colors()),
        "steps": steps,
        "seconds": round(time.time() - t0, 2),
    }
    (output_dir / "mesh_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("[Mesh] Final mesh exported (%s): %d vertices, %d faces (method=%s)",
                ", ".join(exported_files), report["vertices"], report["faces"], method)
    return report
