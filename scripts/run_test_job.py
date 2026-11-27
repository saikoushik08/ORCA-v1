import mimetypes
import sys
import time
from pathlib import Path
import httpx

BASE_URL = "http://127.0.0.1:8000"
IMAGES_DIR = Path(r"K:\Dissertation\orca-backend\uploads\images")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def check_response(r: httpx.Response, step: str):
    if r.status_code >= 400:
        print(f"ERROR at {step} (HTTP {r.status_code}): {r.text}")
        sys.exit(1)


def main():
    open_viewer = "--view" in sys.argv
    limit = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        limit = int(sys.argv[idx + 1])

    if not IMAGES_DIR.is_dir():
        print(f"Error: Folder not found: {IMAGES_DIR}")
        sys.exit(1)

    image_files = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )
    if limit:
        image_files = image_files[:limit]

    if len(image_files) < 5:
        print(f"Error: Need at least 5 images, found {len(image_files)} in {IMAGES_DIR}")
        sys.exit(1)

    print(f"Using {len(image_files)} images from {IMAGES_DIR}")

    with httpx.Client(base_url=BASE_URL, timeout=180.0) as client:
        # 1. Create job
        r = client.post("/jobs", params={"method": "method_1"})
        check_response(r, "create job")
        job_id = r.json()["job_id"]
        print(f"[1/4] Created job: {job_id}")

        # 2. Upload images in batches of 10 with proper MIME types
        batch_size = 10
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i : i + batch_size]
            files_payload = [
                (
                    "files",
                    (
                        p.name,
                        p.read_bytes(),
                        mimetypes.guess_type(p.name)[0] or "application/octet-stream",
                    ),
                )
                for p in batch
            ]
            r = client.post(f"/upload/image/{job_id}", files=files_payload)
            check_response(r, f"upload batch {i // batch_size + 1}")
        print(f"[2/4] Uploaded {len(image_files)} images")

        # 3. Start job
        r = client.post(f"/jobs/{job_id}/start")
        check_response(r, "start job")
        print(f"[3/4] Started job (HTTP {r.status_code}). Waiting for worker...")

        # 4. Poll until completed or failed
        last_status = None
        while True:
            r = client.get(f"/jobs/{job_id}")
            check_response(r, "poll status")
            data = r.json()
            status = data.get("status")
            stage = data.get("current_stage") or "-"
            progress = data.get("progress")

            status_line = f"status={status} | stage={stage} | progress={progress}"
            if status_line != last_status:
                print(f"      {status_line}")
                last_status = status_line

            if status in ("completed", "failed"):
                break
            time.sleep(3)

    # 5. Show result
    print("\n--- Final Result ---")
    if status == "failed":
        print(f"Job FAILED: {data.get('error_message') or data}")
        sys.exit(1)

    mesh_path = Path("outputs") / job_id / "final_mesh.ply"
    print("Job COMPLETED!")
    print(f"Output mesh: {mesh_path.resolve()}")
    if data.get("result"):
        print(f"Metrics/Result: {data['result']}")

    if open_viewer and mesh_path.exists():
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], window_name=f"ORCA - {job_id}")


if __name__ == "__main__":
    main()