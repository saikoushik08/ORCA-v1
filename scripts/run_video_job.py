"""Run a 3D reconstruction job from a video file through ORCA.

Usage:
    python -m scripts.run_video_job path/to/video.mp4 [--preset draft|standard|high] [--local] [--url http://127.0.0.1:8000]

Examples:
    python -m scripts.run_video_job my_object.mp4
    python -m scripts.run_video_job my_object.mp4 --preset draft
    python -m scripts.run_video_job my_object.mp4 --preset high --local
"""
import argparse
import mimetypes
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

import httpx

from app.config import settings
from app.services.power_utils import keep_awake
from app.services.presets import get_preset

VALID_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def run_via_api(video_path: Path, preset: str, base_url: str, max_frames: Optional[int] = None):
    print(f"\n=======================================================")
    print(f"  ORCA Video Reconstruction Pipeline (API Mode)")
    print(f"=======================================================")
    print(f"Video   : {video_path.resolve()} ({video_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"Preset  : {preset}")
    if max_frames:
        print(f"Max Frames: {max_frames}")
    print(f"Endpoint: {base_url}")
    print(f"-------------------------------------------------------\n")

    with httpx.Client(base_url=base_url, timeout=300.0) as client:
        # 1. Health check
        try:
            r = client.get("/health")
            if r.status_code != 200:
                print(f"[Warning] /health returned {r.status_code}: {r.text}")
        except Exception as exc:
            print(f"[Error] Could not connect to ORCA server at {base_url}: {exc}")
            print(f"Tip: Start the server first with:\n    uvicorn app.main:app --reload")
            print(f"Or run in offline local mode with:\n    python -m scripts.run_video_job \"{video_path}\" --local")
            sys.exit(1)

        # 2. Create job
        print("[1/5] Creating reconstruction job...")
        r = client.post("/jobs", params={"method": "method_1", "preset": preset})
        if r.status_code >= 400:
            print(f"Failed to create job: {r.text}")
            sys.exit(1)
        job_id = r.json()["job_id"]
        print(f"      Job ID: {job_id}")

        # 3. Upload video
        print(f"[2/5] Uploading video and selecting keyframes...")
        mime_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
        upload_params = {"max_frames": max_frames} if max_frames else {}
        with open(video_path, "rb") as vf:
            r = client.post(
                f"/upload/video/{job_id}",
                params=upload_params,
                files={"video": (video_path.name, vf, mime_type)},
            )
        if r.status_code >= 400:
            print(f"Video upload failed: {r.text}")
            sys.exit(1)
        up_data = r.json()
        print(f"      Candidate frames extracted : {up_data.get('frames_extracted')}")
        print(f"      Selected quality keyframes : {up_data.get('frames_kept')}")
        print(f"      FPS / Duration             : {up_data.get('fps')} fps / {up_data.get('duration_s')}s")
        if up_data.get("reasons_dropped"):
            print(f"      Filter breakdown           : {up_data.get('reasons_dropped')}")

        # 4. Start reconstruction pipeline
        print(f"\n[3/5] Starting reconstruction worker pipeline...")
        r = client.post(f"/jobs/{job_id}/start")
        if r.status_code >= 400:
            print(f"Failed to start job: {r.text}")
            sys.exit(1)

        # 5. Poll progress
        print(f"[4/5] Pipeline executing on GPU (streaming live progress)...")
        last_status = None
        while True:
            r = client.get(f"/jobs/{job_id}")
            if r.status_code == 200:
                data = r.json()
                status = data.get("status")
                stage = data.get("stage") or data.get("current_stage") or "running"
                prog = data.get("progress", 0)
                status_line = f"      [{prog:>3}%] Stage: {stage:<18} (Status: {status})"
                if status_line != last_status:
                    print(status_line)
                    last_status = status_line
                if status in ("completed", "failed"):
                    break
            time.sleep(3)

        # 6. Final report and outputs
        print(f"\n[5/5] Job Finished!")
        if status == "failed":
            print(f"ERROR: Job failed at stage '{data.get('error_stage')}': {data.get('error_message')}")
            sys.exit(1)

        # Quality Report
        try:
            rq = client.get(f"/jobs/{job_id}/quality")
            if rq.status_code == 200:
                qdata = rq.json()
                print(f"\n--- Quality Report (F-30 100-Point Matrix) ---")
                print(f"Quality Score : {qdata.get('score')} / 100")
                print(f"Verdict       : {qdata.get('verdict')}")
                print(f"Vertices      : {qdata.get('vertices'):,}")
                print(f"Faces         : {qdata.get('faces'):,}")
                print(f"Watertight    : {qdata.get('watertight')}")
                print(f"Euler Number  : {qdata.get('euler_number')}")
        except Exception:
            pass

        # Result Files
        res = client.get(f"/jobs/{job_id}/result").json()
        print(f"\n--- Exported 3D Models & Files (F-40) ---")
        for f in res.get("files", []):
            print(f"  • {f['name']:<22} -> {base_url}{f['url']}")

        out_local = settings.outputs_root / job_id
        if out_local.exists():
            print(f"\nLocal Output Folder: {out_local.resolve()}")


def run_locally(video_path: Path, preset: str, max_frames: Optional[int] = None, fps: Optional[float] = None, max_duration: Optional[int] = None):
    print(f"\n=======================================================")
    print(f"  ORCA Video Reconstruction Pipeline (Local Direct Mode)")
    print(f"=======================================================")
    print(f"Video   : {video_path.resolve()} ({video_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"Preset  : {preset}")
    if max_frames:
        print(f"Max Frames: {max_frames}")
    if max_duration:
        print(f"Max Duration: {max_duration}s")
    print(f"-------------------------------------------------------\n")

    from app.jobs.job_manager import JobManager
    from app.jobs.job_repository import JobRepository
    from app.logging_config import setup_logging
    from app.services.video_utils import extract_and_select_keyframes
    from app.workers.local_worker import LocalWorker

    setup_logging(settings.log_level)

    preset_obj = get_preset(preset)
    budget = max_frames or getattr(preset_obj, "video_frame_budget", 250)
    candidate_fps = fps or settings.video_candidate_fps

    # 1. Create job
    job_id = uuid4()
    JobRepository.create_job(job_id=job_id, method="method_1", preset=preset, input_type="video")
    JobManager.mark_uploading(job_id)
    print(f"[1/4] Created job: {job_id}")

    # 2. Extract keyframes
    print(f"[2/4] Extracting candidate frames and selecting keyframes...")
    target_dir = settings.uploads_root / str(job_id) / "images"
    target_dir.mkdir(parents=True, exist_ok=True)
    keyframes, v_report = extract_and_select_keyframes(
        video_path=video_path,
        frame_budget=budget,
        candidate_fps=candidate_fps,
        min_parallax_ratio=0.01,
        min_blur_ratio=0.45,
        max_duration_s=max_duration,
    )
    print(f"      Extracted {v_report['frames_extracted']} candidates -> Kept {v_report['frames_kept']} keyframes")
    print(f"      Duration: {v_report['duration_s']:.1f}s, FPS: {v_report['fps']:.1f}")
    if v_report.get("reasons_dropped"):
        print(f"      Filter breakdown: {v_report['reasons_dropped']}")

    if len(keyframes) < settings.min_images_per_job:
        print(f"Error: Only {len(keyframes)} usable keyframes could be extracted (minimum {settings.min_images_per_job}).")
        sys.exit(1)

    # Save frames to target_dir and register in repository
    for kf in keyframes:
        filename = f"frame_{kf.frame_index:06d}.jpg"
        (target_dir / filename).write_bytes(kf.image_bytes)
        JobRepository.add_image(
            job_id=job_id,
            image_path=f"{job_id}/{filename}",
            original_name=filename,
            sha256=kf.sha256,
            width=kf.width,
            height=kf.height,
            size_bytes=kf.size_bytes,
            source_type="video_frame",
            source_video=video_path.name,
            frame_index=kf.frame_index,
            frame_time_s=kf.timestamp_s,
        )

    # 3. Start & execute job
    print(f"[3/4] Starting photogrammetry pipeline...")
    from app.jobs.job_states import JobStatus
    from app.workers.worker_executor import WorkerExecutor

    current_status = JobManager._status(job_id)
    if current_status == JobStatus.UPLOADING:
        JobManager.start_job(job_id)
    current_status = JobManager._status(job_id)
    if current_status == JobStatus.PENDING:
        LocalWorker.execute(job_id)
    elif current_status == JobStatus.PROCESSING:
        WorkerExecutor.run(job_id)

    # 4. Results
    job = JobRepository.get_job(job_id)
    print(f"\n[4/4] Pipeline Complete! Final Status: {job.get('status')}")
    out_dir = settings.outputs_root / str(job_id)
    glb_exists = (out_dir / "final_mesh.glb").exists()
    if job.get("status") == "failed" and not glb_exists:
        print(f"ERROR: {job.get('error_message')}")
        sys.exit(1)
    elif job.get("status") == "failed" and glb_exists:
        print(f"NOTE: Mesh was successfully produced despite status update conflict.")

    out_dir = settings.outputs_root / str(job_id)
    print(f"\n--- Output Files in {out_dir.resolve()} ---")
    for ext in ("glb", "ply", "obj", "stl"):
        m = out_dir / f"final_mesh.{ext}"
        if m.exists():
            print(f"  • {m.name} ({m.stat().st_size / 1024:.1f} KB)")

    q_file = out_dir / "quality.json"
    if q_file.exists():
        import json
        q = json.loads(q_file.read_text(encoding="utf-8"))
        print(f"\nQuality Score : {q.get('score')} / 100 ({q.get('verdict')})")


def main():
    parser = argparse.ArgumentParser(description="Run a video reconstruction job in ORCA.")
    parser.add_argument("video", type=str, help="Path to video file (.mp4, .mov, .webm, etc.)")
    parser.add_argument("--preset", type=str, default="standard", choices=["draft", "standard", "high"],
                        help="Quality preset profile (default: standard)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of frames to extract (e.g. 250, 350). Defaults to preset budget.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Candidate frame extraction FPS (default: 10.0)")
    parser.add_argument("--max-duration", type=int, default=None,
                        help="Maximum video duration in seconds (defaults to settings.video_max_duration_s)")
    parser.add_argument("--local", action="store_true",
                        help="Run directly in local Python process (bypasses HTTP server)")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000",
                        help="ORCA API URL (default: http://127.0.0.1:8000)")

    args = parser.parse_args()
    video_path = Path(args.video)

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    if video_path.suffix.lower() not in VALID_VIDEO_EXTS:
        print(f"Error: Unsupported video format {video_path.suffix}. Allowed: {', '.join(sorted(VALID_VIDEO_EXTS))}")
        sys.exit(1)

    if args.local:
        run_locally(video_path, args.preset, max_frames=args.max_frames, fps=args.fps, max_duration=args.max_duration)
    else:
        run_via_api(video_path, args.preset, args.url, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
