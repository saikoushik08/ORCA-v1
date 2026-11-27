"""Core logic tests: state machine, repository, worker flow, COLMAP runner, image checks.
None of these need COLMAP, a GPU or the network."""
import io
import os
import stat
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

import fake_supabase
from fake_supabase import DB
from helpers import make_png


# ---------------------------------------------------------------- helpers
def _new_job(images: int = 0, method: str = "method_1"):
    from app.jobs.job_repository import JobRepository
    jid = uuid4()
    JobRepository.create_job(jid, method)
    for i in range(images):
        JobRepository.add_image(jid, f"{jid}/{i}.png")
    return jid


def _status(jid):
    from app.jobs.job_repository import JobRepository
    return JobRepository.get_job(jid)["status"]


def _queued_job_with_files(n: int = 6):
    """UPLOADING -> PENDING job whose image files really exist on disk."""
    from app.config import settings
    from app.jobs.job_manager import JobManager
    jid = _new_job()
    JobManager.mark_uploading(jid)
    folder = settings.uploads_root / str(jid) / "images"
    folder.mkdir(parents=True, exist_ok=True)
    from app.jobs.job_repository import JobRepository
    for i in range(n):
        (folder / f"{i}.png").write_bytes(make_png(i))
        JobRepository.add_image(jid, f"{jid}/{i}.png")
    JobManager.start_job(jid)
    return jid


# ---------------------------------------------------------------- state machine
def test_transition_table_rules():
    from app.jobs.job_states import JobStatus as S, is_valid_transition as ok
    assert not ok(S.CREATED, S.PENDING)           # no start without images
    assert ok(S.UPLOADING, S.PENDING)
    assert ok(S.PROCESSING, S.PENDING)            # the one recovery edge
    assert not ok(S.PENDING, S.COMPLETED)
    assert not ok(S.COMPLETED, S.FAILED) and not ok(S.FAILED, S.PENDING)


def test_full_lifecycle_and_audit_trail():
    from app.jobs.job_manager import JobManager
    jid = _new_job(5)
    JobManager.mark_uploading(jid)
    JobManager.start_job(jid)
    assert _status(jid) == "pending"
    JobManager.claim_for_processing(jid)
    assert _status(jid) == "processing"
    JobManager.mark_completed(jid, result={"quality": {"verdict": "PASS"}})
    assert _status(jid) == "completed"
    events = [(e["from_status"], e["to_status"]) for e in DB["job_events"]]
    assert events == [("created", "uploading"), ("uploading", "pending"),
                      ("pending", "processing"), ("processing", "completed")]


def test_cannot_start_a_job_that_has_no_upload_state():
    from app.jobs.errors import InvalidTransition
    from app.jobs.job_manager import JobManager
    jid = _new_job()
    with pytest.raises(InvalidTransition):
        JobManager.start_job(jid)
    assert _status(jid) == "created"


def test_transition_is_atomic_second_caller_loses():
    from app.jobs.errors import TransitionConflict
    from app.jobs.job_manager import JobManager
    from app.jobs.job_repository import JobRepository
    from app.jobs.job_states import JobStatus as S
    jid = _new_job(5)
    JobManager.mark_uploading(jid)
    JobRepository.transition(jid, S.UPLOADING, S.PENDING)          # caller A wins
    with pytest.raises(TransitionConflict):                        # caller B used a stale view
        JobRepository.transition(jid, S.UPLOADING, S.PENDING)
    assert _status(jid) == "pending"


def test_only_one_worker_can_claim_a_job():
    from app.jobs.errors import TransitionConflict
    from app.jobs.job_manager import JobManager
    jid = _new_job(5)
    JobManager.mark_uploading(jid)
    JobManager.start_job(jid)
    JobManager.claim_for_processing(jid)
    with pytest.raises(TransitionConflict):
        JobManager.claim_for_processing(jid)


def test_illegal_transition_is_rejected_before_touching_the_database():
    from app.jobs.errors import InvalidTransition
    from app.jobs.job_repository import JobRepository
    from app.jobs.job_states import JobStatus as S
    jid = _new_job()
    with pytest.raises(InvalidTransition):
        JobRepository.transition(jid, S.CREATED, S.COMPLETED)
    assert _status(jid) == "created"


def test_failure_reason_stage_and_code_are_stored():
    from app.jobs.job_manager import JobManager
    from app.jobs.job_repository import JobRepository
    jid = _new_job(5)
    JobManager.mark_failed(jid, reason="boom", stage="dense", code="colmap_failed")
    job = JobRepository.get_job(jid)
    assert job["status"] == "failed"
    assert (job["error_message"], job["error_stage"], job["error_code"]) == (
        "boom", "dense", "colmap_failed")
    assert job["finished_at"]


def test_failed_job_cannot_fail_again():
    from app.jobs.errors import InvalidTransition
    from app.jobs.job_manager import JobManager
    jid = _new_job()
    JobManager.mark_failed(jid, "x")
    with pytest.raises(InvalidTransition):
        JobManager.mark_failed(jid, "y")


def test_old_database_schema_still_works():
    """Without the optional columns the status change must still succeed."""
    from app.jobs.job_manager import JobManager
    fake_supabase.RESTRICT["jobs"] = {"job_id", "method", "status", "created_at", "updated_at"}
    jid = _new_job()
    JobManager.mark_failed(jid, reason="boom", stage="sfm", code="x")
    assert _status(jid) == "failed"


def test_unknown_job_raises_not_found():
    from app.jobs.errors import JobNotFound
    from app.jobs.job_manager import JobManager
    with pytest.raises(JobNotFound):
        JobManager.start_job(uuid4())


# ---------------------------------------------------------------- recovery
def test_recover_interrupted_requeues_then_fails():
    from app.jobs.job_manager import JobManager
    from app.jobs.job_repository import JobRepository
    from app.workers.local_worker import LocalWorker

    def processing_job(attempt):
        jid = _new_job(5)
        JobManager.mark_uploading(jid)
        JobManager.start_job(jid)
        JobManager.claim_for_processing(jid)
        JobRepository.update_fields(jid, attempt=attempt)
        return jid

    first, exhausted = processing_job(1), processing_job(3)
    assert LocalWorker.recover_interrupted() == 2
    assert _status(first) == "pending"
    assert JobRepository.get_job(first)["attempt"] == 2
    assert _status(exhausted) == "failed"
    assert JobRepository.get_job(exhausted)["error_code"] == "worker_interrupted"


# ---------------------------------------------------------------- worker flow
def _fake_outputs(workspace: Path, verdict="PASS"):
    workspace.mkdir(parents=True, exist_ok=True)
    final, fused = workspace / "final_mesh.ply", workspace / "fused.ply"
    final.write_text("mesh"); fused.write_text("cloud")
    (workspace / "mesh_report.json").write_text("{}")
    sparse = workspace / "colmap" / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    (sparse / "images.bin").write_text("1")
    return {
        "final_mesh": str(final), "fused_ply": str(fused), "dense_mesh": None,
        "sparse_model": str(sparse),
        "sparse": {"model_dir": str(sparse), "registered_ratio": 0.9},
        "dense": {"fused_points": 9000, "poisson_attempts": [{"depth": 10, "ok": True}]},
        "mesh": {"method_used": "poisson", "fallback_used": False, "faces": 5000,
                 "vertices": 2500, "has_vertex_colors": True, "steps": []},
        "quality": {"verdict": verdict, "faces": 5000, "warnings": [],
                    "errors": ["too few faces"] if verdict == "FAIL" else []},
        "timings": {"total_s": 1.0},
    }


def test_worker_success_publishes_outputs_and_stores_result(monkeypatch):
    from app.config import settings
    from app.jobs.job_repository import JobRepository
    from app.workers import worker_executor
    from app.workers.task_dispatcher import TaskDispatcher

    jid = _queued_job_with_files()
    seen = {}

    def fake_pipeline(workspace, on_stage=None):
        seen["images"] = sorted(p.name for p in (workspace / "colmap" / "images").iterdir())
        on_stage("sfm", 5)
        return _fake_outputs(workspace)

    monkeypatch.setattr(worker_executor, "run_photogrammetry_pipeline", fake_pipeline)
    TaskDispatcher.dispatch_job(jid)

    job = JobRepository.get_job(jid)
    assert job["status"] == "completed" and job["progress"] == 100
    assert len(seen["images"]) == 6
    out = settings.outputs_root / str(jid)
    for name in ("final_mesh.ply", "fused.ply", "quality.json", "mesh_report.json", "job.log"):
        assert (out / name).exists(), name
    assert (out / "sparse_model" / "images.bin").exists()
    assert job["result"]["quality"]["verdict"] == "PASS"
    assert job["result"]["sparse"]["registered_ratio"] == 0.9
    # the uploaded originals are untouched
    assert len(list((settings.uploads_root / str(jid) / "images").iterdir())) == 6


def test_worker_stores_pipeline_error_reason(monkeypatch):
    from app.jobs.errors import PipelineError
    from app.jobs.job_repository import JobRepository
    from app.workers import worker_executor
    from app.workers.task_dispatcher import TaskDispatcher

    jid = _queued_job_with_files()

    def failing(workspace, on_stage=None):
        raise PipelineError("Only 2 of 6 photos could be placed in 3D", "sparse_gate", "sfm")

    monkeypatch.setattr(worker_executor, "run_photogrammetry_pipeline", failing)
    TaskDispatcher.dispatch_job(jid)
    job = JobRepository.get_job(jid)
    assert job["status"] == "failed"
    assert job["error_code"] == "sparse_gate" and job["error_stage"] == "sfm"
    assert "Only 2 of 6" in job["error_message"]


def test_worker_unexpected_exception_is_recorded_not_lost(monkeypatch):
    from app.jobs.job_repository import JobRepository
    from app.workers import worker_executor
    from app.workers.task_dispatcher import TaskDispatcher

    jid = _queued_job_with_files()

    def crash(workspace, on_stage=None):
        raise ZeroDivisionError("oops")

    monkeypatch.setattr(worker_executor, "run_photogrammetry_pipeline", crash)
    TaskDispatcher.dispatch_job(jid)
    job = JobRepository.get_job(jid)
    assert job["status"] == "failed" and job["error_code"] == "internal_error"
    assert "ZeroDivisionError" in job["error_message"]


def test_failed_quality_verdict_fails_the_job_but_keeps_the_mesh(monkeypatch):
    from app.config import settings
    from app.jobs.job_repository import JobRepository
    from app.workers import worker_executor
    from app.workers.task_dispatcher import TaskDispatcher

    jid = _queued_job_with_files()
    monkeypatch.setattr(worker_executor, "run_photogrammetry_pipeline",
                        lambda ws, on_stage=None: _fake_outputs(ws, verdict="FAIL"))
    TaskDispatcher.dispatch_job(jid)
    job = JobRepository.get_job(jid)
    assert job["status"] == "failed" and job["error_code"] == "quality_failed"
    assert (settings.outputs_root / str(jid) / "final_mesh.ply").exists()


def test_missing_uploads_fail_with_a_clear_reason():
    from app.jobs.job_manager import JobManager
    from app.jobs.job_repository import JobRepository
    from app.workers.task_dispatcher import TaskDispatcher
    jid = _new_job(5)
    JobManager.mark_uploading(jid)
    JobManager.start_job(jid)
    TaskDispatcher.dispatch_job(jid)
    job = JobRepository.get_job(jid)
    assert job["status"] == "failed" and job["error_code"] == "missing_file"


def test_unsupported_method_fails_instead_of_hanging():
    from app.workers.worker_executor import WorkerExecutor
    jid = _new_job(method="method_2")
    WorkerExecutor.run_job(jid, "method_2")
    assert _status(jid) == "failed"


def test_dispatch_next_pending_job_runs_oldest_and_handles_empty_queue(monkeypatch):
    from app.workers import worker_executor
    from app.workers.task_dispatcher import TaskDispatcher
    assert TaskDispatcher.dispatch_next_pending_job() is None
    jid = _queued_job_with_files()
    monkeypatch.setattr(worker_executor, "run_photogrammetry_pipeline",
                        lambda ws, on_stage=None: _fake_outputs(ws))
    assert TaskDispatcher.dispatch_next_pending_job() == str(jid)
    assert _status(jid) == "completed"
    assert TaskDispatcher.dispatch_next_pending_job() is None


# ---------------------------------------------------------------- image checks
def _check(raw, **kw):
    from app.services.image_utils import check_image
    args = dict(display_name="a.png", allowed_formats={"JPEG", "PNG", "TIFF"},
                min_short_side=32, max_bytes=5 * 1024 * 1024)
    args.update(kw)
    return check_image(raw, **args)


def test_valid_image_is_accepted_and_hashed():
    c = _check(make_png(1))
    assert c.extension == ".png" and (c.width, c.height) == (64, 48)
    assert c.sha256 == _check(make_png(1)).sha256 != _check(make_png(2)).sha256


def test_fake_image_with_image_content_type_is_rejected():
    from app.services.image_utils import ImageRejected
    with pytest.raises(ImageRejected):
        _check(b"this is not an image at all")


def test_truncated_image_is_rejected():
    from app.services.image_utils import ImageRejected
    with pytest.raises(ImageRejected):
        _check(make_png(1)[:60])


def test_disallowed_format_too_small_empty_and_oversize_are_rejected():
    from app.services.image_utils import ImageRejected
    buf = io.BytesIO(); Image.new("RGB", (64, 48)).save(buf, "BMP")
    with pytest.raises(ImageRejected, match="not allowed"):
        _check(buf.getvalue())
    with pytest.raises(ImageRejected, match="too small"):
        _check(make_png(1), min_short_side=100)
    with pytest.raises(ImageRejected, match="empty"):
        _check(b"")
    with pytest.raises(ImageRejected, match="larger than"):
        _check(make_png(1), max_bytes=10)


def test_exif_orientation_is_baked_into_the_pixels():
    img = Image.new("RGB", (80, 40), (10, 200, 10))
    exif = Image.Exif(); exif[274] = 6                 # rotate 90 degrees
    buf = io.BytesIO(); img.save(buf, "JPEG", exif=exif)
    c = _check(buf.getvalue(), display_name="p.jpg")
    assert c.orientation_fixed and (c.width, c.height) == (40, 80)
    assert Image.open(io.BytesIO(c.data)).size == (40, 80)


# ---------------------------------------------------------------- COLMAP runner
def test_run_colmap_captures_output(tmp_path):
    from app.services.method1_photogrammetry.colmap_runner import run_colmap
    lines = run_colmap([sys.executable, "-c", "print('hello'); print('world')"],
                       tmp_path / "c.log", "demo", 30)
    assert lines == ["hello", "world"]
    assert "hello" in (tmp_path / "c.log").read_text()


def test_run_colmap_failure_reports_the_real_reason(tmp_path):
    from app.jobs.errors import PipelineError
    from app.services.method1_photogrammetry.colmap_runner import run_colmap
    code = "import sys; print('CUDA error: out of memory'); sys.exit(3)"
    with pytest.raises(PipelineError) as exc:
        run_colmap([sys.executable, "-c", code], tmp_path / "c.log", "patch_match_stereo", 30)
    assert exc.value.code == "colmap_failed" and exc.value.stage == "patch_match_stereo"
    assert "out of memory" in str(exc.value) and "exit code 3" in str(exc.value)


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX process groups in the test")
def test_run_colmap_kills_a_hung_process(tmp_path):
    from app.jobs.errors import PipelineError
    from app.services.method1_photogrammetry.colmap_runner import run_colmap
    t0 = time.time()
    with pytest.raises(PipelineError) as exc:
        run_colmap([sys.executable, "-c", "import time; time.sleep(60)"],
                   tmp_path / "c.log", "mapper", timeout=1)
    assert exc.value.code == "timeout" and time.time() - t0 < 15


def test_run_colmap_missing_binary_is_a_clear_error(tmp_path):
    from app.jobs.errors import PipelineError
    from app.services.method1_photogrammetry.colmap_runner import run_colmap
    with pytest.raises(PipelineError, match="Could not start COLMAP"):
        run_colmap([str(tmp_path / "nope")], tmp_path / "c.log", "x", 5)


def test_parse_analyzer_output():
    from app.services.method1_photogrammetry.colmap_runner import _parse_analyzer
    text = ["I0101 model_analyzer.cc:1 Registered images: 102", "Points: 54000",
            "Mean track length: 5.25", "Mean reprojection error: 0.61px"]
    assert _parse_analyzer(text) == {
        "registered_images": 102, "points": 54000,
        "mean_track_length": 5.25, "mean_reprojection_error": 0.61}


def test_model_dirs_sort_numerically(tmp_path):
    from app.services.method1_photogrammetry.colmap_runner import _model_dirs
    for n in ("0", "1", "2", "10"):
        (tmp_path / n).mkdir()
    assert [p.name for p in _model_dirs(tmp_path)] == ["0", "1", "2", "10"]


def test_option_names_follow_the_installed_colmap_version(monkeypatch):
    from app.services.method1_photogrammetry import colmap_runner as cr
    monkeypatch.setattr(cr, "_help_text", lambda exe, cmd: "--SiftExtraction.use_gpu arg")
    got = cr.pick_option(Path("colmap"), "feature_extractor",
                         ["FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"], 1)
    assert got == ["--SiftExtraction.use_gpu", "1"]
    monkeypatch.setattr(cr, "_help_text", lambda exe, cmd: "--FeatureExtraction.use_gpu arg")
    got = cr.pick_option(Path("colmap"), "feature_extractor",
                         ["FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"], 1)
    assert got == ["--FeatureExtraction.use_gpu", "1"]


def test_count_ply_vertices(tmp_path):
    from app.services.method1_photogrammetry.dense_reconstruction import count_ply_vertices
    ply = tmp_path / "a.ply"
    ply.write_text("ply\nformat ascii 1.0\nelement vertex 1234\nproperty float x\nend_header\n")
    assert count_ply_vertices(ply) == 1234


# A tiny stand-in for the colmap executable: lets us test model selection and the
# registered-image gate without COLMAP.
_FAKE_COLMAP = '''#!{py}
import os, pathlib, sys
a = sys.argv[1:]
def opt(n): return a[a.index(n) + 1] if n in a else None
if "-h" in a:
    print("--FeatureExtraction.use_gpu --FeatureMatching.use_gpu "
          "--FeatureExtraction.max_image_size --SiftExtraction.max_num_features")
    sys.exit(0)
cmd = a[0]
if cmd == "mapper":
    for i, c in enumerate(os.environ["FAKE_MODELS"].split(",")):
        d = pathlib.Path(opt("--output_path")) / str(i); d.mkdir(parents=True, exist_ok=True)
        (d / "images.bin").write_text(c)
elif cmd == "model_analyzer":
    c = int((pathlib.Path(opt("--path")) / "images.bin").read_text())
    print("I0101 x.cc:1] Registered images: %d" % c)
    print("Points: 1234"); print("Mean track length: 4.5"); print("Mean reprojection error: 0.7px")
elif cmd == "model_converter":
    c = int((pathlib.Path(opt("--input_path")) / "images.bin").read_text())
    out = pathlib.Path(opt("--output_path")); out.mkdir(parents=True, exist_ok=True)
    rows = ["# Image list"]
    for i in range(c):
        rows.append("%d 1 0 0 0 0 0 0 1 img%02d.png" % (i + 1, i))
        rows.append("" if i % 2 == 0 else "1 2 3 4")     # points lines may be empty
    (out / "images.txt").write_text("\\n".join(rows) + "\\n")
'''


def _sfm_setup(tmp_path, monkeypatch, models: str):
    from app.config import settings
    exe = tmp_path / "fake_colmap"
    exe.write_text(_FAKE_COLMAP.format(py=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(settings, "colmap_bin", exe)
    monkeypatch.setattr(settings, "skip_cuda_check", True)
    monkeypatch.setenv("FAKE_MODELS", models)
    ws = tmp_path / "ws"
    images = ws / "colmap" / "images"
    images.mkdir(parents=True)
    for i in range(10):
        (images / f"img{i:02d}.png").write_bytes(make_png(i))
    return images, ws


@pytest.mark.skipif(os.name == "nt", reason="fake executable is a POSIX script")
def test_sfm_picks_the_model_with_most_registered_images(tmp_path, monkeypatch):
    from app.services.method1_photogrammetry.colmap_runner import run_colmap_sfm
    images, ws = _sfm_setup(tmp_path, monkeypatch, "2,8")
    res = run_colmap_sfm(images, ws)
    assert Path(res["model_dir"]).name == "1"          # not simply the first folder
    assert (res["registered_images"], res["total_images"]) == (8, 10)
    assert res["registered_ratio"] == 0.8 and res["num_models"] == 2
    assert res["unregistered_images"] == ["img08.png", "img09.png"]
    assert res["mean_reprojection_error"] == 0.7 and res["warnings"] == []


@pytest.mark.skipif(os.name == "nt", reason="fake executable is a POSIX script")
def test_sfm_gate_stops_the_job_before_dense_work(tmp_path, monkeypatch):
    from app.jobs.errors import PipelineError
    from app.services.method1_photogrammetry.colmap_runner import run_colmap_sfm
    images, ws = _sfm_setup(tmp_path, monkeypatch, "2,2")
    with pytest.raises(PipelineError) as exc:
        run_colmap_sfm(images, ws)
    assert exc.value.code == "sparse_gate" and "2 of 10" in str(exc.value)


@pytest.mark.skipif(os.name == "nt", reason="fake executable is a POSIX script")
def test_sfm_warns_when_ratio_is_low_but_acceptable(tmp_path, monkeypatch):
    from app.services.method1_photogrammetry.colmap_runner import run_colmap_sfm
    images, ws = _sfm_setup(tmp_path, monkeypatch, "5")
    res = run_colmap_sfm(images, ws)
    assert res["registered_ratio"] == 0.5 and res["warnings"]


@pytest.mark.skipif(os.name == "nt", reason="fake executable is a POSIX script")
def test_sfm_does_not_reuse_a_stale_sparse_model(tmp_path, monkeypatch):
    from app.services.method1_photogrammetry.colmap_runner import run_colmap_sfm
    images, ws = _sfm_setup(tmp_path, monkeypatch, "8")
    stale = ws / "colmap" / "sparse" / "7"
    stale.mkdir(parents=True)
    (stale / "images.bin").write_text("10")            # would win if it were kept
    res = run_colmap_sfm(images, ws)
    assert Path(res["model_dir"]).name == "0" and not stale.exists()
