"""Tests for Batch 1 fixes: F-01, F-02, F-03, F-06, F-07, F-12, F-15."""
import re
from pathlib import Path
from uuid import uuid4

from helpers import make_png

APP = Path(__file__).resolve().parents[1] / "app"


def _files(client, png, n=5):
    # every image differs, otherwise duplicate detection would merge them
    return [("files", (f"im{i}.png", make_png(i), "image/png")) for i in range(n)]


# ---- F-01 upload prefix ---------------------------------------------------
def test_upload_path_has_single_prefix(client, png_bytes):
    jid = client.post("/jobs/").json()["job_id"]
    assert client.post(f"/upload/image/{jid}", files=_files(client, png_bytes)).status_code == 200
    assert client.post(f"/upload/upload/image/{jid}", files=_files(client, png_bytes)).status_code == 404
    paths = client.get("/openapi.json").json()["paths"]
    assert "/upload/image/{job_id}" in paths
    assert not any(p.startswith("/upload/upload") for p in paths)


# ---- F-02 one configuration ----------------------------------------------
def test_no_hardcoded_windows_path_outside_config():
    offenders = []
    for py in APP.rglob("*.py"):
        if py.name == "config.py":
            continue
        if re.search(r"[A-Za-z]:\\\\|r\"[A-Za-z]:\\", py.read_text(encoding="utf-8")):
            offenders.append(py.name)
    assert offenders == []


def test_image_limits_defined_once():
    from app.config import settings
    assert settings.min_images_per_job == 5 and settings.max_images_per_job == 200
    assert not (APP / "constants.py").exists()
    for py in APP.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert not re.search(r"^(MIN|MAX)_IMAGES\w*\s*=", text, re.M), py.name


def test_upload_limit_uses_config(client, png_bytes, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "max_images_per_job", 3)
    jid = client.post("/jobs/").json()["job_id"]
    r = client.post(f"/upload/image/{jid}", files=_files(client, png_bytes, 5))
    assert r.status_code == 400 and "limit" in r.text.lower()


def test_start_requires_min_images_from_config(client, png_bytes):
    jid = client.post("/jobs/").json()["job_id"]
    client.post(f"/upload/image/{jid}", files=_files(client, png_bytes, 2))
    r = client.post(f"/jobs/{jid}/start")
    assert r.status_code == 400 and "At least 5" in r.text


def test_missing_colmap_gives_clear_message():
    import pytest
    from app.services.method1_photogrammetry.colmap_runner import get_colmap_exe
    with pytest.raises(FileNotFoundError, match="COLMAP_BIN"):
        get_colmap_exe()


# ---- F-03 404 -------------------------------------------------------------
def test_unknown_job_is_404(client):
    unknown = str(uuid4())
    assert client.get(f"/jobs/{unknown}").status_code == 404
    assert client.get(f"/jobs/{unknown}/logs").status_code == 404
    assert client.post(f"/jobs/{unknown}/start").status_code == 404


def test_upload_to_unknown_job_is_404(client, png_bytes):
    r = client.post(f"/upload/image/{uuid4()}", files=_files(client, png_bytes))
    assert r.status_code == 404


def test_get_job_returns_none_when_missing():
    from app.jobs.job_repository import JobRepository
    assert JobRepository.get_job(uuid4()) is None


def test_job_not_found_is_a_valueerror():
    from app.jobs.errors import JobNotFound
    assert issubclass(JobNotFound, ValueError)


# ---- F-06 admin route -----------------------------------------------------
def test_status_admin_route_removed(client):
    jid = client.post("/jobs/").json()["job_id"]
    r = client.put(f"/jobs/{jid}/status", params={"action": "completed"})
    assert r.status_code in (404, 405)
    assert client.get(f"/jobs/{jid}").json()["status"] == "created"


# ---- F-07 method validation ----------------------------------------------
def test_invalid_method_rejected(client):
    assert client.post("/jobs/?method=banana").status_code == 422


def test_method2_returns_501_and_creates_no_job(client):
    from fake_supabase import DB
    r = client.post("/jobs/?method=method_2")
    assert r.status_code == 501
    assert DB["jobs"] == []


def test_method1_default_still_works(client):
    r = client.post("/jobs/")
    assert r.status_code == 200 and r.json()["method"] == "method_1"


def test_method2_package_removed():
    assert not (APP / "services" / "method2").exists()


def test_worker_fails_unknown_method_instead_of_hanging():
    from app.jobs.job_repository import JobRepository
    from app.workers.worker_executor import WorkerExecutor
    jid = uuid4()
    JobRepository.create_job(jid, "method_2")
    WorkerExecutor.run_job(jid, "method_2")
    assert JobRepository.get_job(jid)["status"] == "failed"


# ---- F-12 CORS ------------------------------------------------------------
def test_cors_only_listed_origins_and_no_credentials(client):
    ok = client.options("/jobs/", headers={"Origin": "http://localhost:5173",
                                             "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-credentials" not in ok.headers
    bad = client.options("/jobs/", headers={"Origin": "http://evil.example",
                                              "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


# ---- F-15 logging + F-10 non-blocking start ---------------------------------
def test_start_is_non_blocking_and_job_log_explains_failure(client, png_bytes):
    from app.workers.task_dispatcher import TaskDispatcher
    jid = client.post("/jobs/").json()["job_id"]
    client.post(f"/upload/image/{jid}", files=_files(client, png_bytes, 5))

    r = client.post(f"/jobs/{jid}/start")
    assert r.status_code == 202                       # returns at once, nothing ran yet
    assert r.json()["status"] == "pending"
    assert client.get(f"/jobs/{jid}").json()["status"] == "pending"

    TaskDispatcher.dispatch_next_pending_job()        # the worker; COLMAP is missing here
    status = client.get(f"/jobs/{jid}").json()
    assert status["status"] == "failed"
    assert "COLMAP" in status["error"]["message"]     # the reason is stored and returned
    lines = client.get(f"/jobs/{jid}/logs?tail=50").json()["lines"]
    assert any(f"job={jid}" in l for l in lines)
    assert any("COLMAP executable not found" in l for l in lines)


def test_no_print_in_app_code():
    offenders = [py.name for py in APP.rglob("*.py") if re.search(r"^\s*print\(", py.read_text(encoding="utf-8"), re.M)]
    assert offenders == []


# ---- dispatcher fix -------------------------------------------------------
def test_dispatch_next_pending_job_works(client):
    from app.jobs.job_repository import JobRepository
    from app.jobs.job_states import JobStatus
    from app.workers.task_dispatcher import TaskDispatcher
    assert TaskDispatcher.dispatch_next_pending_job() is None
    jid = uuid4()
    JobRepository.create_job(jid, "method_1")
    JobRepository.update_status(jid, JobStatus.PENDING)
    assert TaskDispatcher.dispatch_next_pending_job() == str(jid)
