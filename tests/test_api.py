"""API behaviour: hardened upload, start rules, result endpoints, health."""
import io
from uuid import uuid4

from PIL import Image

from fake_supabase import DB
from helpers import make_png


def _upload(client, jid, items):
    return client.post(f"/upload/image/{jid}",
                       files=[("files", (n, b, "image/png")) for n, b in items])


def _job(client):
    return client.post("/jobs/").json()["job_id"]


# ---- upload hardening -------------------------------------------------------
def test_upload_ignores_client_filename_path_traversal(client):
    from app.config import settings
    jid = _job(client)
    r = _upload(client, jid, [("../../evil.png", make_png(1))])
    assert r.status_code == 200
    stored = list((settings.uploads_root / jid / "images").iterdir())
    assert len(stored) == 1 and stored[0].parent.name == "images"
    assert not (settings.uploads_root.parent / "evil.png").exists()
    assert len(stored[0].stem) == 32                 # server-chosen uuid name


def test_upload_response_does_not_leak_server_paths(client):
    jid = _job(client)
    text = _upload(client, jid, [("a.png", make_png(1))]).text
    assert "local_path" not in text and "local_images_dir" not in text and "uploads" not in text


def test_fake_image_is_rejected_and_nothing_is_saved(client):
    jid = _job(client)
    r = _upload(client, jid, [("good.png", make_png(1)), ("bad.png", b"not an image")])
    assert r.status_code == 400
    assert "bad.png" in r.text
    assert DB["images"] == []                         # all-or-nothing
    assert client.get(f"/jobs/{jid}").json()["status"] == "created"


def test_duplicates_are_skipped_and_reported(client):
    jid = _job(client)
    r = _upload(client, jid, [("a.png", make_png(1)), ("b.png", make_png(1)), ("c.png", make_png(2))])
    body = r.json()
    assert body["uploaded_count"] == 2 and body["skipped_duplicates"] == ["b.png"]
    again = _upload(client, jid, [("a-again.png", make_png(1))]).json()
    assert again["uploaded_count"] == 0 and again["total_images"] == 2


def test_limit_check_happens_before_the_job_changes_state(client, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "max_images_per_job", 2)
    jid = _job(client)
    r = _upload(client, jid, [(f"{i}.png", make_png(i)) for i in range(4)])
    assert r.status_code == 400
    assert client.get(f"/jobs/{jid}").json()["status"] == "created"


def test_exif_rotated_jpeg_is_normalised(client):
    img = Image.new("RGB", (80, 40), (5, 90, 200))
    exif = Image.Exif(); exif[274] = 6
    buf = io.BytesIO(); img.save(buf, "JPEG", exif=exif)
    jid = _job(client)
    r = client.post(f"/upload/image/{jid}",
                    files=[("files", ("p.jpg", buf.getvalue(), "image/jpeg"))])
    assert r.status_code == 200
    assert r.json()["uploaded_images"][0]["orientation_fixed"] is True


def test_cloud_failure_does_not_lose_the_image(client, monkeypatch):
    import fake_supabase

    def boom(self, path, data, file_options=None):
        raise ConnectionError("cloud down")
    monkeypatch.setattr(fake_supabase.Bucket, "upload", boom)
    monkeypatch.setattr("time.sleep", lambda s: None)
    jid = _job(client)
    r = _upload(client, jid, [("a.png", make_png(1))])
    assert r.status_code == 200 and r.json()["total_images"] == 1
    assert len(DB["images"]) == 1


# ---- start rules -----------------------------------------------------------
def test_job_without_images_cannot_start(client):
    jid = _job(client)
    assert client.post(f"/jobs/{jid}/start").status_code == 400


def test_second_start_is_rejected(client):
    jid = _job(client)
    _upload(client, jid, [(f"{i}.png", make_png(i)) for i in range(5)])
    assert client.post(f"/jobs/{jid}/start").status_code == 202
    assert client.post(f"/jobs/{jid}/start").status_code in (400, 409)
    assert len([j for j in DB["jobs"] if j["status"] == "pending"]) == 1


def test_upload_after_start_is_locked(client):
    jid = _job(client)
    _upload(client, jid, [(f"{i}.png", make_png(i)) for i in range(5)])
    client.post(f"/jobs/{jid}/start")
    assert _upload(client, jid, [("late.png", make_png(99))]).status_code == 400


# ---- status / results --------------------------------------------------------
def test_status_has_progress_fields(client):
    body = client.get(f"/jobs/{_job(client)}").json()
    for key in ("job_id", "status", "stage", "progress", "images", "error", "quality"):
        assert key in body


def test_result_not_available_before_completion(client):
    jid = _job(client)
    assert client.get(f"/jobs/{jid}/result").status_code == 409
    assert client.get(f"/jobs/{jid}/quality").status_code == 404


def test_result_download_for_completed_job(client):
    from app.config import settings
    from app.jobs.job_manager import JobManager
    jid = _job(client)
    _upload(client, jid, [(f"{i}.png", make_png(i)) for i in range(5)])
    JobManager.start_job(jid)
    JobManager.claim_for_processing(jid)
    out = settings.outputs_root / jid
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_mesh.ply").write_bytes(b"ply")
    (out / "quality.json").write_text('{"verdict": "PASS"}')
    JobManager.mark_completed(jid, result={"quality": {"verdict": "PASS"}})

    res = client.get(f"/jobs/{jid}/result").json()
    assert res["mesh_url"].endswith("file=final_mesh.ply")
    assert client.get(res["mesh_url"]).content == b"ply"
    assert client.get(f"/jobs/{jid}/quality").json()["verdict"] == "PASS"
    assert client.get(f"/jobs/{jid}").json()["quality"]["verdict"] == "PASS"


def test_download_only_allows_whitelisted_files(client):
    jid = _job(client)
    for bad in ("../../.env", "..%2F..%2Fmain.py", "something.txt"):
        assert client.get(f"/jobs/{jid}/result/download", params={"file": bad}).status_code == 400


def test_failed_job_status_shows_reason(client):
    from app.jobs.job_manager import JobManager
    jid = _job(client)
    JobManager.mark_failed(jid, "Only 2 of 10 photos registered",
                           stage="sfm", code="sparse_gate")
    err = client.get(f"/jobs/{jid}").json()["error"]
    assert err == {"message": "Only 2 of 10 photos registered", "code": "sparse_gate", "stage": "sfm"}


# ---- health ----------------------------------------------------------------
def test_health_reports_components_instead_of_crashing(client):
    r = client.get("/health")
    assert r.status_code == 503                       # no COLMAP / GPU in the test environment
    comps = r.json()["components"]
    assert comps["colmap"]["ok"] is False and comps["supabase"]["ok"] is True
