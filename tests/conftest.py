import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# Isolated folders + fake Supabase BEFORE the app is imported.
_TMP = Path(tempfile.mkdtemp(prefix="orca_tests_"))
os.environ["UPLOADS_ROOT"] = str(_TMP / "uploads")
os.environ["WORKSPACE_ROOT"] = str(_TMP / "workspace")
os.environ["OUTPUTS_ROOT"] = str(_TMP / "outputs")
os.environ["SUPABASE_URL"] = "http://fake"
os.environ["SUPABASE_SERVICE_KEY"] = "fake"
os.environ["COLMAP_BIN"] = str(_TMP / "no_such_colmap.exe")  # never runs the real COLMAP
os.environ["EMBEDDED_WORKER"] = "false"
os.environ["MIN_IMAGE_SHORT_SIDE"] = "32"                    # test images are tiny
os.environ["SKIP_CUDA_CHECK"] = "true"
os.environ["SKIP_PREFLIGHT_OVERLAP"] = "true"                 # mock test images skip SIFT RANSAC

import fake_supabase  # noqa: E402

sys.modules["supabase"] = fake_supabase


@pytest.fixture(autouse=True)
def clean_state():
    fake_supabase.reset()
    yield


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def png_bytes():
    from helpers import make_png
    return make_png(0)
