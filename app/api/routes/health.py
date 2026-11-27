import shutil

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.services.method1_photogrammetry.colmap_runner import has_cuda
from app.storage.supabase_client import supabase

router = APIRouter()

MIN_FREE_GB = 20


def _supabase_status() -> dict:
    try:
        res = supabase.table("jobs").select("job_id").limit(1).execute()
        return {"ok": True, "rows": len(res.data or [])}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200]}


@router.get("")
@router.get("/", include_in_schema=False)
def health():
    """Component health. Returns 503 if anything needed to run a job is missing."""
    colmap = settings.resolve_colmap()
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(settings.workspace_root).free / 1024 ** 3
    components = {
        "colmap": {"ok": bool(colmap and colmap.exists()), "path": str(colmap) if colmap else None},
        "cuda": {"ok": has_cuda()},
        "disk": {"ok": free_gb >= MIN_FREE_GB, "free_gb": round(free_gb, 1)},
        "supabase": _supabase_status(),
    }
    healthy = all(c["ok"] for c in components.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "healthy" if healthy else "unhealthy", "components": components},
    )


@router.get("/supabase")
def supabase_health():
    status = _supabase_status()
    if not status["ok"]:
        return JSONResponse(status_code=503, content={"supabase": "unreachable", **status})
    return {"supabase": "connected", "rows": status["rows"]}
