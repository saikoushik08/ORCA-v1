import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import health, jobs, upload
from app.config import settings
from app.jobs.errors import InvalidTransition, JobNotFound
from app.logging_config import setup_logging

setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

for folder in (settings.uploads_root, settings.workspace_root, settings.outputs_root):
    folder.mkdir(parents=True, exist_ok=True)


def _startup_checks() -> None:
    from app.services.method1_photogrammetry.colmap_runner import has_cuda

    colmap = settings.resolve_colmap()
    if colmap and colmap.exists():
        logger.info("COLMAP found: %s", colmap)
    else:
        logger.error("COLMAP NOT FOUND. Jobs will fail until COLMAP_BIN is set or colmap is on PATH.")
    if not has_cuda():
        logger.error("No NVIDIA GPU detected (nvidia-smi). Dense reconstruction will fail.")
    if settings.public_outputs:
        logger.warning("PUBLIC_OUTPUTS is on: /outputs serves every job's files without "
                       "authentication. Use GET /jobs/{id}/result/download and turn it off.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _startup_checks()
    worker_thread = stop_event = None
    if settings.embedded_worker:
        from app.workers.local_worker import LocalWorker
        worker_thread, stop_event = LocalWorker.start_in_background()
        logger.info("Embedded worker started")
    yield
    if stop_event:
        stop_event.set()


app = FastAPI(
    title=settings.app_title,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS: only the origins listed in CORS_ORIGINS; no credentials (no cookies are used).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(JobNotFound)
async def job_not_found_handler(request: Request, exc: JobNotFound):
    return JSONResponse(status_code=404, content={"detail": "Job not found"})


@app.exception_handler(InvalidTransition)
async def invalid_transition_handler(request: Request, exc: InvalidTransition):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


if settings.public_outputs:
    app.mount("/outputs", StaticFiles(directory=str(settings.outputs_root)), name="outputs")


@app.get("/")
def root():
    return {"status": "ORCA backend running"}


# Routers. Each router declares only its own relative paths; the prefix is set once here.
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(jobs.router)
