"""Logging with a job_id on every line and an optional per-job log file (F-15)."""
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_job_id: ContextVar[str] = ContextVar("job_id", default="-")

LOG_FORMAT = "%(asctime)s %(levelname)-7s [job=%(job_id)s] %(name)s: %(message)s"


class _JobIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = _job_id.get()
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure the 'app' logger once. Safe to call several times."""
    root = logging.getLogger("app")
    if getattr(root, "_orca_configured", False):
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(_JobIdFilter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False
    root._orca_configured = True  # type: ignore[attr-defined]


@contextmanager
def job_logging(job_id, log_file: Path):
    """Tag every log line with the job id and also write it to log_file."""
    token = _job_id.set(str(job_id))
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    file_handler.addFilter(_JobIdFilter())
    app_logger = logging.getLogger("app")
    app_logger.addHandler(file_handler)
    try:
        yield
    finally:
        app_logger.removeHandler(file_handler)
        file_handler.close()
        _job_id.reset(token)


def tail_file(path: Path, n: int = 200) -> list:
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]
