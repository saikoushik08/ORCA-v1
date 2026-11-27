"""Power management utilities for long-running compute jobs on Windows.

Prevents the operating system from entering Modern Standby, sleep mode,
turning off the display, or downclocking the GPU into power-saving states
during active photogrammetry and COLMAP reconstruction pipelines.
"""
import ctypes
import logging
import sys
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Windows SetThreadExecutionState flags:
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_AWAYMODE_REQUIRED = 0x00000040


@contextmanager
def keep_awake(display: bool = True):
    """Context manager to prevent Windows sleep, screen turn-off, and GPU throttling."""
    if sys.platform == "win32":
        try:
            flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            if display:
                flags |= ES_DISPLAY_REQUIRED
            res = ctypes.windll.kernel32.SetThreadExecutionState(flags)
            if res == 0:
                logger.warning("[Power] Failed to set SetThreadExecutionState to keep awake")
            else:
                logger.info("[Power] System sleep and screen timeout prevented during reconstruction")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Power] Could not set Windows execution state: %s", exc)

        try:
            yield
        finally:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                logger.info("[Power] System power state restored to normal")
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Power] Could not reset Windows execution state: %s", exc)
    else:
        yield
