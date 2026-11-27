"""Image validation shared by the upload endpoint.

A file is accepted only if Pillow can really decode it, its format is on the
whitelist and it is large enough for structure-from-motion. The stored name is
never taken from the client (no path traversal, no overwrites).
"""
import hashlib
import io
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageOps, UnidentifiedImageError

FORMAT_EXTENSION = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif"}


class ImageRejected(ValueError):
    """The file cannot be used; the message is safe to show to the user."""


@dataclass
class CheckedImage:
    data: bytes              # bytes to store (re-encoded only if EXIF rotation was needed)
    extension: str           # '.jpg' / '.png' / '.tif'
    width: int
    height: int
    sha256: str
    size_bytes: int
    camera_key: str          # make|model|WxH, used to detect mixed cameras
    orientation_fixed: bool


def _camera_key(img: Image.Image, width: int, height: int) -> str:
    try:
        exif = img.getexif()
        make = str(exif.get(271, "")).strip()
        model = str(exif.get(272, "")).strip()
    except Exception:  # noqa: BLE001
        make = model = ""
    return f"{make}|{model}|{width}x{height}"


def check_image(raw: bytes, display_name: str, allowed_formats: set,
                min_short_side: int, max_bytes: int) -> CheckedImage:
    if not raw:
        raise ImageRejected(f"{display_name}: file is empty")
    if len(raw) > max_bytes:
        raise ImageRejected(
            f"{display_name}: larger than {max_bytes // (1024 * 1024)} MB"
        )

    try:
        probe = Image.open(io.BytesIO(raw))
        fmt = (probe.format or "").upper()
        if fmt == "MPO":
            fmt = "JPEG"
        probe.verify()                      # structural check
        img = Image.open(io.BytesIO(raw))   # verify() invalidates the object
        img.load()                          # full decode: catches truncated files
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageRejected(
            f"{display_name}: not a readable image ({type(exc).__name__}). "
            "HEIC/RAW files are not supported; export as JPEG or PNG."
        )

    if fmt not in allowed_formats:
        raise ImageRejected(
            f"{display_name}: format {fmt or 'unknown'} is not allowed "
            f"(allowed: {', '.join(sorted(allowed_formats))})"
        )

    data = raw
    orientation_fixed = False
    try:
        orientation = img.getexif().get(274, 1)
    except Exception:  # noqa: BLE001
        orientation = 1
    if orientation not in (None, 1):
        # Bake the rotation into the pixels so portrait/landscape shots agree.
        fixed = ImageOps.exif_transpose(img)
        buf = io.BytesIO()
        if fmt == "JPEG":
            fixed.convert("RGB").save(buf, "JPEG", quality=95, subsampling=0)
        else:
            fixed.save(buf, fmt)
        data = buf.getvalue()
        img = fixed
        orientation_fixed = True

    width, height = img.size
    if min(width, height) < min_short_side:
        raise ImageRejected(
            f"{display_name}: {width}x{height} is too small "
            f"(shorter side must be at least {min_short_side}px)"
        )

    return CheckedImage(
        data=data,
        extension=FORMAT_EXTENSION[fmt],
        width=width,
        height=height,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        camera_key=_camera_key(img, width, height),
        orientation_fixed=orientation_fixed,
    )


def read_image_size(path) -> Optional[tuple]:
    """(width, height) without decoding the whole image, or None if unreadable."""
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:  # noqa: BLE001
        return None