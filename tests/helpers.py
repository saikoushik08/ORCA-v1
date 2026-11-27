import io

from PIL import Image


def make_png(i: int = 0, size=(64, 48)) -> bytes:
    """A valid PNG that differs for every i (so duplicate detection does not merge them)."""
    img = Image.new("RGB", size, ((i * 37) % 256, (i * 91) % 256, (i * 53) % 256))
    img.putpixel((0, 0), (i % 256, (i // 256) % 256, 7))
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()
