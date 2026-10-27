"""Inspect basic image metadata and EXIF presence with Pillow."""

from __future__ import annotations

from PIL import Image

from agent.tools.common import run_bounded


def inspect_metadata(path, timeout=None):
    def work():
        with Image.open(path) as image:
            exif = image.getexif()
            fields = {str(image.getexif().get(key, key)): str(value)
                      for key, value in exif.items()}
            result = {"format": image.format, "width": image.width, "height": image.height,
                      "mode": image.mode, "exif": fields,
                      "exif_status": "present" if fields else "absent"}
        return result

    return run_bounded("inspect_metadata", work, timeout)
