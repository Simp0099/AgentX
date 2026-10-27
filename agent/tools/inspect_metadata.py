"""Inspect basic image metadata and EXIF presence with Pillow."""

from __future__ import annotations

from PIL import Image
from PIL.ExifTags import TAGS

from agent.config import get_config
from agent.schemas import ToolResult
from agent.tools.common import run_bounded


def inspect_metadata(path: str, timeout: float | None = None) -> ToolResult:
    def work():
        configured_fields = get_config()["forensics"]["metadata_fields"]
        with Image.open(path) as image:
            parse_errors = []
            try:
                exif = image.getexif()
                fields = {str(TAGS.get(key, key)): str(value) for key, value in exif.items()}
            except Exception as exc:  # malformed EXIF must not look like tampering
                fields = {}
                parse_errors.append(str(exc))
            absent_fields = [key for key in configured_fields if key not in fields]
            result = {
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "exif": fields,
                "present_fields": fields,
                "absent_fields": absent_fields,
                "parse_errors": parse_errors,
                "exif_status": (
                    "parse_error" if parse_errors else "present" if fields else "absent"
                ),
            }
        return result

    return run_bounded("inspect_metadata", work, timeout)
