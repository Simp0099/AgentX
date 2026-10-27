"""Pillow-based image validation and digesting."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from agent.config import get_config
from agent.tools.common import run_bounded


def inspect_image(path: str | Path, timeout: float | None = None):
    """Validate and inspect a local image without raising to the caller."""
    def work():
        cfg = get_config().get("forensics", {})
        file_path = Path(path)
        max_bytes = int(cfg.get("max_file_bytes", 50_000_000))
        max_pixels = int(cfg.get("max_image_pixels", 40_000_000))
        if not file_path.is_file():
            raise FileNotFoundError("image file does not exist")
        if file_path.stat().st_size > max_bytes:
            raise ValueError("image exceeds configured byte limit")
        with Image.open(file_path) as image:
            image.verify()
        with Image.open(file_path) as image:
            width, height = image.size
            if width * height > max_pixels:
                raise ValueError("image exceeds configured pixel limit")
            image_format, mode = image.format, image.mode
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        return {"valid": True, "path": str(file_path), "format": image_format, "width": width,
                "height": height, "mode": mode, "byte_size": file_path.stat().st_size,
                "sha256": digest}

    return run_bounded("inspect_image", work, timeout)
