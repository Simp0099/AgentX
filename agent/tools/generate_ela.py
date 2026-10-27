"""Generate an ELA image in the configured managed artifact directory."""

from __future__ import annotations

import uuid
from pathlib import Path

from PIL import Image

from agent.config import get_config
from agent.schemas import ToolResult
from agent.tools.common import run_bounded
from src.data.ela import compute_ela


def generate_ela(
    path: str | Path, quality: int | None = None, timeout: float | None = None
) -> ToolResult:
    def work():
        cfg = get_config()
        data_cfg = cfg.get("data", {})
        quality_value = int(quality if quality is not None else data_cfg.get("ela_quality", 90))
        output_dir = Path(cfg.get("forensics", {}).get("ela_temp_dir", "artifacts/ela"))
        output_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            ela = compute_ela(image, quality=quality_value)
        output_path = output_dir / f"{uuid.uuid4().hex}.png"
        ela.save(output_path, format="PNG")
        return {
            "artifact": str(output_path),
            "width": ela.width,
            "height": ela.height,
            "quality": quality_value,
        }

    return run_bounded("generate_ela", work, timeout)
