"""Gradio-independent upload handler; inference runs through the orchestrator."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from agent.orchestrator import run_investigation
from agent.rendering import escape_untrusted
from src.data.ela import compute_ela
from src.visualize import draw_boxes

UPLOAD_DIR = Path("artifacts/uploads")


def analyze(image, threshold, detector=None):
    """Investigate one uploaded image and return UI outputs from its record."""
    if image is None:
        return None, None, None, "Upload an image to begin."
    supplied_name = os.path.basename(escape_untrusted(getattr(image, "filename", "")))
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    image = image.convert("RGB")
    ela = compute_ela(image)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".png", dir=UPLOAD_DIR, delete=False) as tmp:
        image_path = Path(tmp.name)
    relative_path = image_path.relative_to(Path.cwd()).as_posix()
    try:
        image.save(image_path, format="PNG")
        record = run_investigation(
            relative_path,
            goal="quick_check",
            config={"threshold": float(threshold), "max_tool_calls_per_image": 10},
            detector=detector,
        )
    finally:
        image_path.unlink(missing_ok=True)

    if supplied_name:
        record.filename = supplied_name
    localizer = next(
        (obs.result or {} for obs in record.observations if obs.tool == "run_localizer"),
        {},
    )
    score = localizer.get("tampered_prob")
    bbox = localizer.get("bbox")
    overlay = draw_boxes(image, [bbox]) if bbox else image
    scores = {"tampered score (uncalibrated)": score} if score is not None else None
    verdict = record.verdict.value if record.verdict else "Inconclusive"
    checks = [obs.tool for obs in record.observations]
    check_list = "\n".join(f"- {escape_untrusted(name)}" for name in checks) or "- none"
    score_text = f"{float(score):.4f}" if score is not None else "unavailable"
    detail = (
        f"### Verdict: **{escape_untrusted(verdict)}**\n\n"
        f"**tampered score (uncalibrated):** {score_text}\n\n"
        f"**Executed checks:**\n{check_list}"
    )
    return ela, scores, overlay, detail
