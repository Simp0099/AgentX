"""Gradio-independent upload handler; inference runs through the orchestrator."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from agent.orchestrator import run_investigation
from agent.rendering import escape_untrusted
from src.visualize import draw_boxes

UPLOAD_DIR = Path("artifacts/uploads")


def analyze(image, threshold, detector=None, goal="Quick check", custom_text=""):
    """Investigate one uploaded image and return UI outputs from its record."""
    if image is None:
        return None, None, None, "Upload an image to begin."
    supplied_name = os.path.basename(escape_untrusted(getattr(image, "filename", "")))
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    image = image.convert("RGB")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".png", dir=UPLOAD_DIR, delete=False) as tmp:
        image_path = Path(tmp.name)
    relative_path = image_path.relative_to(Path.cwd()).as_posix()
    try:
        image.save(image_path, format="PNG")
        record = run_investigation(
            relative_path,
            goal={
                "Quick check": "quick_check",
                "Detailed": "detailed",
                "Custom text": "custom",
            }.get(goal, "quick_check"),
            goal_text=custom_text[:2000] if goal == "Custom text" else "",
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
    validation = next(
        (obs.result or {} for obs in record.observations if obs.tool == "validate_prediction"),
        {},
    )
    bbox = localizer.get("bbox")
    is_flagged = record.verdict and record.verdict.value in {
        "Flagged for review",
        "Flagged, region unavailable",
    }
    overlay = draw_boxes(image, [bbox]) if is_flagged and validation.get("valid") and bbox else None
    ela_result = next(
        (obs.result or {} for obs in record.observations if obs.tool == "generate_ela"),
        None,
    )
    ela = None
    if ela_result and ela_result.get("artifact"):
        try:
            with Image.open(ela_result["artifact"]) as generated:
                ela = generated.copy()
        except (OSError, ValueError):
            ela = None
    scores = {"tampered score (uncalibrated)": score} if score is not None else None
    verdict = record.verdict.value if record.verdict else "Inconclusive"
    checks = [obs for obs in record.observations if obs.status != "rejected"]
    check_list = (
        "\n".join(
            f"- {escape_untrusted(obs.tool)} ({escape_untrusted(obs.status)})" for obs in checks
        )
        or "- none"
    )
    all_checks = (
        "inspect_image",
        "run_localizer",
        "validate_prediction",
        "generate_ela",
        "inspect_metadata",
        "compare_evidence",
    )
    executed = {obs.tool for obs in checks}
    not_run = [name for name in all_checks if name not in executed]
    not_run_text = "\n".join(f"- {escape_untrusted(name)}" for name in not_run) or "- none"
    errors = list(record.errors)
    errors.extend(
        f"{obs.tool}: {obs.rationale}"
        for obs in record.observations
        if obs.status in ("error", "timeout")
    )
    error_text = "\n".join(f"- {escape_untrusted(error)}" for error in errors) or "- none"
    score_text = f"{float(score):.4f}" if score is not None else "unavailable"
    detail = (
        f"### Verdict: **{escape_untrusted(verdict)}**\n\n"
        f"**tampered score (uncalibrated):** {score_text}\n\n"
        f"**Executed checks:**\n{check_list}\n\n"
        f"**Checks not run:**\n{not_run_text}\n\n**Errors:**\n{error_text}"
    )
    return ela, scores, overlay, detail
