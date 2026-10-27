"""Run the shared model inference API and return an uncalibrated score."""

from __future__ import annotations

from pathlib import Path

from agent.config import get_config
from agent.schemas import ToolResult
from agent.tools.common import run_bounded


def run_localizer(
    path: str | Path,
    threshold: float | None = None,
    detector: object | None = None,
    timeout: float | None = None,
) -> ToolResult:
    def work():
        active_detector = detector
        if active_detector is None:
            from src.inference import TamperingDetector

            cfg = get_config()
            checkpoint = cfg.get("output", {}).get("checkpoint_dir", "checkpoints/")
            name = cfg.get("output", {}).get("checkpoint_name", "best_localizer.keras")
            active_detector = TamperingDetector(f"{checkpoint.rstrip('/')}/{name}")
        raw = active_detector.predict_raw(path)
        score = float(raw["tampered_prob"])
        active_threshold = threshold
        if active_threshold is None:
            active_threshold = get_config()["output"]["threshold"]
        active_threshold = float(active_threshold)
        return {
            "tampered_prob": score,
            "label": "tampered" if score >= active_threshold else "not_tampered",
            "bbox": raw.get("bbox_raw"),
            "image_size": raw.get("image_size"),
            "model_version": raw.get("model_version"),
            "threshold": active_threshold,
        }

    return run_bounded("run_localizer", work, timeout)
