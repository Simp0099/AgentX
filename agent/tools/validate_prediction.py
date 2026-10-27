"""Validate prediction score and normalized bounding box."""

from __future__ import annotations

import math

from agent.config import get_config
from agent.schemas import ToolResult
from agent.tools.common import run_bounded


def validate_prediction(
    raw_score: float | dict[str, object] | ToolResult | None = None,
    bbox: list[float] | tuple[float, ...] | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    timeout: float | None = None,
    run_result: dict[str, object] | ToolResult | None = None,
) -> ToolResult:
    """Validate a localizer result (or its expanded score/box fields)."""
    if run_result is None and (isinstance(raw_score, dict) or hasattr(raw_score, "result")):
        run_result, raw_score = raw_score, None
    if run_result is not None:
        if hasattr(run_result, "result"):
            payload = run_result.result or {}
            if run_result.status != "ok":
                raw_score = None
        else:
            payload = run_result
        if isinstance(payload, dict):
            raw_score = payload.get("tampered_prob", payload.get("raw_score", raw_score))
            bbox = payload.get("bbox", payload.get("bbox_raw", bbox))
            image_width = payload.get("image_width", image_width)
            image_height = payload.get("image_height", image_height)
            size = payload.get("image_size")
            if size and len(size) == 2:
                image_width = image_width or size[0]
                image_height = image_height or size[1]

    def work():
        cfg = get_config().get("forensics", {})
        issues = []
        if (
            raw_score is None
            or not isinstance(raw_score, (int, float))
            or not math.isfinite(raw_score)
        ):
            issues.append("score must be finite")
        elif not 0 <= raw_score <= 1:
            issues.append("score out of range")
        if bbox is not None:
            finite_coordinates = (
                all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox)
                if len(bbox) == 4
                else False
            )
            if not finite_coordinates:
                issues.append("box must contain four finite coordinates")
            else:
                xmin, ymin, xmax, ymax = bbox
                if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
                    issues.append("box coordinates out of range or unordered")
                else:
                    area = (xmax - xmin) * (ymax - ymin)
                    if area < float(cfg.get("min_box_area", 0.001)):
                        issues.append("box area below minimum")
                    if area > float(cfg.get("max_box_coverage", 0.95)):
                        issues.append("box coverage above maximum")
        if image_width is None or image_height is None or image_width <= 0 or image_height <= 0:
            issues.append("image dimensions must be positive")
        return {"valid": not issues, "issues": issues}

    return run_bounded("validate_prediction", work, timeout)
