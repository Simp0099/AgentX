"""Validate prediction score and normalized bounding box."""

from __future__ import annotations

import math

from agent.config import get_config
from agent.tools.common import run_bounded


def validate_prediction(raw_score, bbox, image_width, image_height, timeout=None):
    def work():
        cfg = get_config().get("forensics", {})
        issues = []
        if (raw_score is None or not isinstance(raw_score, (int, float))
                or not math.isfinite(raw_score)):
            issues.append("score must be finite")
        elif not 0 <= raw_score <= 1:
            issues.append("score out of range")
        if bbox is not None:
            finite_coordinates = all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in bbox
            ) if len(bbox) == 4 else False
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
