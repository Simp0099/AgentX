"""Deterministic conditional investigation policy."""

from __future__ import annotations

from agent.config import get_config
from agent.schemas import InvestigationRecord
from agent.tool_registry import ToolCall


def _observations(record):
    return [obs for obs in record.observations if not isinstance(obs, list)]


def next_checks(record: InvestigationRecord) -> list[list[ToolCall]]:
    """Return the next unrun checks based only on recorded evidence."""
    obs = _observations(record)
    by_tool = {item.tool: item for item in obs}
    done = set(by_tool)
    if "inspect_image" not in done:
        return [[ToolCall("inspect_image", {"path": record.filename})]]
    inspected = by_tool["inspect_image"].result or {}
    if not inspected.get("valid", True):
        return []
    if "run_localizer" not in done:
        return [[ToolCall("run_localizer", {"path": record.filename,
                                             "threshold": record.threshold})]]
    localizer = by_tool["run_localizer"].result or {}
    score = localizer.get("tampered_prob")
    if score is None:
        return []
    cfg = get_config().get("forensics", {})
    low = float(cfg.get("uncertain_low", 0.4))
    high = float(cfg.get("uncertain_high", 0.6))
    uncertain = low <= score <= high
    flagged = score >= record.threshold
    goal = f"{record.goal} {record.goal_text}".lower()
    origin = any(word in goal for word in ("origin", "source", "camera", "editing", "history"))
    if (uncertain or flagged) and "generate_ela" not in done:
        return [[ToolCall("generate_ela", {"path": record.filename})]]
    if (uncertain or origin) and "inspect_metadata" not in done:
        return [[ToolCall("inspect_metadata", {"path": record.filename})]]
    if "validate_prediction" not in done:
        return [[ToolCall("validate_prediction", {
            "raw_score": score, "bbox": localizer.get("bbox"),
            "image_width": (by_tool["inspect_image"].result or {}).get("width"),
            "image_height": (by_tool["inspect_image"].result or {}).get("height"),
        })]]
    validation = by_tool["validate_prediction"].result or {}
    conflict = (flagged and not validation.get("valid", False)) or uncertain
    if conflict and "generate_ela" not in done:
        return [[ToolCall("generate_ela", {"path": record.filename})]]
    if conflict and "inspect_metadata" not in done:
        return [[ToolCall("inspect_metadata", {"path": record.filename})]]
    if conflict and "compare_evidence" not in done:
        return [[ToolCall("compare_evidence", {"record": record})]]
    return []
