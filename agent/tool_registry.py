"""Approved forensic tool registry and argument validation."""

from __future__ import annotations

from collections import namedtuple
from typing import Any

from agent.tools.compare_evidence import compare_evidence
from agent.tools.generate_ela import generate_ela
from agent.tools.inspect_image import inspect_image
from agent.tools.inspect_metadata import inspect_metadata
from agent.tools.run_localizer import run_localizer
from agent.tools.validate_prediction import validate_prediction

ToolCall = namedtuple("ToolCall", ["tool", "args"])

tool_registry: dict[str, dict[str, Any]] = {
    "inspect_image": {
        "callable": inspect_image,
        "arg_schema": {"path": str},
        "category": "baseline",
    },
    "run_localizer": {
        "callable": run_localizer,
        "arg_schema": {"path": str, "threshold": (int, float)},
        "category": "baseline",
    },
    "validate_prediction": {
        "callable": validate_prediction,
        "arg_schema": {"run_result": dict, "image_width": int, "image_height": int},
        "category": "baseline",
    },
    "generate_ela": {
        "callable": generate_ela,
        "arg_schema": {"path": str, "quality": (int, type(None))},
        "category": "conditional",
    },
    "inspect_metadata": {
        "callable": inspect_metadata,
        "arg_schema": {"path": str},
        "category": "conditional",
    },
    "compare_evidence": {
        "callable": compare_evidence,
        "arg_schema": {"record": object},
        "category": "conditional",
    },
}

PLANNER_TOOLS: dict[str, dict[str, Any]] = {
    "generate_ela": {
        "description": "Generate ELA evidence for this investigation",
        "input_schema": {
            "type": "object",
            "properties": {"quality": {"type": "integer"}},
            "additionalProperties": False,
        },
    },
    "inspect_metadata": {
        "description": "Inspect configured metadata fields for this image",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "compare_evidence": {
        "description": "Compare evidence already present in this investigation",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}


def validate_args(tool_name: str, args: object) -> tuple[bool, str]:
    """Validate an exact argument mapping before dispatch."""
    info = tool_registry.get(tool_name)
    if info is None:
        return False, "unapproved tool name"
    if not isinstance(args, dict):
        return False, "arguments must be an object"
    schema = info["arg_schema"]
    required = {
        "inspect_image": {"path"},
        "run_localizer": {"path", "threshold"},
        "validate_prediction": {"run_result", "image_width", "image_height"},
        "generate_ela": {"path"},
        "inspect_metadata": {"path"},
        "compare_evidence": {"record"},
    }[tool_name]
    if not required.issubset(args) or not set(args).issubset(schema):
        return False, "arguments do not match the tool schema"
    for key, value in args.items():
        expected = schema[key]
        if expected is object:
            continue
        if isinstance(expected, tuple):
            valid = isinstance(value, expected)
        else:
            valid = isinstance(value, expected)
        if not valid or (expected in (int, float) and isinstance(value, bool)):
            return False, f"invalid argument type: {key}"
    if tool_name == "generate_ela" and args.get("quality") is not None:
        if not 1 <= args["quality"] <= 100:
            return False, "quality must be between 1 and 100"
    if tool_name == "run_localizer" and not 0 <= args["threshold"] <= 1:
        return False, "threshold must be between 0 and 1"
    return True, ""


def get(tool_name: str) -> dict | None:
    """Return the tool information for an approved name."""
    return tool_registry.get(tool_name)
