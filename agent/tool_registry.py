"""agent/tool_registry.py: single source of allowed tool names.

Each entry maps a tool name to (callable, arg_schema, timeout, category).
Unknown names are rejected. Argument validation: types, ranges, allowed
enum values.  The orchestrator supplies the image path; the LLM never does.

ToolCall = namedtuple("ToolCall", ["tool", "args"])
"""

from __future__ import annotations

from collections import namedtuple

from agent.tools.compare_evidence import compare_evidence
from agent.tools.generate_ela import generate_ela
from agent.tools.inspect_image import inspect_image
from agent.tools.inspect_metadata import inspect_metadata
from agent.tools.run_localizer import run_localizer
from agent.tools.validate_prediction import validate_prediction

#: Mapping of allowed tool names to their callable and argument schema.
#: The orchestrator only ever executes tools from this registry.
tool_registry: dict = {
    "inspect_image": {
        "callable": inspect_image,
        "arg_schema": {
            "path": str,
        },
        "timeout": 30.0,
        "category": "baseline",
    },
    "run_localizer": {
        "callable": run_localizer,
        "arg_schema": {
            "path": str,
            "threshold": float,
        },
        "timeout": 60.0,
        "category": "baseline",
    },
    "validate_prediction": {
        "callable": validate_prediction,
        "arg_schema": {
            "raw_score": float,
            "bbox": tuple,
            "image_width": int,
            "image_height": int,
        },
        "timeout": 30.0,
        "category": "baseline",
    },
    "generate_ela": {
        "callable": generate_ela,
        "arg_schema": {
            "path": str,
            "quality": int,
        },
        "timeout": 30.0,
        "category": "policy",
    },
    "inspect_metadata": {
        "callable": inspect_metadata,
        "arg_schema": {
            "path": str,
        },
        "timeout": 30.0,
        "category": "baseline",
    },
    "compare_evidence": {
        "callable": compare_evidence,
        "arg_schema": {
            "score": float,
            "validation": dict,
            "ela": dict,
            "metadata": dict,
        },
        "timeout": 15.0,
        "category": "policy",
    },
}

ToolCall = namedtuple("ToolCall", ["tool", "args"])


def get(tool_name: str) -> dict | None:
    """Return the tool info dict for *tool_name*, or None if unknown."""
    return tool_registry.get(tool_name)
