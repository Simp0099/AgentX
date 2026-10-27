"""agent/tool_registry.py: approved forensic tool registry and argument validation.

The registry is the allowlist the agent is allowed to call. Each tool entry
carries:
    name        -- the approved tool name
    description -- one-line what the tool does
    input_schema -- JSON Schema for the tool's arguments (sent to the LLM)
    arg_schema  -- the typed python arg map used for pre-validation
    category    -- "baseline" (always useful) or "conditional" (only when evidence warrants)
    callable    -- the underlying tool implementation
    run         -- a typed dispatcher that validates arguments and returns a
                   ToolResult (the typed dict with status and errors)

No tool accepts arbitrary paths: when ``config["forensics"]["upload_root"]`` is
set, every path is resolved and must fall inside that directory. The Gradio app
writes every upload under ``artifacts/uploads``, so applying a root there keeps
the forensic tools confined to the upload directory while tests (which call the
tools directly with tmp files) run unrestricted.

Tools:
    inspect_image     -- validate/digest an image
    run_localizer     -- run the unchanged model inference (TamperingDetector.predict_raw)
    generate_ela      -- produce an ELA map in the managed artifact dir
    inspect_metadata  -- read image metadata / EXIF fields
    validate_prediction -- validate the localizer score and bbox
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Any

from agent.schemas import ToolResult
from agent.tools.compare_evidence import compare_evidence
from agent.tools.generate_ela import generate_ela
from agent.tools.inspect_image import inspect_image
from agent.tools.inspect_metadata import inspect_metadata
from agent.tools.run_localizer import run_localizer
from agent.tools.validate_prediction import validate_prediction

ToolCall = namedtuple("ToolCall", ["tool", "args"])

# ---------------------------------------------------------------------------
# Per-tool JSON input schemas (sent to the LLM). Derived from arg_schema.
# ---------------------------------------------------------------------------
def _input_schema(arg_schema: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for key, expected in arg_schema.items():
        if key == "quality":
            properties[key] = {"type": ["integer", "null"]}
        elif key == "record" or key == "run_result":
            properties[key] = {"type": "object"}
        elif isinstance(expected, tuple):
            # (int, float) -> number; (int, type(None)) -> integer|null
            if type(None) in expected:
                properties[key] = {"type": "integer", "nullable": True}
            elif int in expected:
                properties[key] = {"type": "integer"}
            else:
                properties[key] = {"type": "number"}
        else:
            properties[key] = {"type": type(expected).__name__}
        required.append(key)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


UPLOAD_ROOT_KEY = "upload_root"
DEFAULT_UPLOAD_ROOT = None  # unrestricted unless the Gradio app configures a root


def _upload_root(config: dict[str, Any] | None = None) -> Path | None:
    """Return the configured upload root, or None (unrestricted)."""
    cfg = config or {}
    return (
        Path(cfg.get("forensics", {}).get(UPLOAD_ROOT_KEY))
        if cfg.get("forensics", {}).get(UPLOAD_ROOT_KEY)
        else None
    )


def _resolve_within_root(path: Any, root: Path) -> tuple[bool, str]:
    """Return (ok, error). Path must be a local file inside ``root``."""
    if not isinstance(path, (str, Path)):
        return False, "path must be a string or Path"
    candidate = Path(path)
    try:
        candidate = candidate.resolve()
        root = root.resolve()
        candidate.relative_to(root)
    except (OSError, ValueError):
        return False, f"{candidate} is not inside the upload directory"
    return True, ""


def _make_dispatcher(name: str, info: dict[str, Any]):
    """Build the typed ``run()`` closure for one tool entry."""

    def run(
        config: dict[str, Any] | None = None,
        detector: Any | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        # 1. Validate argument mapping against the tool schema.
        valid, reason = validate_args(name, kwargs)
        if not valid:
            return ToolResult(name, "error", error=f"invalid arguments: {reason}")
        # 2. Path allowlist (no arbitrary paths outside the upload directory).
        root = _upload_root(config)
        if root is not None:
            allowed, err = _resolve_within_root(kwargs.get("path"), root)
            if not allowed:
                return ToolResult(name, "error", error=err)
        try:
            # Forward timeout (last positional param accepted by every tool fn).
            value = info["callable"](**kwargs, timeout=timeout)
        except Exception as exc:  # tool boundary must not leak exceptions
            return ToolResult(name, "error", error=str(exc))
        if not isinstance(value, ToolResult):
            value = ToolResult(name, "error", error="tool returned an invalid result")
        return value

    run.__name__ = f"run_{name}"
    return run


tool_registry: dict[str, dict[str, Any]] = {
    "inspect_image": {
        "name": "inspect_image",
        "description": "Inspect and validate a local image; returns digest, dims, and format.",
        "input_schema": _input_schema({"path": str}),
        "arg_schema": {"path": str},
        "category": "baseline",
        "callable": inspect_image,
    },
    "run_localizer": {
        "name": "run_localizer",
        "description": "Run the model's dual-head localizer; returns uncalibrated score and box.",
        "input_schema": _input_schema({"path": str, "threshold": (int, float)}),
        "arg_schema": {"path": str, "threshold": (int, float)},
        "category": "baseline",
        "callable": run_localizer,
    },
    "validate_prediction": {
        "name": "validate_prediction",
        "description": "Validate the localizer score and bounding box; return validity and issues.",
        "input_schema": _input_schema(
            {"run_result": dict, "image_width": int, "image_height": int}
        ),
        "arg_schema": {
            "run_result": dict,
            "image_width": int,
            "image_height": int,
        },
        "category": "baseline",
        "callable": validate_prediction,
    },
    "generate_ela": {
        "name": "generate_ela",
        "description": "Generate an ELA map for the image in the managed artifact directory.",
        "input_schema": _input_schema({"path": str, "quality": (int, type(None))}),
        "arg_schema": {"path": str, "quality": (int, type(None))},
        "category": "conditional",
        "callable": generate_ela,
    },
    "inspect_metadata": {
        "name": "inspect_metadata",
        "description": "Inspect configured metadata / EXIF fields for the image.",
        "input_schema": _input_schema({"path": str}),
        "arg_schema": {"path": str},
        "category": "conditional",
        "callable": inspect_metadata,
    },
    "compare_evidence": {
        "name": "compare_evidence",
        "description": "Compare recorded score, validation, ELA and metadata evidence.",
        "input_schema": _input_schema({"record": object}),
        "arg_schema": {"record": object},
        "category": "conditional",
        "callable": compare_evidence,
    },
}

# Fill in the typed per-tool ``run()`` dispatcher (kept after construction so
# the entries above stay readable). Tests only monkeypatch ``callable``.
for _name, _entry in tool_registry.items():
    _entry["run"] = _make_dispatcher(_name, _entry)

# Legacy look-up for the deterministic planner and the LLM dispatcher.
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
