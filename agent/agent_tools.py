"""agent/agent_tools.py: the model-facing tool API.

The LLM never sees a Python object. Every tool here takes only simple JSON
types (str, float, int) and returns a plain dict, so ``json.dumps`` always
succeeds. State that a tool needs but the model must not supply -- the detector,
the upload directory, results produced earlier in the same investigation -- is
carried in a :class:`ToolContext` injected by trusted code through
:func:`run_tool`.

This module sits *on top of* ``agent/tool_registry.py`` and
``agent/tools/*``; it does not replace them. The deterministic pipeline keeps
using the registry directly.

Design rules enforced here:

* ``run_tool`` never raises. Failures come back as ``{"error": ..., "hint": ...}``.
* Every success payload is JSON-safe: floats rounded to 4 places, tuples become
  lists, numpy scalars/arrays are converted, no PIL objects.
* Paths stay confined to ``ctx.upload_root`` when one is configured, matching
  the allowlist behaviour of ``agent/tool_registry.py``.

The detector is ELA-based, trained on CASIA v2 splices, and emits an
*uncalibrated* score rather than a probability. It cannot see fully AI-generated
images or heavily re-compressed screenshots. The tool descriptions say so, so
the model does not over-claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.schemas import ToolResult
from agent.tools.compare_evidence import compare_evidence
from agent.tools.generate_ela import generate_ela
from agent.tools.inspect_image import inspect_image
from agent.tools.inspect_metadata import inspect_metadata
from agent.tools.run_localizer import run_localizer
from agent.tools.validate_prediction import validate_prediction

# ---------------------------------------------------------------------------
# Context: everything the model must NOT have to supply
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Per-investigation state injected by trusted code.

    Attributes:
        detector: an object exposing ``predict_raw(path)``; injected by the app
            so the model can never choose or replace the model.
        upload_root: when set, every path argument must resolve inside it.
        threshold: decision threshold used when a tool accepts one.
        results: successful payloads recorded so far, keyed by tool name, so
            later tools can read earlier evidence instead of being handed
            whole objects by the model.
    """

    detector: Any | None = None
    upload_root: str | Path | None = None
    threshold: float = 0.5
    results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def remember(self, tool_name: str, payload: dict[str, Any]) -> None:
        """Record a successful tool payload for later tools in this run."""
        self.results[tool_name] = payload

    def recall(self, tool_name: str) -> dict[str, Any] | None:
        """Return a previously recorded payload, or None."""
        return self.results.get(tool_name)


class ToolError(Exception):
    """Internal control flow carrying an error payload. Never escapes run_tool."""

    def __init__(self, error: str, hint: str) -> None:
        super().__init__(error)
        self.payload = {"error": error, "hint": hint}


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Convert ``value`` to something ``json.dumps`` accepts.

    Floats are rounded to 4 decimal places; non-finite floats become ``None``
    (JSON has no NaN/Infinity); numpy arrays and scalars, tuples, sets and
    anything else unexpected are converted or stringified rather than allowed
    to break serialisation.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return float(round(float(value), 4)) if math.isfinite(float(value)) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):  # numpy arrays
        return _json_safe(value.tolist())
    if hasattr(value, "item"):  # numpy scalars
        return _json_safe(value.item())
    return str(value)


def _as_payload(result: ToolResult, hint: str) -> dict[str, Any]:
    """Turn a ToolResult into a success payload or an error dict."""
    if result.status != "ok":
        return {"error": result.error or f"tool ended in {result.status}", "hint": hint}
    return _json_safe(result.result or {})


# ---------------------------------------------------------------------------
# Path guard (mirrors agent/tool_registry.py's upload-root allowlist)
# ---------------------------------------------------------------------------


def _resolve_image(ctx: ToolContext, image_path: Any) -> Path:
    """Validate and return the image path, or raise :class:`ToolError`."""
    if not isinstance(image_path, str) or not image_path.strip():
        raise ToolError(
            "image_path must be a non-empty string",
            "pass the absolute path of an uploaded image",
        )
    path = Path(image_path).expanduser()
    if ctx.upload_root is not None:
        root = Path(ctx.upload_root).expanduser()
        try:
            path.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            raise ToolError(
                f"{image_path} is outside the allowed upload directory",
                f"pass a path inside {root}",
            ) from None
    if not path.is_file():
        raise ToolError(
            f"image file not found: {image_path}",
            "check the path exists and is readable before calling this tool",
        )
    return path


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _inspect_image(ctx: ToolContext, image_path: str) -> dict[str, Any]:
    """Validate an image and return its digest, dimensions and format."""
    path = _resolve_image(ctx, image_path)
    result = inspect_image(path)
    return _as_payload(
        result,
        "check the file is a readable PNG/JPEG inside the upload directory "
        "and within the configured byte and pixel limits",
    )


def _run_localizer(
    ctx: ToolContext, image_path: str, threshold: float | None = None
) -> dict[str, Any]:
    """Run the ELA-based localizer and return an uncalibrated score and box."""
    path = _resolve_image(ctx, image_path)
    active = float(ctx.threshold if threshold is None else threshold)
    if not 0.0 <= active <= 1.0:
        raise ToolError(
            f"threshold must be between 0 and 1, got {threshold}",
            "omit threshold to use the configured default",
        )
    result = run_localizer(path, threshold=active, detector=ctx.detector)
    return _as_payload(
        result,
        "confirm checkpoints/ contains the trained model, or inject a detector "
        "through ToolContext",
    )


def _validate_prediction(ctx: ToolContext, image_path: str) -> dict[str, Any]:
    """Check the localizer result already recorded in this investigation."""
    _resolve_image(ctx, image_path)
    localizer = ctx.recall("run_localizer")
    if not localizer or "error" in localizer:
        raise ToolError(
            "no localizer result to validate",
            "call run_localizer for this image first",
        )
    inspected = ctx.recall("inspect_image") or {}
    width = inspected.get("width")
    height = inspected.get("height")
    if width is None or height is None:
        size = localizer.get("image_size")
        if isinstance(size, list) and len(size) == 2:
            width, height = size[0], size[1]
    result = validate_prediction(
        run_result=localizer, image_width=width, image_height=height
    )
    return _as_payload(
        result,
        "run inspect_image first so the image dimensions are known",
    )


def _generate_ela(
    ctx: ToolContext, image_path: str, quality: int | None = None
) -> dict[str, Any]:
    """Write an ELA map to the artifact directory and return its path."""
    path = _resolve_image(ctx, image_path)
    if quality is not None and (isinstance(quality, bool) or not 1 <= int(quality) <= 100):
        raise ToolError(
            f"quality must be an integer between 1 and 100, got {quality}",
            "omit quality to use the configured default",
        )
    result = generate_ela(path, quality=None if quality is None else int(quality))
    return _as_payload(
        result,
        "the artifact directory must be writable; check the configured "
        "forensics.ela_temp_dir",
    )


def _inspect_metadata(ctx: ToolContext, image_path: str) -> dict[str, Any]:
    """Read metadata/EXIF fields and report which ones are present."""
    path = _resolve_image(ctx, image_path)
    result = inspect_metadata(str(path))
    return _as_payload(
        result,
        "the file must be a readable image; missing EXIF is normal and is not "
        "evidence of tampering",
    )


def _compare_evidence(ctx: ToolContext) -> dict[str, Any]:
    """Compare the evidence already recorded in this investigation."""
    if not ctx.results:
        raise ToolError(
            "no evidence recorded yet",
            "run run_localizer and validate_prediction first",
        )
    record = {
        "threshold": ctx.threshold,
        "observations": [
            {"tool": name, "result": payload} for name, payload in ctx.results.items()
        ],
    }
    result = compare_evidence(record)
    return _as_payload(
        result,
        "record at least the localizer and validation results before comparing",
    )


# ---------------------------------------------------------------------------
# Registry + schemas
# ---------------------------------------------------------------------------

# name -> implementation. The first positional argument is the injected
# ToolContext; the model only ever supplies the keyword arguments in the schema.
TOOLS: dict[str, Any] = {
    "inspect_image": _inspect_image,
    "run_localizer": _run_localizer,
    "validate_prediction": _validate_prediction,
    "generate_ela": _generate_ela,
    "inspect_metadata": _inspect_metadata,
    "compare_evidence": _compare_evidence,
}

_IMAGE_PATH_PROPERTY = {
    "type": "string",
    "description": "Absolute path of the image, inside the upload directory.",
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_image",
            "description": (
                "Validate an image and return its SHA-256 digest, dimensions and format. "
                "Use this FIRST, before any other tool: every later tool needs a readable "
                "image, and the digest identifies the file in the record. Returns "
                "{valid, issues, format, width, height, mode, byte_size, sha256}. It says "
                "nothing about tampering and does not modify the file. If 'valid' is "
                "false, stop rather than running detection."
            ),
            "parameters": {
                "type": "object",
                "properties": {"image_path": _IMAGE_PATH_PROPERTY},
                "required": ["image_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_localizer",
            "description": (
                "Run the project's ELA-based tampering localizer and return its score and "
                "region. Use it to find out whether the image looks spliced. Returns "
                "{tampered_prob, label, bbox, image_size, model_version, threshold}. "
                "tampered_prob is an UNCALIBRATED score in [0,1], not a probability; bbox "
                "is normalized [xmin, ymin, xmax, ymax] or null. Trained on CASIA v2 "
                "splices only: it cannot detect fully AI-generated images and is "
                "unreliable on heavily re-compressed screenshots. Always follow it with "
                "validate_prediction before trusting the region."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": _IMAGE_PATH_PROPERTY,
                    "threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": (
                            "Score at or above which the image is labelled tampered. "
                            "Omit to use the configured default."
                        ),
                    },
                },
                "required": ["image_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_prediction",
            "description": (
                "Check the localizer output already recorded for this image: is the score "
                "finite and in range, and is the bounding box ordered, in range and "
                "plausibly sized? Use it after run_localizer and before showing a region. "
                "Returns {valid, issues[]}. It does not re-read the image and cannot "
                "repair a missing or malformed box; 'valid' false means the region must "
                "not be presented as the tampered area."
            ),
            "parameters": {
                "type": "object",
                "properties": {"image_path": _IMAGE_PATH_PROPERTY},
                "required": ["image_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_ela",
            "description": (
                "Write an Error Level Analysis map next to the investigation and return "
                "its path. Use it when the score is borderline or the box looks wrong and "
                "you want supporting visual evidence. Returns {artifact, width, height, "
                "quality}. It only produces an image: it cannot decide tampering, because "
                "ELA also reacts to honest re-saving and re-compression."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": _IMAGE_PATH_PROPERTY,
                    "quality": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "JPEG quality used for the ELA round trip.",
                    },
                },
                "required": ["image_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_metadata",
            "description": (
                "Read EXIF/metadata and report which configured fields are present. Use it "
                "when the goal mentions origin, camera, editing history or source, since a "
                "stripped or contradictory EXIF block is weak corroboration. Returns "
                "{format, width, height, mode, exif, present_fields, absent_fields, "
                "parse_errors, exif_status}. Most images legitimately have no EXIF, so "
                "absence alone is NOT evidence, and deleted metadata cannot be recovered."
            ),
            "parameters": {
                "type": "object",
                "properties": {"image_path": _IMAGE_PATH_PROPERTY},
                "required": ["image_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_evidence",
            "description": (
                "Compare the score, validation, ELA and metadata already recorded in this "
                "investigation and list agreements and conflicts. Use it last, once at "
                "least two other tools have run, to judge whether the signals agree. "
                "Returns {agreements[], conflicts[], metadata_fields[]}. It reads only "
                "recorded evidence and cannot gather anything new; an empty conflict list "
                "means the recorded signals agree, not that the image is authentic."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


def run_tool(
    name: str, args: dict[str, Any] | None = None, ctx: ToolContext | None = None
) -> dict[str, Any]:
    """Run one approved tool by name and never raise.

    Args:
        name: a key of :data:`TOOLS`.
        args: only the simple arguments declared in the tool's schema.
        ctx: injected state; a fresh :class:`ToolContext` is used when omitted.

    Returns:
        The tool's JSON-safe payload, or ``{"error": ..., "hint": ...}``. An
        unknown tool returns an error that lists the available tool names.
    """
    if name not in TOOLS:
        return {
            "error": f"unknown tool: {name}",
            "available": sorted(TOOLS),
        }

    context = ctx if ctx is not None else ToolContext()
    try:
        payload = TOOLS[name](context, **(args or {}))
    except ToolError as exc:
        return exc.payload
    except TypeError as exc:
        return {
            "error": f"invalid arguments for {name}: {exc}",
            "hint": f"call {name} with only the arguments in its schema",
        }
    except Exception as exc:  # tool boundary must not leak exceptions
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "hint": f"{name} failed unexpectedly; the evidence is unusable",
        }

    if "error" not in payload:
        context.remember(name, payload)
    return payload
