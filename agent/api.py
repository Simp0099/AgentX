"""agent/api.py: HTTP surface over the same tools the agent uses.

The API is a thin wrapper around :func:`agent.agent_tools.run_tool`. It contains
no inference code and never touches the model directly: a detector is built once
(lazily) and injected through ``ToolContext``, so the tools stay the only place
evidence is produced.

Endpoints
    GET  /health          liveness probe
    GET  /tools           the OpenAI-format tool schemas
    POST /tools/{name}    run one tool

Status conventions
    * A malformed body (missing ``image_path``, wrong type, unknown field) is a
      client error and FastAPI answers ``422``.
    * An unknown tool name answers ``404`` and lists the available tools.
    * A tool-level failure (bad path, unreadable image, missing checkpoint) is a
      normal result, so it answers ``200`` with ``{"error": ..., "hint": ...}``.

Running it (loopback only by default):

    uvicorn agent.api:app --host 127.0.0.1 --port 8000

Safety
    * The documented binding is ``127.0.0.1``. Do not expose it on a wildcard
      address without a token.
    * No CORS middleware is installed, so browsers on other origins get nothing.
    * Image paths must resolve inside the upload directory (``AGENT_UPLOAD_ROOT``,
      default ``artifacts/uploads``), exactly as in :mod:`agent.agent_tools`.
    * Setting ``AGENT_API_TOKEN`` in the environment turns on a shared-secret
      check: every ``POST /tools/{name}`` must then send a matching
      ``X-Agent-Token`` header. It is off by default, and the token value is
      never logged.

Logging
    Exactly one line per tool request, carrying the tool name, the outcome and
    the duration. Argument values are never logged, because they can be file
    contents or paths a user considers private.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from agent.agent_tools import TOOL_SCHEMAS, TOOLS, ToolContext, run_tool

logger = logging.getLogger("agent.api")


def _configure_logging() -> None:
    """Make the one-line-per-request log actually reach the console.

    Uvicorn configures only its own loggers, so an ``agent.api`` record with no
    handler of its own is silently dropped. A dedicated handler keeps the line
    independent of how the app is served, and ``propagate = False`` guarantees
    exactly one line even when the root logger is also configured.
    """
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


_configure_logging()

app = FastAPI(
    title="AI Deepfake & Tampering Localizer - agent tools",
    description=(
        "Local forensic tools for the tampering agent. Bind to 127.0.0.1; the "
        "detector score is an uncalibrated review signal, not proof."
    ),
    version="0.1.0",
)

ENV_UPLOAD_ROOT = "AGENT_UPLOAD_ROOT"
ENV_API_TOKEN = "AGENT_API_TOKEN"
DEFAULT_UPLOAD_ROOT = "artifacts/uploads"
TOKEN_HEADER = "X-Agent-Token"


# ---------------------------------------------------------------------------
# Request models: one explicit model per tool, mirroring each schema
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    """Reject unknown fields so a typo is a 422, not a silently ignored key."""

    model_config = ConfigDict(extra="forbid")


class InspectImageRequest(_StrictModel):
    image_path: str = Field(min_length=1, description="Absolute path inside the upload directory.")


class RunLocalizerRequest(_StrictModel):
    image_path: str = Field(min_length=1, description="Absolute path inside the upload directory.")
    threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Score at or above which the label is tampered."
    )


class GenerateElaRequest(_StrictModel):
    image_path: str = Field(min_length=1, description="Absolute path inside the upload directory.")
    quality: int | None = Field(default=None, ge=1, le=100, description="JPEG quality 1-100.")


class NoArgumentsRequest(_StrictModel):
    """``compare_evidence`` takes no model-supplied arguments."""


# tool name -> request model. Keeping this explicit means a missing field is a
# schema error at import time, not a surprise 500 at request time.
REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "inspect_image": InspectImageRequest,
    "inspect_metadata": InspectImageRequest,
    "validate_prediction": InspectImageRequest,
    "run_localizer": RunLocalizerRequest,
    "generate_ela": GenerateElaRequest,
    "compare_evidence": NoArgumentsRequest,
}


def _assert_models_cover_the_tools() -> None:
    missing = sorted(set(TOOLS) - set(REQUEST_MODELS))
    extra = sorted(set(REQUEST_MODELS) - set(TOOLS))
    if missing or extra:
        raise RuntimeError(
            f"request models out of sync with TOOLS: missing={missing} extra={extra}"
        )


_assert_models_cover_the_tools()


# ---------------------------------------------------------------------------
# Detector: built once, lazily, and never required for the app to start
# ---------------------------------------------------------------------------

# Only these tools touch the model; every other tool must stay fast and usable
# while the checkpoint is missing.
DETECTOR_TOOLS = frozenset({"run_localizer"})

_detector: Any | None = None
_detector_loaded = False
_detector_lock = threading.Lock()


def _checkpoint_path() -> str:
    """Return the configured checkpoint path (never validates existence)."""
    from agent.config import get_config

    config = get_config()
    directory = str(config.get("output", {}).get("checkpoint_dir", "checkpoints/")).rstrip("/")
    name = str(config.get("output", {}).get("checkpoint_name", "best_localizer.keras"))
    return f"{directory}/{name}"


def get_detector() -> Any | None:
    """Load the detector once, or return None when it is unavailable.

    A missing checkpoint must not stop the app: ``/health`` and ``/tools`` work
    regardless, and tool calls that need the model come back as an error dict.
    """
    global _detector, _detector_loaded
    if _detector_loaded:
        return _detector
    with _detector_lock:
        if _detector_loaded:
            return _detector
        try:
            # Imported here, not at module level: importing agent.api must not
            # pull in TensorFlow or require a checkpoint.
            from src.inference import TamperingDetector

            _detector = TamperingDetector(_checkpoint_path())
        except Exception as exc:  # missing checkpoint, unreadable file, bad env
            logger.warning(
                "detector unavailable (%s); tool calls will report an error",
                type(exc).__name__,
            )
            _detector = None
        _detector_loaded = True
    return _detector


def set_detector(detector: Any | None) -> None:
    """Inject a detector and mark loading as done (used by tests and callers)."""
    global _detector, _detector_loaded
    _detector = detector
    _detector_loaded = True


def reset_detector() -> None:
    """Forget the cached detector so the next call attempts a fresh load."""
    global _detector, _detector_loaded
    _detector = None
    _detector_loaded = False


def upload_root() -> Path:
    """Return the directory image paths are confined to."""
    return Path(os.environ.get(ENV_UPLOAD_ROOT, DEFAULT_UPLOAD_ROOT)).expanduser()


# ---------------------------------------------------------------------------
# Optional shared-secret check (off unless AGENT_API_TOKEN is set)
# ---------------------------------------------------------------------------


def require_token(x_agent_token: str | None = Header(default=None)) -> None:
    """Enforce ``X-Agent-Token`` when ``AGENT_API_TOKEN`` is configured."""
    expected = os.environ.get(ENV_API_TOKEN)
    if not expected:
        return
    if not x_agent_token or not secrets.compare_digest(x_agent_token, expected):
        raise HTTPException(status_code=401, detail=f"missing or invalid {TOKEN_HEADER} header")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe; does not touch the model."""
    return {"status": "ok"}


@app.get("/tools")
def list_tools() -> list[dict[str, Any]]:
    """Return the approved tool schemas in OpenAI ``tools`` format."""
    return TOOL_SCHEMAS


def _run_and_log(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run one tool, log exactly one line, and return the payload."""
    started = time.perf_counter()
    # Only the localizer needs the model. Loading TensorFlow for a metadata read
    # would cost seconds per cold start and would make every tool depend on a
    # checkpoint that only one of them uses.
    detector = get_detector() if tool_name in DETECTOR_TOOLS else None
    context = ToolContext(detector=detector, upload_root=upload_root())
    try:
        result = run_tool(tool_name, args, context)
    finally:
        duration_ms = (time.perf_counter() - started) * 1000.0
    status = "error" if "error" in result else "ok"
    # Tool name, outcome and duration only: never the arguments.
    logger.info("tool=%s status=%s duration_ms=%.1f", tool_name, status, duration_ms)
    return result


def _make_endpoint(tool_name: str, model: type[BaseModel]):
    """Build the POST handler for one tool with its own request model.

    The parameter annotation is set to the concrete model object rather than
    left as ``BaseModel``: FastAPI reads ``__annotations__`` to decide which
    model validates the body, and only then does a missing field become a 422.
    """

    def endpoint(payload: Any) -> dict[str, Any]:
        # exclude_none keeps optional fields out so each tool applies its own
        # configured default instead of receiving an explicit null.
        args = payload.model_dump(exclude_none=True)
        return _run_and_log(tool_name, args)

    endpoint.__name__ = f"post_tools_{tool_name}"
    endpoint.__doc__ = f"Run the {tool_name} tool."
    endpoint.__annotations__ = {"payload": model, "return": dict[str, Any]}
    return endpoint


# Registered before the catch-all so the real paths win.
for _name, _model in REQUEST_MODELS.items():
    app.post(
        f"/tools/{_name}",
        response_model=None,
        dependencies=[Depends(require_token)],
        summary=f"Run {_name}",
    )(_make_endpoint(_name, _model))


@app.post("/tools/{name}", response_model=None, dependencies=[Depends(require_token)])
def unknown_tool(name: str) -> dict[str, Any]:
    """Answer 404 for any tool name that is not registered."""
    raise HTTPException(
        status_code=404,
        detail={"error": f"unknown tool: {name}", "available": sorted(TOOLS)},
    )
