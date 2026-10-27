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
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from agent.agent_tools import TOOL_SCHEMAS, TOOLS, ToolContext, run_tool
from agent.env import load_env

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
load_env()


def _llm_availability() -> dict[str, Any]:
    """Describe LLM configuration without returning credential values."""
    from agent.agent_config import get_agent_config

    try:
        config = get_agent_config()
        provider = str(config["provider"])
        key_name = str(config["api_key_env"])
        model = str(config["model"])
        key_value = os.environ.get(key_name, "").strip()
        placeholder = key_value.casefold() in {
            "your_key_here",
            "replace_me",
            "changeme",
            "placeholder",
        }
        available = bool(key_value) and not placeholder and bool(model)
        reason = None if available else f"missing or placeholder {key_name}, or model configuration"
        return {"available": available, "provider": provider, "model": model, "reason": reason}
    except Exception as exc:  # configuration details should remain concise
        return {
            "available": False,
            "provider": os.environ.get("LLM_PROVIDER", "configured provider"),
            "model": None,
            "reason": f"configuration error: {type(exc).__name__}",
        }


_startup_llm = _llm_availability()
if not _startup_llm["available"]:
    logger.warning("language model unavailable at startup: %s", _startup_llm["reason"])

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


class ChatRequest(_StrictModel):
    """A chat turn over an image already present in the allowed upload root."""

    message: str = Field(min_length=1, max_length=4000)
    image_path: str = Field(min_length=1)
    history: list[dict[str, Any]] = Field(default_factory=list)
    threshold: float | None = Field(default=None, ge=0.1, le=0.9)


class BatchRequest(_StrictModel):
    zip_path: str = Field(min_length=1)


class ReportRequest(_StrictModel):
    image_paths: list[str] = Field(min_length=1, max_length=50)
    threshold: float = Field(default=0.5, ge=0.1, le=0.9)


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


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    """Report app liveness and whether the configured LLM has a key."""
    llm = _llm_availability()
    return {"status": "ok", "llm": llm}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the standalone local dashboard."""
    return FileResponse(Path(__file__).resolve().parent.parent / "app" / "dashboard.html")


@app.get("/tools")
def list_tools() -> list[dict[str, Any]]:
    """Return the approved tool schemas in OpenAI ``tools`` format."""
    return TOOL_SCHEMAS


@app.post("/api/chat", dependencies=[Depends(require_token)])
def chat(payload: ChatRequest) -> dict[str, Any]:
    """Run one chat turn and return the evidence card fields as JSON."""
    image_path = Path(payload.image_path).expanduser().resolve()
    try:
        image_path.relative_to(upload_root().resolve())
    except ValueError:
        raise HTTPException(
            status_code=400, detail="image_path is outside the allowed upload directory"
        ) from None
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="image_path does not exist")

    from app.chat_handler import agent_chat

    result = agent_chat(
        payload.message,
        None,
        {
            "image_path": str(image_path),
            "filename": image_path.name,
            "history": payload.history,
        },
        detector=get_detector(),
        threshold=payload.threshold,
    )
    reply = {key: value for key, value in result.items() if not key.startswith("_")}
    internal_state = result.get("_state")
    if isinstance(internal_state, dict):
        reply["conversation_state"] = {
            "history": internal_state.get("history", payload.history),
            "image_path": str(image_path),
            "filename": image_path.name,
        }
    return reply


@app.post("/api/upload", dependencies=[Depends(require_token)])
async def upload_file(
    request: Request,
    filename: str = Query(min_length=1, max_length=255),
) -> dict[str, Any]:
    """Store one browser upload under the guarded upload root."""
    suffix = Path(filename).suffix.lower()
    supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".zip"}
    if suffix not in supported:
        raise HTTPException(status_code=415, detail="unsupported file type")
    root = upload_root().resolve()
    root.mkdir(parents=True, exist_ok=True)
    max_bytes = 50_000_000
    try:
        from agent.config import get_config

        max_bytes = int(get_config().get("forensics", {}).get("max_file_bytes", max_bytes))
    except Exception:
        pass
    target = root / f"upload-{uuid.uuid4().hex}{suffix}"
    total = 0
    try:
        with target.open("wb") as handle:
            async for chunk in request.stream():
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413, detail="upload exceeds configured size limit"
                    )
                handle.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="upload is empty")
        if suffix != ".zip":
            from PIL import Image

            with Image.open(target) as image:
                image.verify()
            return {
                "image_path": str(target.resolve()),
                "filename": Path(filename).name,
                "bytes": total,
            }

        extracted, warnings = _extract_valid_zip_images(target, root)
        if not extracted:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "archive contains no valid supported images",
                    "warnings": warnings,
                },
            )
        return {
            "image_path": str(target.resolve()),
            "filename": Path(filename).name,
            "bytes": total,
            "images": extracted,
            "warnings": warnings,
        }
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422, detail=f"invalid upload: {type(exc).__name__}"
        ) from None


_ZIP_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
_ZIP_MAX_ENTRIES = 2000
_ZIP_MAX_IMAGES = 100
_ZIP_MAX_UNCOMPRESSED_BYTES = 250_000_000


def _extract_valid_zip_images(
    archive_path: Path, upload_directory: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Safely extract and validate supported images from an uploaded ZIP."""
    from PIL import Image

    image_limit = 50_000_000
    try:
        from agent.config import get_config

        image_limit = int(get_config().get("forensics", {}).get("max_file_bytes", image_limit))
    except Exception:
        pass
    warnings: list[dict[str, str]] = []
    extracted: list[dict[str, str]] = []
    destination_root = Path(tempfile.mkdtemp(prefix="zip-images-", dir=upload_directory)).resolve()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if len(infos) > _ZIP_MAX_ENTRIES:
                raise HTTPException(status_code=413, detail="archive contains too many entries")
            declared_candidates = []
            for info in infos:
                normalized_name = info.filename.replace("\\", "/")
                parts = PurePosixPath(normalized_name).parts
                if (
                    not info.is_dir()
                    and Path(normalized_name).suffix.lower() in _ZIP_IMAGE_EXTENSIONS
                    and not any(part.startswith(".") or part == "__MACOSX" for part in parts)
                ):
                    declared_candidates.append(info)
            if len(declared_candidates) > _ZIP_MAX_IMAGES:
                raise HTTPException(status_code=413, detail="archive contains too many images")
            declared_total = sum(info.file_size for info in declared_candidates)
            if declared_total > _ZIP_MAX_UNCOMPRESSED_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="archive uncompressed image data is too large",
                )
            total_written = 0
            for info in infos:
                filename = info.filename
                normalized = filename.replace("\\", "/")
                parts = PurePosixPath(normalized).parts
                if info.is_dir() or not parts:
                    continue
                if any(part == "__MACOSX" or part.startswith(".") for part in parts):
                    continue
                extension = Path(normalized).suffix.lower()
                if extension not in _ZIP_IMAGE_EXTENSIONS:
                    warnings.append(
                        {"filename": filename, "warning": "unsupported or non-image file skipped"}
                    )
                    continue
                if ".." in parts:
                    warnings.append(
                        {"filename": filename, "warning": "unsafe archive path skipped"}
                    )
                    continue
                if info.file_size > image_limit:
                    warnings.append(
                        {
                            "filename": filename,
                            "warning": "file exceeds configured image size limit",
                        }
                    )
                    continue
                target = (destination_root.joinpath(*parts)).resolve()
                try:
                    target.relative_to(destination_root)
                except ValueError:
                    warnings.append(
                        {"filename": filename, "warning": "unsafe archive path skipped"}
                    )
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                try:
                    with archive.open(info) as source, target.open("wb") as output:
                        while chunk := source.read(1024 * 1024):
                            written += len(chunk)
                            total_written += len(chunk)
                            if written > image_limit:
                                raise ValueError("file exceeds configured image size limit")
                            if total_written > _ZIP_MAX_UNCOMPRESSED_BYTES:
                                raise HTTPException(
                                    status_code=413,
                                    detail="archive uncompressed image data is too large",
                                )
                            output.write(chunk)
                    with Image.open(target) as image:
                        image.verify()
                except HTTPException:
                    raise
                except Exception as exc:
                    target.unlink(missing_ok=True)
                    warnings.append(
                        {
                            "filename": filename,
                            "warning": f"invalid image skipped ({type(exc).__name__})",
                        }
                    )
                    continue
                extracted.append({"filename": filename, "image_path": str(target)})
        if not extracted:
            shutil.rmtree(destination_root, ignore_errors=True)
        return extracted, warnings
    except HTTPException:
        shutil.rmtree(destination_root, ignore_errors=True)
        raise
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        shutil.rmtree(destination_root, ignore_errors=True)
        raise HTTPException(
            status_code=422,
            detail=f"invalid ZIP archive ({type(exc).__name__})",
        ) from None


@app.get("/api/assets", dependencies=[Depends(require_token)])
def asset(path: str = Query(min_length=1)) -> FileResponse:
    """Serve only uploaded images, ELA maps, batch exports and reports."""
    candidate = Path(path).expanduser().resolve()
    roots = [upload_root().resolve(), (Path.cwd() / "artifacts" / "ela").resolve()]
    try:
        from agent.config import get_config

        config = get_config().get("forensics", {})
        for key in ("ela_temp_dir", "report_asset_roots"):
            configured = config.get(key, [])
            if isinstance(configured, str):
                configured = [configured]
            roots.extend(Path(item).expanduser().resolve() for item in configured)
        roots.extend(
            (Path.cwd() / "artifacts" / folder).resolve()
            for folder in ("batch", "reports")
        )
    except Exception:
        pass
    if not candidate.is_file() or not any(candidate.is_relative_to(root) for root in roots):
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(candidate)


@app.post("/api/batch", dependencies=[Depends(require_token)])
def analyze_batch(payload: BatchRequest) -> dict[str, Any]:
    """Run the existing bounded ZIP batch pipeline."""
    archive = Path(payload.zip_path).expanduser().resolve()
    try:
        archive.relative_to(upload_root().resolve())
    except ValueError:
        raise HTTPException(
            status_code=400, detail="zip_path is outside the upload directory"
        ) from None
    if archive.suffix.lower() != ".zip" or not archive.is_file():
        raise HTTPException(status_code=404, detail="ZIP upload not found")
    from agent.batch import process_batch

    result = process_batch(str(archive), "artifacts/batch", detector=get_detector())
    safe_results = [
        {key: value for key, value in item.items() if key != "record"}
        for item in result["results"]
    ]
    return {**result, "results": safe_results}


@app.post("/api/report", dependencies=[Depends(require_token)])
def export_report(payload: ReportRequest) -> dict[str, str]:
    """Generate a PDF report from fresh recorded tool results."""
    upload_root_path = upload_root().resolve()
    paths = []
    for item in payload.image_paths:
        candidate = Path(item).expanduser().resolve()
        try:
            candidate.relative_to(upload_root_path)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="image_path is outside the upload directory"
            ) from None
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="image_path does not exist")
        paths.append(candidate)

    from agent.orchestrator import run_investigation
    from agent.reporting import generate_batch_pdf, generate_pdf

    records = [
        run_investigation(
            str(path),
            goal="quick_check",
            config={"threshold": payload.threshold},
            planner=False,
            detector=get_detector(),
        )
        for path in paths
    ]
    report_dir = Path.cwd() / "artifacts" / "reports"
    report_path = report_dir / f"deepfake-localizer-{uuid.uuid4().hex}.pdf"
    if len(records) == 1:
        generate_pdf(records[0], report_path)
    else:
        generate_batch_pdf(records, report_path)
    return {"report_path": str(report_path.resolve())}


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
