"""Gradio-independent handler for agent chat and same-image follow-ups."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from agent.agent_tools import ToolContext, run_tool
from agent.loop import AgentResult, run_agent_full
from agent.memory_store import image_hash
from agent.orchestrator import run_investigation
from agent.prompts import CAVEAT_TEXT
from agent.rendering import escape_untrusted

UPLOAD_DIR = Path("artifacts/uploads")
ELA_TOOL_NAME = "generate_ela"
_MAX_PREVIEW_CHARS = 1200


def _upload_dir() -> Path:
    return Path(os.environ.get("AGENT_UPLOAD_ROOT", str(UPLOAD_DIR))).expanduser()


def _managed(path: str | Path) -> Path | None:
    """Return a path only when it is inside the chat upload directory."""
    root = _upload_dir().resolve()
    candidate = Path(path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _save_upload(image: Any) -> tuple[Path, str]:
    """Copy an uploaded image to a durable, conversation-owned PNG."""
    upload_dir = _upload_dir()
    upload_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        opened = Image.fromarray(image.astype(np.uint8))
        filename = "upload.png"
    elif isinstance(image, Image.Image):
        opened = image.copy()
        filename = Path(str(getattr(image, "filename", "upload.png"))).name
    else:
        source = Path(getattr(image, "name", image))
        filename = source.name
        with Image.open(source) as input_image:
            opened = input_image.copy()
    destination = upload_dir / f"chat-{uuid.uuid4().hex}.png"
    opened.convert("RGB").save(destination, format="PNG")
    return destination.resolve(), filename


def _temporary_copy(source: Path) -> Path:
    """Make a per-turn image copy that can be removed after tools finish."""
    with tempfile.NamedTemporaryFile(
        prefix="chat-turn-", suffix=".png", dir=_upload_dir(), delete=False
    ) as handle:
        destination = Path(handle.name)
    try:
        shutil.copyfile(source, destination)
    except OSError:
        destination.unlink(missing_ok=True)
        raise
    return destination.resolve()


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _agent_tool_log(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Pair model calls with tool messages and escape all displayed text."""
    entries: list[dict[str, str]] = []
    by_call_id: dict[str, dict[str, str]] = {}
    for message in messages:
        for call in message.get("tool_calls", []) or []:
            function = call.get("function", {})
            try:
                args = json.loads(function.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = function.get("arguments", "")
            entry = {
                "source": "agent",
                "tool": escape_untrusted(function.get("name", "unknown")),
                "arguments": escape_untrusted(_json_text(args)),
                "status": "pending",
                "result_preview": "No result was returned.",
            }
            entries.append(entry)
            by_call_id[str(call.get("id", ""))] = entry
        if message.get("role") != "tool":
            continue
        entry = by_call_id.get(str(message.get("tool_call_id", "")))
        if entry is None:
            continue
        raw = message.get("content", "")
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            payload = raw
        entry["status"] = "error" if isinstance(payload, dict) and "error" in payload else "ok"
        entry["result_preview"] = escape_untrusted(_json_text(payload)[:_MAX_PREVIEW_CHARS])
    return entries


def _record_tool_log(
    record: Any, *, source: str = "saved result"
) -> list[dict[str, str]]:
    """Build a safe evidence log for memory or deterministic fallback results."""
    entries = []
    for observation in record.observations:
        result = observation.result or observation.rationale or "No result returned."
        entries.append(
            {
                "source": escape_untrusted(source),
                "tool": escape_untrusted(observation.tool),
                "arguments": escape_untrusted(_json_text(observation.args)),
                "status": escape_untrusted(observation.status),
                "result_preview": escape_untrusted(_json_text(result)[:_MAX_PREVIEW_CHARS]),
            }
        )
    return entries


def _ela_from_record(record: Any) -> Image.Image | None:
    for observation in record.observations:
        artifact = (observation.result or {}).get("artifact")
        if observation.tool == ELA_TOOL_NAME and artifact:
            try:
                with Image.open(artifact) as generated:
                    return generated.copy()
            except (OSError, ValueError):
                return None
    return None


def _ensure_ela(image_path: Path) -> Image.Image | None:
    return _ensure_ela_with_path(image_path)[0]


def _ensure_ela_with_path(image_path: Path) -> tuple[Image.Image | None, str | None]:
    result = run_tool(
        ELA_TOOL_NAME,
        {"image_path": str(image_path)},
        ToolContext(upload_root=_upload_dir().resolve()),
    )
    artifact = result.get("artifact")
    if not artifact:
        return None, None
    try:
        with Image.open(artifact) as generated:
            return generated.copy(), str(artifact)
    except (OSError, ValueError):
        return None, None


def _clean_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(message) for message in messages if message.get("role") != "system"]


def _ok_result(record: Any, tool_name: str) -> dict[str, Any] | None:
    for observation in record.observations:
        if observation.tool == tool_name and observation.status == "ok":
            return observation.result or {}
    return None


def _fixed_answer(record: Any) -> str:
    """Summarize only the verdict and score actually present in the record."""
    verdict = record.verdict.value if record.verdict else "Inconclusive"
    localizer = _ok_result(record, "run_localizer")
    score = localizer.get("tampered_prob") if localizer else None
    score_text = f"{score}" if isinstance(score, (int, float)) else "unavailable"
    return f"Verdict: {verdict}. Tampered score (uncalibrated): {score_text}."


def _structured_reply(
    record: Any,
    answer_text: str,
    tool_log: list[dict[str, str]],
    mode: str,
    state: dict[str, Any],
    generated_ela_path: str | None = None,
) -> dict[str, Any]:
    """Build the chat contract from successful tool observations only."""
    inspected = _ok_result(record, "inspect_image")
    localized = _ok_result(record, "run_localizer")
    image_size = localized.get("image_size") if localized else None
    width = inspected.get("width") if inspected else None
    height = inspected.get("height") if inspected else None
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        width = width if width is not None else image_size[0]
        height = height if height is not None else image_size[1]
    width = int(width) if isinstance(width, (int, float)) and width > 0 else None
    height = int(height) if isinstance(height, (int, float)) and height > 0 else None

    score = localized.get("tampered_prob") if localized else None
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        score = None
    verdict = record.verdict.value if record.verdict else None

    region = None
    coverage_percent = None
    bbox = localized.get("bbox") if localized else None
    if (
        verdict == "Flagged for review"
        and isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and width is not None
        and height is not None
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox)
    ):
        xmin, ymin, xmax, ymax = (float(value) for value in bbox)
        if 0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1:
            horizontal = [round(xmin * width), round(xmax * width)]
            vertical = [round(ymin * height), round(ymax * height)]
            if horizontal[0] < horizontal[1] and vertical[0] < vertical[1]:
                region = {"horizontal": horizontal, "vertical": vertical}
                coverage_percent = round(
                    (horizontal[1] - horizontal[0])
                    * (vertical[1] - vertical[0])
                    / (width * height)
                    * 100,
                    1,
                )

    durations = [
        observation.duration_ms
        for observation in record.observations
        if observation.duration_ms is not None
        and isinstance(observation.duration_ms, (int, float))
    ]
    processing_seconds = round(sum(durations) / 1000, 3) if durations else None
    ela_path = next(
        (
            str((observation.result or {}).get("artifact"))
            for observation in record.observations
            if observation.tool == ELA_TOOL_NAME
            and (observation.result or {}).get("artifact")
        ),
        None,
    )
    if ela_path is None:
        ela_path = generated_ela_path

    return {
        "verdict": verdict,
        "tampered_score_uncalibrated": score,
        "image": {
            "filename": record.filename or None,
            "width": width,
            "height": height,
        },
        "processing_seconds": processing_seconds,
        "region": region,
        "coverage_percent": coverage_percent,
        "caveat": CAVEAT_TEXT,
        "answer_text": answer_text,
        "tool_log": tool_log,
        "ela_image_path": ela_path,
        "mode": mode,
        "_state": state,
    }


def _empty_reply(answer_text: str, history: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """Return the same structured shape when no image evidence is available."""
    return {
        "verdict": None,
        "tampered_score_uncalibrated": None,
        "image": {"filename": None, "width": None, "height": None},
        "processing_seconds": None,
        "region": None,
        "coverage_percent": None,
        "caveat": CAVEAT_TEXT,
        "answer_text": answer_text,
        "tool_log": [],
        "ela_image_path": None,
        "mode": mode,
        "_state": {"history": history, "image_path": None, "image_hash": None},
    }


def agent_chat(
    message: str,
    image: Any,
    state: dict[str, Any] | None,
    *,
    detector: Any | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Answer a chat turn with grounded fields and updated conversation state.

    The durable image remains under ``artifacts/uploads`` for same-image
    follow-ups. A separate per-turn copy is passed to the agent and removed in
    ``finally`` after its tool calls and any ELA fallback have completed.
    """
    current = dict(state or {})
    history = list(current.get("history", []))
    image_path = _managed(current.get("image_path", "")) if current.get("image_path") else None
    filename = str(current.get("filename", "image"))

    if image is not None:
        try:
            previous = image_path
            image_path, filename = _save_upload(image)
            if previous and previous != image_path:
                previous.unlink(missing_ok=True)
        except (OSError, ValueError) as exc:
            safe_error = escape_untrusted(f"{type(exc).__name__}: {exc}")
            return _empty_reply(
                f"Upload error: {safe_error}",
                history,
                mode="error",
            )
    if image_path is None or not image_path.is_file():
        return _empty_reply("Upload an image with your question to begin.", history, mode="waiting")

    stable_path = image_path
    digest = image_hash(stable_path)
    turn_path = _temporary_copy(stable_path)
    tool_log: list[dict[str, str]] = []
    ela: Image.Image | None = None
    generated_ela_path: str | None = None
    reused = False
    mode = "agent"
    try:
        try:
            result: AgentResult = run_agent_full(
                goal=message,
                image_path=str(turn_path),
                history=history,
                client=client,
            )
            if result.stopped_reason == "model_unavailable":
                mode = "fixed_pipeline"
                record = run_investigation(
                    str(turn_path),
                    goal="custom",
                    goal_text=message,
                    config={"max_tool_calls_per_image": 10},
                    planner=False,
                    detector=detector,
                )
                record.filename = filename
                tool_log = _record_tool_log(record, source="fixed pipeline")
                ela = _ela_from_record(record)
                answer_text = _fixed_answer(record)
                marker = "models tried:"
                if marker in result.answer.casefold():
                    marker_index = result.answer.casefold().rfind(marker)
                    tried_models = result.answer[marker_index + len(marker) :].strip().rstrip(".")
                    answer_text += f" Models tried: {tried_models}."
                saved_history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": answer_text},
                ]
                new_state = {
                    "history": saved_history,
                    "image_path": str(stable_path),
                    "image_hash": digest,
                    "filename": filename,
                    "reused_saved_result": False,
                }
                if ela is None:
                    ela, generated_ela_path = _ensure_ela_with_path(turn_path)
            else:
                reused = result.steps == 0 and not any(
                    item.get("tool_calls") for item in result.messages
                )
                tool_log = _agent_tool_log(result.messages)
                if reused:
                    tool_log = _record_tool_log(result.record)
                answer_text = result.answer
                ela = _ela_from_record(result.record)
                if ela is None and ELA_TOOL_NAME not in {
                    entry["tool"] for entry in tool_log
                }:
                    ela, generated_ela_path = _ensure_ela_with_path(turn_path)
                saved_history = _clean_history(result.messages)
                if reused:
                    saved_payload = [
                        {
                            "tool": item.tool,
                            "status": item.status,
                            "result": item.result,
                            "note": item.rationale,
                        }
                        for item in result.record.observations
                    ]
                    saved_history.append(
                        {
                            "role": "assistant",
                            "content": "Previously saved tool results (data): "
                            + _json_text(saved_payload),
                        }
                    )
                record = result.record
                record.filename = filename
                new_state = {
                    "history": saved_history,
                    "image_path": str(stable_path),
                    "image_hash": digest,
                    "filename": filename,
                    "reused_saved_result": reused,
                }
        except Exception:  # unavailable providers and unexpected client errors
            mode = "fixed_pipeline"
            record = run_investigation(
                str(turn_path),
                goal="custom",
                goal_text=message,
                config={"max_tool_calls_per_image": 10},
                planner=False,
                detector=detector,
            )
            record.filename = filename
            tool_log = _record_tool_log(record, source="fixed pipeline")
            ela = _ela_from_record(record)
            if ela is None:
                ela, generated_ela_path = _ensure_ela_with_path(turn_path)
            answer_text = _fixed_answer(record)
            updated_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer_text},
            ]
            new_state = {
                "history": updated_history,
                "image_path": str(stable_path),
                "image_hash": digest,
                "filename": filename,
                "reused_saved_result": False,
            }
    finally:
        turn_path.unlink(missing_ok=True)

    return _structured_reply(
        record, answer_text, tool_log, mode, new_state, generated_ela_path
    )
