"""Gradio-independent handler for agent chat and same-image follow-ups."""

from __future__ import annotations

import json
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
from agent.rendering import escape_untrusted, render_record

UPLOAD_DIR = Path("artifacts/uploads")
ELA_TOOL_NAME = "generate_ela"
_MAX_PREVIEW_CHARS = 1200


def _managed(path: str | Path) -> Path | None:
    """Return a path only when it is inside the chat upload directory."""
    root = UPLOAD_DIR.resolve()
    candidate = Path(path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _save_upload(image: Any) -> tuple[Path, str]:
    """Copy an uploaded image to a durable, conversation-owned PNG."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
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
    destination = UPLOAD_DIR / f"chat-{uuid.uuid4().hex}.png"
    opened.convert("RGB").save(destination, format="PNG")
    return destination.resolve(), filename


def _temporary_copy(source: Path) -> Path:
    """Make a per-turn image copy that can be removed after tools finish."""
    with tempfile.NamedTemporaryFile(
        prefix="chat-turn-", suffix=".png", dir=UPLOAD_DIR, delete=False
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
    result = run_tool(
        ELA_TOOL_NAME,
        {"image_path": str(image_path)},
        ToolContext(upload_root=UPLOAD_DIR.resolve()),
    )
    artifact = result.get("artifact")
    if not artifact:
        return None
    try:
        with Image.open(artifact) as generated:
            return generated.copy()
    except (OSError, ValueError):
        return None


def _clean_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(message) for message in messages if message.get("role") != "system"]


def agent_chat(
    message: str,
    image: Any,
    state: dict[str, Any] | None,
    *,
    detector: Any | None = None,
    client: Any | None = None,
) -> tuple[str, list[dict[str, str]], Image.Image | None, dict[str, Any]]:
    """Answer a chat turn about an uploaded image and return updated state.

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
            return (
                f"### Upload error\n\n{safe_error}",
                [],
                None,
                {"history": history, "image_path": None, "image_hash": None},
            )
    if image_path is None or not image_path.is_file():
        return (
            "Upload an image with your question to begin.",
            [],
            None,
            {"history": history, "image_path": None, "image_hash": None},
        )

    stable_path = image_path
    digest = image_hash(stable_path)
    turn_path = _temporary_copy(stable_path)
    tool_log: list[dict[str, str]] = []
    ela: Image.Image | None = None
    reused = False
    fallback_reason = ""
    try:
        try:
            result: AgentResult = run_agent_full(
                goal=message,
                image_path=str(turn_path),
                history=history,
                client=client,
            )
            if result.stopped_reason == "model_unavailable":
                fallback_reason = "No configured model provider was available."
                raise RuntimeError("model unavailable")
            reused = result.steps == 0 and not any(
                item.get("tool_calls") for item in result.messages
            )
            tool_log = _agent_tool_log(result.messages)
            if reused:
                tool_log = _record_tool_log(result.record)
            answer_text = escape_untrusted(result.answer)
            if reused:
                answer_text = "**Reused saved result.**\n\n" + answer_text
            ela = _ela_from_record(result.record)
            if ela is None and ELA_TOOL_NAME not in {
                entry["tool"] for entry in tool_log
            }:
                ela = _ensure_ela(turn_path)
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
            new_state = {
                "history": saved_history,
                "image_path": str(stable_path),
                "image_hash": digest,
                "filename": filename,
                "reused_saved_result": reused,
            }
        except Exception as exc:  # unavailable providers and unexpected client errors
            record = run_investigation(
                str(turn_path),
                goal="custom",
                goal_text=message,
                config={"max_tool_calls_per_image": 10},
                planner=False,
                detector=detector,
            )
            record.filename = filename
            tool_log = _record_tool_log(record, source="fixed pipeline (no LLM)")
            ela = _ela_from_record(record)
            if ela is None:
                ela = _ensure_ela(turn_path)
            answer_text = (
                "**fixed pipeline (no LLM)**\n\n"
                + render_record(record)
                + "\n\n"
                + escape_untrusted(
                    fallback_reason or f"Agent request failed ({type(exc).__name__})."
                )
            )
            updated_history = history + [
                {"role": "user", "content": escape_untrusted(message)},
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

    safe_filename = escape_untrusted(filename)
    rendered_answer = f"### {safe_filename}\n\n{answer_text}"
    return rendered_answer, tool_log, ela, new_state
