"""Secret-free settings for the Gradio interface."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path("artifacts/settings.json")
GENERAL_DEFAULTS = {"history_keep": 10, "appearance": "Light"}


def _defaults() -> dict[str, Any]:
    from agent.config import get_config

    config = get_config()
    forensics = config.get("forensics", {})
    return {
        "threshold": float(config.get("output", {}).get("threshold", 0.5)),
        "max_file_size_mb": max(
            1, round(int(forensics.get("max_file_bytes", 50_000_000)) / 1_000_000)
        ),
        "max_zip_members": int(forensics.get("max_zip_members", 100)),
        "batch_time_limit_seconds": int(forensics.get("max_batch_processing_seconds", 120)),
        "max_workers": int(forensics.get("batch_workers", 2)),
        "run_ela": True,
        "run_metadata_inspection": True,
        **GENERAL_DEFAULTS,
    }


def _validated(values: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    result = dict(defaults)
    try:
        result["threshold"] = max(
            0.0, min(1.0, float(values.get("threshold", result["threshold"])))
        )
    except (TypeError, ValueError):
        pass
    bounds = {
        "max_file_size_mb": (1, 500),
        "max_zip_members": (1, 5000),
        "batch_time_limit_seconds": (1, 3600),
        "max_workers": (1, 32),
        "history_keep": (1, 100),
    }
    for key, (minimum, maximum) in bounds.items():
        try:
            result[key] = max(minimum, min(maximum, int(values.get(key, result[key]))))
        except (TypeError, ValueError):
            pass
    for key in ("run_ela", "run_metadata_inspection"):
        result[key] = bool(values.get(key, result[key]))
    appearance = values.get("appearance", result["appearance"])
    if appearance in {"Light", "Dark"}:
        result["appearance"] = appearance
    return result


def load_ui_settings(path: str | Path = SETTINGS_PATH) -> dict[str, Any]:
    """Read the secret-free settings JSON, falling back to configs/default.yaml."""
    defaults = _defaults()
    try:
        stored = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        stored = {}
    if not isinstance(stored, dict):
        stored = {}
    return _validated(stored, defaults)


def _write_settings(settings: dict[str, Any], path: str | Path = SETTINGS_PATH) -> None:
    """Atomically save only known non-secret settings."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe = _validated(settings, _defaults())
    fd, temporary = tempfile.mkstemp(prefix="settings-", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(safe, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def apply_analysis_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Apply upload/batch bounds to the existing runtime config cache."""
    from agent import config as config_module

    safe = _validated(settings, _defaults())
    config = config_module.get_config()
    config.setdefault("output", {})["threshold"] = safe["threshold"]
    for key, value in {
        "max_file_bytes": safe["max_file_size_mb"] * 1_000_000,
        "max_zip_members": safe["max_zip_members"],
        "max_batch_processing_seconds": safe["batch_time_limit_seconds"],
        "batch_workers": safe["max_workers"],
        "run_ela": safe["run_ela"],
        "run_metadata_inspection": safe["run_metadata_inspection"],
    }.items():
        config.setdefault("forensics", {})[key] = value
    return safe


def save_analysis_settings(
    threshold: float,
    max_file_size_mb: int,
    max_zip_members: int,
    batch_time_limit_seconds: int,
    max_workers: int,
    run_ela: bool,
    run_metadata_inspection: bool,
    *,
    path: str | Path = SETTINGS_PATH,
    apply: bool = True,
) -> tuple[dict[str, Any], str]:
    """Merge analysis options into settings JSON and apply them immediately."""
    all_settings = load_ui_settings(path)
    all_settings.update(
        threshold=threshold,
        max_file_size_mb=max_file_size_mb,
        max_zip_members=max_zip_members,
        batch_time_limit_seconds=batch_time_limit_seconds,
        max_workers=max_workers,
        run_ela=run_ela,
        run_metadata_inspection=run_metadata_inspection,
    )
    safe = _validated(all_settings, _defaults())
    _write_settings(safe, path)
    if apply:
        apply_analysis_settings(safe)
    return safe, "Saved to artifacts/settings.json. Changes apply to the next run."


def save_general_settings(
    history_keep: int,
    appearance: str,
    *,
    path: str | Path = SETTINGS_PATH,
) -> tuple[dict[str, Any], str]:
    all_settings = load_ui_settings(path)
    all_settings.update(history_keep=history_keep, appearance=appearance)
    safe = _validated(all_settings, _defaults())
    _write_settings(safe, path)
    return safe, "Saved to artifacts/settings.json."


def record_ui_run(
    files: list[dict[str, Any]], path: str | Path = Path("artifacts/history")
) -> None:
    """Store a compact Gradio run summary and enforce JSON history retention."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    run = {"id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "files": files}
    (root / f"{run_id}.json").write_text(json.dumps(run), encoding="utf-8")
    keep = int(load_ui_settings()["history_keep"])
    runs = sorted(
        (item for item in root.glob("*.json") if item.is_file()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for old in runs[keep:]:
        old.unlink(missing_ok=True)


def recent_runs_html(path: str | Path = Path("artifacts/history"), limit: int | None = None) -> str:
    """Render recent run items as stacked, escaped rows with truncated names."""
    root = Path(path)
    if not root.exists():
        return '<p class="recent-runs-empty">No recent runs yet.</p>'
    keep = int(limit if limit is not None else load_ui_settings()["history_keep"])
    runs = []
    for item in root.glob("*.json"):
        try:
            runs.append(json.loads(item.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    runs.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    cards = []
    for run in runs[:keep]:
        first = (run.get("files") or [{}])[0]
        data = first.get("data") or first
        batch_first = (run.get("batch") or {}).get("results", [{}])[0] if run.get("batch") else {}
        filename = str(first.get("filename") or batch_first.get("filename") or "Analysis run")
        verdict = str(
            data.get("verdict") or batch_first.get("verdict") or run.get("verdict") or "Mixed"
        )
        score = data.get(
            "tampered_score_uncalibrated",
            first.get("score", batch_first.get("tampered_prob", run.get("score"))),
        )
        score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "Unavailable"
        timestamp = str(run.get("created_at", "Time unavailable"))
        cards.append(
            '<article class="recent-run-item">'
            f'<div class="recent-run-name" title="{escape(filename, quote=True)}">'
            f"{escape(filename)}</div>"
            f'<div class="recent-run-meta"><span>{escape(verdict)}</span>'
            f"<span>Score {escape(score_text)}</span></div>"
            f'<div class="recent-run-time">{escape(timestamp)}</div>'
            "</article>"
        )
    return "".join(cards) if cards else '<p class="recent-runs-empty">No recent runs yet.</p>'
