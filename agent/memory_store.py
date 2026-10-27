"""SQLite-backed investigation memory keyed by image content hash."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.schemas import InvestigationRecord, Observation, Verdict


def _default_path() -> str:
    """Read the configured SQLite path without exposing credentials."""
    from agent.agent_config import get_agent_config

    return str(get_agent_config()["memory"]["path"])


@dataclass
class SavedInvestigation:
    """Saved answer and record returned from long-term memory."""

    answer: str
    record: InvestigationRecord
    model: str
    threshold: float
    config_hash: str
    timestamp: str


def image_hash(image_path: str | Path) -> str:
    """Return SHA-256 for the image file bytes, without saving those bytes."""
    digest = hashlib.sha256()
    with open(image_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _connect(path: str | Path) -> sqlite3.Connection:
    database = Path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS investigations (
            image_hash TEXT PRIMARY KEY, verdict TEXT, tool_results TEXT NOT NULL,
            model TEXT NOT NULL, threshold REAL NOT NULL, config_hash TEXT NOT NULL,
            timestamp TEXT NOT NULL, record_json TEXT NOT NULL, answer TEXT NOT NULL
        )"""
    )
    return connection


def save_investigation(
    path: str | Path,
    key: str,
    record: InvestigationRecord,
    answer: str,
    model: str,
    threshold: float,
    config_hash: str,
) -> None:
    """Persist a record, its observations and cache validation metadata."""
    timestamp = record.created_at_utc
    results = [
        {
            "tool": item.tool,
            "args": item.args,
            "status": item.status,
            "result": item.result,
            "rationale": item.rationale,
        }
        for item in record.observations
    ]
    with _connect(path) as connection:
        connection.execute(
            """INSERT OR REPLACE INTO investigations
            (image_hash, verdict, tool_results, model, threshold, config_hash,
             timestamp, record_json, answer) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                key,
                record.verdict.value if record.verdict else None,
                json.dumps(results),
                model,
                threshold,
                config_hash,
                timestamp,
                json.dumps(record.to_dict()),
                answer,
            ),
        )


def _record_from_dict(data: dict[str, Any]) -> InvestigationRecord:
    observations = [Observation(**item) for item in data.pop("observations", [])]
    verdict_text = data.pop("verdict", None)
    if verdict_text:
        data["verdict"] = Verdict(verdict_text)
    return InvestigationRecord(observations=observations, **data)


def load_investigation(
    path: str | Path,
    key: str,
    model: str,
    threshold: float,
    config_hash: str,
) -> SavedInvestigation | None:
    """Return a matching cache row, ignoring stale model/config results."""
    if not Path(path).exists():
        return None
    with _connect(path) as connection:
        row = connection.execute(
            """SELECT verdict, model, threshold, config_hash, timestamp,
                      record_json, answer FROM investigations WHERE image_hash=?""",
            (key,),
        ).fetchone()
    if row is None or row[1] != model or row[2] != threshold or row[3] != config_hash:
        return None
    record = _record_from_dict(json.loads(row[5]))
    return SavedInvestigation(row[6], record, row[1], row[2], row[3], row[4])


def clear_memory(path: str | Path | None = None) -> None:
    """Delete all saved investigation rows while preserving the database file."""
    path = path or _default_path()
    if not Path(path).exists():
        return
    with _connect(path) as connection:
        connection.execute("DELETE FROM investigations")
