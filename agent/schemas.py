"""agent/schemas.py: Pydantic-style data models for the investigation record.

These are plain Python dataclasses (no pydantic dependency) so the agent
remains lightweight and works without network access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal


class Verdict(Enum):
    """Verdict taxonomy — exactly these four; "authentic" is banned."""

    FLAGGED_FOR_REVIEW = "Flagged for review"
    FLAGGED_REGION_UNAVAILABLE = "Flagged, region unavailable"
    NOT_FLAGGED_BY_THIS_DETECTOR = "Not flagged by this detector"
    INCONCLUSIVE = "Inconclusive"


class Outcome(Enum):
    """High-level outcome of an investigation."""

    COMPLETED = "completed"
    INCONCLUSIVE = "inconclusive"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Observation:
    """A single tool observation in the investigation record."""

    tool: str
    args: dict[str, object]
    status: str  # "ok", "error", "timeout"
    result: dict[str, object] | None = None
    duration_ms: float | None = None
    rationale: str = ""  # fixed string from policy, never LLM text
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ToolResult:
    """Typed outcome shared by all forensic tools."""

    tool: str
    status: Literal["ok", "error", "timeout"]
    result: dict[str, object] | None = None
    error: str = ""
    duration_ms: float = 0.0

    @property
    def valid(self) -> bool:
        return self.status == "ok" and bool((self.result or {}).get("valid", True))

    @property
    def issues(self):
        from types import SimpleNamespace

        if self.status != "ok":
            return [SimpleNamespace(category="tool", message=self.error)]
        data = self.result or {}
        return [
            SimpleNamespace(category="score" if "score" in str(issue) else "bbox", message=issue)
            for issue in data.get("issues", [])
        ]

    def __getattr__(self, name):
        if name in (self.result or {}):
            return self.result[name]
        raise AttributeError(name)


@dataclass
class InvestigationRecord:
    """Core investigation record — the single source of truth for the PDF
    and any summary."""

    investigation_id: str  # uuid4
    image_id: str  # sha256 of the image file (not untrusted)
    filename: str  # untrusted, escaped in output
    goal: str  # enum: quick_check | detailed | custom
    goal_text: str  # untrusted, length-capped
    threshold: float
    config_snapshot_hash: str
    model_version: str
    observations: list[Observation]
    errors: list[str]
    rationale: str  # free-form but must only reference numbers in the record
    sha256: str = ""
    verdict: Verdict | None = None
    verdict_reasons: list[str] = field(default_factory=list)
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    planner_status: str = "not_configured"
    summary: str = ""
    image_path: str = ""
    config_snapshot: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialise the record to a plain dict (JSON-safe)."""
        return {
            "investigation_id": self.investigation_id,
            "image_id": self.image_id,
            "filename": self.filename,
            "goal": self.goal,
            "goal_text": self.goal_text,
            "threshold": self.threshold,
            "config_snapshot_hash": self.config_snapshot_hash,
            "model_version": self.model_version,
            "observations": [
                {
                    "tool": o.tool,
                    "args": o.args,
                    "status": o.status,
                    "result": o.result,
                    "duration_ms": o.duration_ms,
                    "rationale": o.rationale,
                    "timestamp_utc": o.timestamp_utc,
                }
                for o in self.observations
            ],
            "errors": self.errors,
            "rationale": self.rationale,
            "verdict": self.verdict.value if self.verdict else None,
            "verdict_reasons": self.verdict_reasons or [],
            "created_at_utc": self.created_at_utc,
            "sha256": self.sha256,
            "planner_status": self.planner_status,
            "summary": self.summary,
            "image_path": self.image_path,
            "config_snapshot": self.config_snapshot,
        }
