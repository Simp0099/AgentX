"""agent/state.py: the investigation record and its renderable views.

This module is the single source of truth for what was actually executed and
observed during an investigation. ``InvestigationRecord``/``Observation`` and
``ToolResult`` are defined in ``agent.schemas`` (the existing dataclasses) and
are re-exported here so that every component -- the orchestrator, the Gradio
handler, and the report builders -- reads from one place.

The record carries, per tool call:
    tool        : str          name of the approved tool
    args        : dict         the JSON-safe arguments the LLM/planner supplied
    status      : str          "ok", "error", "timeout" or "rejected"
    result      : dict | None  the tool's typed output (facts only)
    duration_ms : float | None wall-clock time the tool took
    rationale   : str          the short reason the LLM/planner gave for the call
    timestamp_utc: str        ISO-8601 UTC of the call

The final report is ALWAYS built from recorded observations only. Nothing is
ever synthesized: if a fact is not in the record, the report says "unavailable".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The concrete record classes live in agent.schemas (single source of truth).
from agent.schemas import (
    InvestigationRecord,
    Observation,
    ToolResult,
    Verdict,
)

__all__ = [
    "InvestigationRecord",
    "Observation",
    "ToolResult",
    "Verdict",
    "ToolCallLog",
    "to_gradio_log",
    "render_tool_call_log",
    "render_report_from_observations",
]


@dataclass
class ToolCallLog:
    """One rendered row in the Gradio tool-call log."""

    tool: str
    args: dict[str, Any]
    status: str
    rationale: str
    timestamp_utc: str
    result_preview: str = ""  # short, JSON-safe summary of the tool result

    @property
    def is_error(self) -> bool:
        return self.status in ("error", "timeout", "rejected")


@dataclass
class GradioLog:
    """The complete, ordered tool-call log produced from a record."""

    record: InvestigationRecord
    rows: list[ToolCallLog] = field(default_factory=list)

    def build(self) -> list[ToolCallLog]:
        """Build the log from the record's ordered observations."""
        self.rows = []
        for obs in self.record.observations:
            preview = ""
            if obs.status == "ok" and obs.result:
                # Only surface facts that are already in the record.
                safe = {}
                for key, value in (obs.result or {}).items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        safe[key] = value
                    else:
                        safe[key] = f"<{type(value).__name__}>"
                if safe:
                    preview = " " + str(safe)[:220]
            self.rows.append(
                ToolCallLog(
                    tool=obs.tool,
                    args=obs.args,
                    status=obs.status,
                    rationale=obs.rationale or "(no rationale)",
                    timestamp_utc=obs.timestamp_utc,
                    result_preview=preview,
                )
            )
        return self.rows

    def render(self) -> str:
        """Render the log as Markdown, exactly mirroring the recorded state."""
        lines = ["**Tool-call log**", ""]  # header
        if not self.rows:
            lines.append("- (no tool calls recorded)")
        for row in self.rows:
            lines.append(f"- `{row.tool}`")
            if row.args:
                lines.append(f"    args: `{self._compact(row.args)}`")
            lines.append(f"    status: `{row.status}`")
            if row.rationale:
                lines.append(f"    rationale: {row.rationale}")
            if row.result_preview:
                lines.append(f"    result: {row.result_preview}")
            if row.is_error and not row.result_preview:
                # Errors carry the message in the tool's `error` field; the
                # orchestrator stores those on the observation's rationale.
                if row.rationale:
                    lines.append(f"    error: {row.rationale}")
        return "\n".join(lines)

    @staticmethod
    def _compact(value: dict[str, Any]) -> str:
        import json

        return json.dumps(value, sort_keys=True, default=str)


def to_gradio_log(record: InvestigationRecord) -> list[ToolCallLog]:
    """Return the Gradio tool-call log for ``record``."""
    return GradioLog(record).build()


def render_tool_call_log(record: InvestigationRecord) -> str:
    """Render the tool-call log as a Markdown string."""
    return GradioLog(record).render()


def render_report_from_observations(
    observations: list[Observation],
    verdict: str | None,
    reasons: list[str],
    goal_name: str,
    filename: str,
) -> str:
    """Build a text report that contains ONLY facts present in observations.

    The report never invents numbers. Any quantity that is absent from the
    recorded observations is reported as "unavailable".
    """

    def fact(tool: str, key: str) -> str:
        for obs in observations:
            if obs.tool == tool and obs.status == "ok" and obs.result:
                if key in obs.result:
                    value = obs.result[key]
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        return str(value)
                    return f"<{type(value).__name__}>"
        return "unavailable"

    score = fact("run_localizer", "tampered_prob")
    bbox = fact("run_localizer", "bbox")
    label = fact("run_localizer", "label")
    validation = fact("validate_prediction", "valid")
    issues = fact("validate_prediction", "issues")

    lines = [
        "Investigation report",
        f"Goal: {goal_name}",
        f"Filename: {filename}",
        f"Verdict: {verdict if verdict else 'unavailable'}",
        f"Tool calls executed: {len(observations)}",
        f"Tampered score (uncalibrated): {score}",
        f"Localizer label: {label}",
        f"Bounding box: {bbox}",
        f"Prediction validation: {validation}",
        f"Validation issues: {issues}",
    ]
    if reasons:
        lines.append("Reasons:")
        lines.extend(f"  - {r}" for r in reasons)
    return "\n".join(lines)
