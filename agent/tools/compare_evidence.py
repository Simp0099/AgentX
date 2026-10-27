"""Pure comparison of recorded score, validation, ELA and metadata evidence."""

from __future__ import annotations

from agent.schemas import InvestigationRecord, ToolResult
from agent.tools.common import run_bounded


def compare_evidence(
    record: InvestigationRecord | dict[str, object], timeout: float | None = None
) -> ToolResult:
    def work():
        observations = (
            record.observations
            if hasattr(record, "observations")
            else record.get("observations", [])
        )
        evidence = {
            (obs.tool if hasattr(obs, "tool") else obs["tool"]): (
                (obs.result or {}) if hasattr(obs, "result") else (obs.get("result") or {})
            )
            for obs in observations
        }
        localizer = evidence.get("run_localizer", {})
        validation = evidence.get("validate_prediction", {})
        ela = evidence.get("generate_ela", {})
        metadata = evidence.get("inspect_metadata", {})
        score = localizer.get("tampered_prob")
        issues = validation.get("issues", [])
        conflicts = []
        agreements = []
        threshold = (
            getattr(record, "threshold", 0.5)
            if hasattr(record, "threshold")
            else record.get("threshold", 0.5)
        )
        if score is not None and validation:
            if score >= threshold and validation.get("valid", False):
                agreements.append("high score and valid localization")
            if score >= threshold and not validation.get("valid", False):
                conflicts.append("high score with invalid localization")
            if score < threshold and issues:
                conflicts.append("low score with validation issues")
        if ela.get("status") == "error" or ela.get("ela_valid") is False:
            conflicts.append("ELA evidence unavailable or invalid")
        return {
            "agreements": agreements,
            "conflicts": conflicts,
            "metadata_fields": sorted(metadata.keys()),
        }

    return run_bounded("compare_evidence", work, timeout)
