"""Deterministic conditional investigation policy."""

from __future__ import annotations

from agent.config import get_config
from agent.schemas import InvestigationRecord, Verdict
from agent.tool_registry import ToolCall
from agent.verdict import decide  # noqa: F821


def next_checks(record: InvestigationRecord) -> list[list[ToolCall]]:
    """Return the next check group based only on recorded evidence."""
    observations = [obs for obs in record.observations if not isinstance(obs, list)]
    by_tool = {obs.tool: obs for obs in observations}
    done = set(by_tool)
    path = record.image_path or record.image_id
    if "inspect_image" not in done:
        return [[ToolCall("inspect_image", {"path": path})]]
    inspected = by_tool["inspect_image"].result or {}
    if not inspected.get("valid", True):
        return []
    if "run_localizer" not in done:
        return [[ToolCall("run_localizer", {"path": path, "threshold": record.threshold})]]
    localizer = by_tool["run_localizer"].result or {}
    if "validate_prediction" not in done:
        return [
            [
                ToolCall(
                    "validate_prediction",
                    {
                        "run_result": localizer,
                        "image_width": inspected.get("width"),
                        "image_height": inspected.get("height"),
                    },
                )
            ]
        ]

    validation = by_tool["validate_prediction"].result or {}
    score = localizer.get("tampered_prob")
    config = get_config().get("forensics", {})
    if not isinstance(score, (int, float)):
        return []
    uncertain = (
        float(config.get("uncertain_low", 0.4)) <= score <= float(config.get("uncertain_high", 0.6))
    )
    flagged = score >= record.threshold
    goal = f"{record.goal} {record.goal_text}".lower()
    metadata_goal = any(
        word in goal for word in ("origin", "source", "camera", "editing", "history")
    )
    issues = validation.get("issues", [])
    box_problem = any("box" in str(issue).lower() for issue in issues)
    coverage_problem = any("coverage" in str(issue).lower() for issue in issues)
    conflict = (flagged and box_problem) or coverage_problem

    need_ela = uncertain or flagged or conflict
    need_metadata = uncertain or metadata_goal or conflict
    if need_ela and "generate_ela" not in done:
        return [[ToolCall("generate_ela", {"path": path})]]
    if need_metadata and "inspect_metadata" not in done:
        return [[ToolCall("inspect_metadata", {"path": path})]]
    if conflict and "compare_evidence" not in done:
        return [[ToolCall("compare_evidence", {"record": record})]]
    return []


class StopPolicy:
    """Bounded investigation stop rules. Applies to both the LLM and the fixed
    pipeline, so both agree on when the loop may end.

    The loop stops when any of these hold:
      1. Budget exhausted -- the configured step cap has been reached.
      2. Evidence consistent -- a confident verdict (not Inconclusive) is
         already available.
      3. Validation failed -- a required check ended in error/timeout; the
         ``decide`` function reports ``Inconclusive`` (the loop keeps running
         on inconclusive evidence until the budget is exhausted, so the LLM
         never presents an invented verdict on top of a failed check).
    Conflicting signals do NOT stop the loop; the loop keeps gathering until
    the budget is exhausted, at which point ``decide`` reports
    ``Inconclusive`` instead of a confident verdict.
    """

    def should_stop(self, record: InvestigationRecord, max_steps: int | None = None) -> tuple[bool, str]:
        """Return ``(should_stop, reason)`` for the current investigation state."""
        # 1. Budget exhausted.
        if max_steps is not None and len(record.observations) >= max_steps:
            return True, "budget exhausted"
        # 2. Evidence consistent -> a confident verdict already exists.
        verdict, _reasons = decide(record)
        if verdict is not None and verdict != Verdict.INCONCLUSIVE.value:
            return True, "evidence_consistent; confident verdict available"
        # 3. Validation failed / inconclusive signals -> keep gathering up to
        #    the budget rather than inventing a verdict.
        return False, "signals_conflict_or_insufficient; continue"

    def remaining_budget(self, record: InvestigationRecord) -> int:
        config = get_config()
        budget = int(config.get("forensics", {}).get("max_tool_calls_per_image", 10))
        return max(0, budget - len(record.observations))


def should_stop(record: InvestigationRecord, max_steps: int | None = None) -> tuple[bool, str]:
    """Module-level convenience wrapper used by the orchestrator."""
    return StopPolicy().should_stop(record, max_steps=max_steps)

