"""Bounded deterministic policy loop with an optional constrained planner."""

from __future__ import annotations

import hashlib
import os
import uuid

from agent.config import get_config
from agent.policy import next_checks
from agent.schemas import InvestigationRecord, Observation, ToolResult
from agent.summary import SummaryBuilder
from agent.tool_registry import PLANNER_TOOLS, ToolCall, tool_registry, validate_args
from agent.verdict import decide


def _planner_instance(planner):
    if planner is False:
        return None
    if planner is not None:
        return planner
    if not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("ANTHROPIC_MODEL"):
        return None
    from agent.planner import AnthropicPlanner

    return AnthropicPlanner()


def run_investigation(
    image_path: str,
    goal: str = "quick_check",
    goal_text: str = "",
    config: dict | None = None,
    planner=None,
    detector=None,
) -> InvestigationRecord:
    """Run baseline and conditional checks through the approved registry."""
    full_config = get_config()
    supplied_config = config or {}
    forensics = full_config.get("forensics", {})
    threshold = float(
        supplied_config.get("threshold", full_config.get("output", {}).get("threshold", 0.5))
    )
    snapshot = dict(full_config)
    snapshot.update(supplied_config)
    active_planner = _planner_instance(planner)
    record = InvestigationRecord(
        investigation_id=str(uuid.uuid4()),
        image_id=str(uuid.uuid4()),
        filename=os.path.basename(image_path),
        goal=goal,
        goal_text=goal_text[:2000],
        threshold=threshold,
        config_snapshot_hash=hashlib.sha256(str(snapshot).encode()).hexdigest()[:16],
        model_version=str(supplied_config.get("model_version", "unknown")),
        observations=[],
        errors=[],
        rationale="",
        image_path=image_path,
        planner_status="ready" if active_planner else "unavailable",
        config_snapshot=snapshot,
    )
    limit = int(
        supplied_config.get(
            "max_tool_calls_per_image", forensics.get("max_tool_calls_per_image", 10)
        )
    )
    calls_used = 0

    def run_one(call: ToolCall, *, retry: bool = True) -> Observation:
        nonlocal calls_used
        args = dict(call.args)
        if call.tool == "run_localizer" and detector is not None:
            args["detector"] = detector
        # Detector is injected by trusted application code, not user/planner input.
        schema_args = {key: value for key, value in args.items() if key != "detector"}
        valid, reason = validate_args(call.tool, schema_args)
        if not valid:
            return Observation(
                call.tool, schema_args, "error", rationale=f"rejected arguments: {reason}"
            )
        callable_ = tool_registry[call.tool]["callable"]
        attempts = 2 if retry else 1
        last: Observation | None = None
        for attempt in range(attempts):
            if calls_used >= limit:
                break
            calls_used += 1
            try:
                result = callable_(**args)
                if not isinstance(result, ToolResult):
                    result = ToolResult(call.tool, "error", error="tool returned an invalid result")
            except Exception as exc:
                result = ToolResult(call.tool, "error", error=str(exc))
            last = Observation(
                tool=call.tool,
                args=schema_args,
                status=result.status,
                result=result.result,
                duration_ms=result.duration_ms,
                rationale=result.error or "tool completed",
            )
            if result.status == "ok":
                return last
            record.errors.append(
                f"{call.tool} attempt {attempt + 1}: {result.error or result.status}"
            )
        if last is None:
            return Observation(
                call.tool, schema_args, "error", rationale="tool-call budget exhausted"
            )
        return last

    def append_observation(observation: Observation) -> None:
        record.observations.append(observation)
        if observation.tool == "inspect_image" and observation.result:
            digest = observation.result.get("sha256")
            if isinstance(digest, str):
                record.sha256 = digest
                record.image_id = digest
        if observation.tool == "run_localizer" and observation.result:
            record.model_version = str(
                observation.result.get("model_version") or record.model_version
            )

    # Deterministic checks always run before conditional checks or planner proposals.
    while calls_used < limit:
        groups = next_checks(record)
        if not groups:
            break
        for group in groups:
            for call in group:
                if calls_used >= limit:
                    break
                observation = run_one(call)
                append_observation(observation)
                if call.tool == "inspect_image" and not (observation.result or {}).get(
                    "valid", True
                ):
                    groups = []
                    break
            if not groups or calls_used >= limit:
                break
        if record.observations and record.observations[-1].tool == "inspect_image":
            if not (record.observations[-1].result or {}).get("valid", True):
                break

    # The planner may request only its additional-check registry entries.
    if active_planner is not None and calls_used < limit:
        try:
            proposals = active_planner.propose(record, supplied_config)
            record.planner_status = "available"
        except Exception as exc:
            proposals = []
            record.planner_status = "unavailable"
            record.errors.append(f"planner unavailable: {type(exc).__name__}")
        proposal_budget = int(forensics.get("planner_budget", 4))
        for proposal in proposals:
            if calls_used >= limit or proposal_budget <= 0:
                break
            proposal_budget -= 1
            name = proposal.get("tool") if isinstance(proposal, dict) else None
            proposed_args = proposal.get("args", {}) if isinstance(proposal, dict) else None
            if name not in PLANNER_TOOLS:
                calls_used += 1
                append_observation(
                    Observation(str(name), {}, "rejected", rationale="unapproved planner tool")
                )
                continue
            if not isinstance(proposed_args, dict):
                calls_used += 1
                append_observation(
                    Observation(name, {}, "rejected", rationale="malformed planner arguments")
                )
                continue
            spec = PLANNER_TOOLS[name]["input_schema"]
            properties = spec.get("properties", {})
            if set(proposed_args) - set(properties):
                calls_used += 1
                append_observation(
                    Observation(name, {}, "rejected", rationale="malformed planner arguments")
                )
                continue
            if name == "generate_ela":
                quality = proposed_args.get("quality")
                if quality is not None and (
                    not isinstance(quality, int)
                    or isinstance(quality, bool)
                    or not 1 <= quality <= 100
                ):
                    calls_used += 1
                    append_observation(
                        Observation(name, {}, "rejected", rationale="malformed planner arguments")
                    )
                    continue
                trusted_args = {"path": image_path}
                if quality is not None:
                    trusted_args["quality"] = quality
            elif name == "inspect_metadata":
                trusted_args = {"path": image_path}
            else:
                trusted_args = {"record": record}
            valid, reason = validate_args(name, trusted_args)
            if not valid:
                calls_used += 1
                append_observation(Observation(name, {}, "rejected", rationale=reason))
                continue
            # This is an additional proposed call; it is separately counted.
            if calls_used >= limit:
                break
            observation = run_one(ToolCall(name, trusted_args))
            append_observation(observation)

    verdict, reasons = decide(record)
    record.verdict = verdict
    record.verdict_reasons = reasons
    record.rationale = "; ".join(reasons)
    template = SummaryBuilder(record)
    record.summary = template.build()
    if active_planner is not None and record.planner_status == "available":
        try:
            candidate = active_planner.summarize(record)
            if isinstance(candidate, str) and template.is_grounded(candidate):
                record.summary = candidate
            else:
                record.errors.append("planner summary discarded: ungrounded numeric claim")
        except Exception as exc:
            record.errors.append(f"planner summary unavailable: {type(exc).__name__}")
    return record
