"""Bounded deterministic policy loop with an optional constrained planner."""

from __future__ import annotations

import hashlib
import os
import uuid
from typing import Any

from agent.config import get_config
from agent.llm_client import LLMClient, LLMNotConfigured, get_llm_client
from agent.policy import next_checks, should_stop
from agent.schemas import InvestigationRecord, Observation, ToolResult
from agent.summary import SummaryBuilder
from agent.tool_registry import PLANNER_TOOLS, ToolCall, tool_registry, validate_args
from agent.verdict import decide


def _planner_instance(planner):
    if planner is False:
        return None
    if planner is not None:
        return planner
    if (
        not os.environ.get("ANTHROPIC_API_KEY")
        or not os.environ.get("ANTHROPIC_MODEL")
        or not os.environ.get("ANTHROPIC_BASE_URL")
    ):
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


# ---------------------------------------------------------------------------
# LLM-driven investigation loop (bounded, per-tool timeout, deterministic
# fallback labelled 'fixed pipeline (no LLM)').
# ---------------------------------------------------------------------------

MAX_STEPS = 5  # default number of LLM turn-steps per investigation


def build_tool_schemas() -> list[dict[str, Any]]:
    """Return the approved tools as OpenAI ``tools`` entries, cached.

    The model-facing shape lives in ``agent.agent_tools.TOOL_SCHEMAS``: each
    entry is ``{"type": "function", "function": {name, description,
    parameters}}``, which is what an OpenAI-compatible endpoint expects. The
    previous version returned bare JSON Schema objects, which the API rejects.
    """
    cache_key = "_tool_schemas"
    if not getattr(build_tool_schemas, cache_key, None):
        from agent.agent_tools import TOOL_SCHEMAS

        setattr(build_tool_schemas, cache_key, list(TOOL_SCHEMAS))
    return getattr(build_tool_schemas, cache_key)  # type: ignore[return-value]


def _run_one_llm_tool(
    record: InvestigationRecord,
    tool_name: str,
    args: dict[str, Any],
    detector: Any | None,
    config: dict[str, Any],
    timeout: float | None,
) -> Observation:
    """Execute one approved tool through the typed registry runner and record it."""
    result = tool_registry[tool_name]["run"](
        config=config, detector=detector, timeout=timeout, **args
    )
    status = result.status
    rationale = result.error or ("tool completed" if status == "ok" else "")
    if status == "ok" and tool_name == "inspect_image" and result.result:
        digest = result.result.get("sha256")
        if isinstance(digest, str):
            record.sha256 = digest
            record.image_id = digest
    if status == "ok" and tool_name == "run_localizer" and result.result:
        record.model_version = str(result.result.get("model_version") or record.model_version)
    observation = Observation(
        tool=tool_name,
        args=dict(args),
        status=status,
        result=result.result,
        duration_ms=result.duration_ms,
        rationale=rationale,
    )
    record.observations.append(observation)
    if status in ("error", "timeout"):
        record.errors.append(f"{tool_name}: {result.error or status}")
    return observation


def _run_llm_loop(
    image_path: str,
    goal: str,
    goal_text: str,
    config: dict[str, Any],
    detector: Any | None,
    llm_client: LLMClient,
    max_steps: int,
    tool_timeout: float | None,
) -> InvestigationRecord:
    """The bounded LLM-driven loop. The report is built only from recorded
    observations -- the LLM proposes tool calls but never results."""
    supplied_config = config or {}
    full_config = get_config()
    threshold = float(
        supplied_config.get(
            "threshold", full_config.get("output", {}).get("threshold", 0.5)
        )
    )

    record = InvestigationRecord(
        investigation_id=str(uuid.uuid4()),
        image_id=str(uuid.uuid4()),
        filename=os.path.basename(image_path),
        goal=goal,
        goal_text=goal_text[:2000],
        threshold=threshold,
        config_snapshot_hash=hashlib.sha256(str(supplied_config).encode()).hexdigest()[:16],
        model_version=str(supplied_config.get("model_version", "unknown")),
        observations=[],
        errors=[],
        rationale="",
        image_path=image_path,
        planner_status="available",
        config_snapshot=dict(full_config, **supplied_config),
    )

    tool_schemas = build_tool_schemas()
    llm = llm_client if llm_client is not None else get_llm_client()

    for _step in range(max_steps):
        # (a) Ask the LLM for the next action: goal + recorded state + schemas.
        try:
            decision = llm.choose_action(
                goal,
                [
                    {
                        "tool": o.tool,
                        "args": o.args,
                        "status": o.status,
                        "result": o.result,
                        "rationale": o.rationale,
                        "timestamp_utc": o.timestamp_utc,
                    }
                    for o in record.observations
                ],
                tool_schemas,
                timeout_seconds=tool_timeout,
            )
        except LLMNotConfigured:
            # No key reachable: fall back to the deterministic fixed pipeline.
            fallback = run_investigation(
                image_path,
                goal=goal,
                goal_text=goal_text,
                config=supplied_config,
                detector=detector,
            )
            fallback.rationale = (
                f"fixed pipeline (no LLM); {fallback.rationale or '(no rationale)'}"
            )
            return fallback

        if decision.finished:
            # (b) LLM says finish -> stop.
            break

        # (c) Validate the tool name against the registry.
        if decision.tool not in tool_registry:
            record.observations.append(
                Observation(
                    decision.tool,
                    decision.arguments,
                    "rejected",
                    rationale=f"unapproved tool name: {decision.tool}",
                )
            )
            record.errors.append(f"rejected {decision.tool}: unapproved tool name")
            continue

        # (c) Validate the arguments against the tool's JSON schema.
        valid, reason = validate_args(decision.tool, decision.arguments)
        if not valid:
            record.observations.append(
                Observation(
                    decision.tool,
                    decision.arguments,
                    "rejected",
                    rationale=f"invalid arguments rejected: {reason}",
                )
            )
            record.errors.append(f"rejected {decision.tool}: {reason}")
            continue

        # (d) Execute the tool with a per-tool timeout and store the observation.
        _run_one_llm_tool(
            record, decision.tool, decision.arguments, detector, config, tool_timeout
        )

        # Stop rules: budget exhausted, evidence consistent.
        stop, _reason = should_stop(record, max_steps)
        if stop:
            break

    # The final report is built ONLY from the recorded observations.
    verdict, reasons = decide(record)
    record.verdict = verdict
    record.verdict_reasons = reasons
    record.rationale = "; ".join(reasons)
    template = SummaryBuilder(record)
    record.summary = template.build()
    return record


def run_llm_investigation(
    image_path: str,
    goal: str = "quick_check",
    goal_text: str = "",
    config: dict[str, Any] | None = None,
    detector=None,
    llm_client=None,
    max_steps: int = MAX_STEPS,
    tool_timeout: float | None = None,
) -> InvestigationRecord:
    """Entry point for the bounded LLM-driven agent.

    When an LLM client is supplied (or can be built from the environment) and
    is configured, the investigation runs via the LLM-driven loop. When no
    LLM key is configured, the deterministic fixed pipeline is run and the
    record is labelled 'fixed pipeline (no LLM)'.
    """
    supplied_config = config or {}
    if llm_client is None and get_llm_client() is None:
        record = run_investigation(
            image_path,
            goal=goal,
            goal_text=goal_text,
            config=supplied_config,
            detector=detector,
        )
        record.rationale = (
            f"fixed pipeline (no LLM); {record.rationale or '(no rationale)'}"
        )
        return record
    return _run_llm_loop(
        image_path,
        goal,
        goal_text,
        supplied_config,
        detector,
        llm_client,
        max_steps,
        tool_timeout,
    )
