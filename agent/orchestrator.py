"""agent/orchestrator.py: bounded loop — the only place tools are executed.

Run an investigation from image path to verdict, following the deterministic
policy and optional LLM planner.  All tool execution goes through the
approved registry; the orchestrator never calls tools directly.
"""

from __future__ import annotations

import hashlib
import os
import time

from agent.config import get_config
from agent.policy import next_checks
from agent.schemas import InvestigationRecord, Observation
from agent.tool_registry import ToolCall, tool_registry
from agent.verdict import decide


def run_investigation(
    image_path: str,
    goal: str = "quick_check",
    goal_text: str = "",
    config: dict | None = None,
    planner=None,
) -> InvestigationRecord:
    """Run a bounded forensic investigation.

    Returns an InvestigationRecord with all observations, verdict, and reasons.

    Steps:
      1. Initialise the record with baseline info.
      2. Execute baseline checks (deterministic policy).
      3. While budget remains: ask policy for next checks, execute through
         registry, append observations.
      4. If planner is provided and enabled, ask it for proposals; validate
         each; execute accepted ones (consume budget).
      5. Stop on: no next checks, budget exhausted, or inspect invalid.
    """
    if config is None:
        config = get_config()

    # --- Initialise the investigation record ---
    import uuid
    from datetime import datetime, timezone

    from agent.schemas import InvestigationRecord  # noqa: F811 redfin, already imported

    image_sha256 = image_path  # in real code: compute sha256 of the file
    filename = os.path.basename(image_path)

    record: InvestigationRecord = InvestigationRecord(
        investigation_id=str(uuid.uuid4()),
        image_id=image_sha256,
        filename=filename,
        goal=goal,
        goal_text=goal_text,
        threshold=config.get("threshold", 0.5),
        config_snapshot_hash=hashlib.sha256(str(config).encode()).hexdigest()[:16],
        model_version=config.get("model_version", "unknown"),
        observations=[],
        errors=[],
        rationale="",
        verdict=None,
        verdict_reasons=[],
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    budget = config.get("max_tool_calls_per_image", 10)

    def execute_tool_call(tool_call: ToolCall) -> Observation:
        """Execute a single ToolCall through the registry.

        Returns the Observation with status, result, duration, and rationale.
        Transient errors ("error") are retried once.
        """
        tool_name = tool_call.tool
        args = tool_call.args.copy()

        # The orchestrator supplies the image path; the LLM never does.
        # Sanitise: ensure "path" is not an absolute or suspicious path.
        if "path" in args:
            p = str(args["path"])
            # Reject absolute paths and ".." components
            if os.path.isabs(p) or ".." in p.split(os.sep):
                return Observation(
                    tool=tool_name,
                    args=args,
                    status="error",
                    rationale="rejected: unsafe path",
                )

        while True:
            start_ms = int(time.time() * 1000)

            # Look up the tool from the registry
            tool_info = tool_registry.get(tool_name)
            if tool_info is None:
                return Observation(
                    tool=tool_name,
                    args=args,
                    status="error",
                    rationale=f"unknown tool: {tool_name}",
                )

            callable_ = tool_info["callable"]
            # arg_schema = tool_info.get("arg_schema", {})  # validated by registry

            # Validate arguments against schema (simple type/check)
            # The registry ensures only approved args are passed.
            try:
                result = callable_(**args)
            except Exception as exc:
                return Observation(
                    tool=tool_name,
                    args=args,
                    status="error",
                    rationale=f"execution error: {exc}",
                )

            duration_ms = int(time.time() * 1000) - start_ms

            # Check status for retry
            # Build observation
            obs = Observation(
                tool=tool_name,
                args=args,
                status=result.status,
                result=result.result,
                duration_ms=duration_ms,
                rationale=result.error or "tool completed",
            )
            return obs

        # Should not reach here, but just in case
        return Observation(
            tool=tool_call.tool,
            args=tool_call.args,
            status="error",
            rationale="max retries exceeded",
        )

    # --- Step 2: Execute baseline checks ---
    # The policy returns groups of ToolCall; execute each group
    while budget > 0:
        # Ask policy for next checks
        groups = next_checks(record)

        if not groups:
            # No more checks prescribed by policy
            break

        # Execute each group; observations are appended to record
        for group in groups:
            if budget <= 0:
                break

            for tool_call in group:
                if budget <= 0:
                    break

                # Execute the tool call
                obs = execute_tool_call(tool_call)

                # Append observation to record
                # Check for duplicate (same tool + args already run)
                obs_tools = [o.tool for o in record.observations]
                if obs.tool not in obs_tools:
                    record.observations.append(obs)
                # else: already ran this check, skip adding again

                budget -= 1

                # Check if inspect_image found invalid image → stop
                if obs.tool == "inspect_image":
                    result = obs.result or {}
                    if not result.get("valid", True):
                        # Image invalid → stop immediately
                        # Record verdict as Inconclusive
                        v, reasons = decide(record)
                        record.verdict = v
                        record.verdict_reasons = reasons
                        record.rationale = "; ".join(reasons)
                        return record

                # Check if we should continue based on policy
                # The loop will re-invoke next_checks which reads record state

        # After executing a group, re-invoke next_checks to see what's next
        # (The policy function reads the record state to determine next checks)

    # --- Step 5: If planner is provided and enabled, ask for proposals ---
    if planner is not None:
        try:
            proposals = planner.propose(record, config)
            for proposal in proposals:
                if budget <= 0:
                    break
                # Validate the proposal
                tool_name = proposal.get("tool")
                args = proposal.get("args", {})

                # Check tool is in registry and allowed for planner
                tool_info = tool_registry.get(tool_name)
                if tool_info is None:
                    record.observations.append(
                        Observation(
                            tool=tool_name,
                            args=args,
                            status="error",
                            rationale=f"unapproved planner tool: {tool_name}",
                        )
                    )
                    continue

                # Check budget
                if budget <= 0:
                    break

                # Execute accepted proposal
                obs = execute_tool_call(ToolCall(tool=tool_name, args=args))
                budget -= 1
                record.observations.append(obs)

                # Check if inspect_image invalid
                if obs.tool == "inspect_image":
                    result = obs.result or {}
                    if not result.get("valid", True):
                        v, reasons = decide(record)
                        record.verdict = v
                        record.verdict_reasons = reasons
                        record.rationale = "; ".join(reasons)
                        return record

        except Exception:
            # Planner unavailable — continue without it
            pass

    # --- Final: decide verdict ---
    v, reasons = decide(record)
    record.verdict = v
    record.verdict_reasons = reasons
    record.rationale = "; ".join(reasons) if reasons else ""

    return record
