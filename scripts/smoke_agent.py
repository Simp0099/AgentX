"""Run three real-model smoke cases, or one candidate at a time.

The script creates authentic and spliced images with the repository's synthetic
generator plus a corrupt file. It prints plan text and tool call/result sizes.
Use ``--candidate NAME`` to limit quota use to one configured candidate.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import loop  # noqa: E402
from agent.agent_config import get_agent_config  # noqa: E402
from agent.env import load_env  # noqa: E402
from src.data.dataset import AUTHENTIC, TAMPERED, synthetic_records  # noqa: E402


def _models() -> list[dict[str, Any]]:
    """Return all reasoning candidates from the checked-in candidate list."""
    import yaml

    path = Path(__file__).resolve().parent.parent / "configs" / "agent_candidates.yaml"
    with path.open(encoding="utf-8") as handle:
        source = yaml.safe_load(handle)
    return list(source["candidates"])


def _candidate_endpoint(candidate: dict[str, Any]) -> dict[str, Any]:
    config = get_agent_config().copy()
    config.update({key: candidate[key] for key in ("provider", "model", "base_url", "api_key_env")})
    config["fallback_models"] = []
    config["memory"] = {"enabled": False, "path": ""}
    return config


def _run_candidate(candidate: dict[str, Any], cases: list[tuple[str, str, str]]) -> bool:
    endpoint = _candidate_endpoint(candidate)
    monkey_config = endpoint
    monkey_entries = [
        {key: candidate[key] for key in ("provider", "model", "base_url", "api_key_env")}
    ]
    original_config, original_entries = loop.get_agent_config, loop.model_entries
    loop.get_agent_config = lambda: monkey_config
    loop.model_entries = lambda _config: monkey_entries
    try:
        key_name = candidate["api_key_env"]
        if not os.environ.get(key_name):
            print(f"{candidate['name']}: SKIP (missing {key_name})")
            return False
        passed = True
        for label, path, goal in cases:
            answer, messages = loop.run_agent(goal, path, use_memory=False)
            first_tool = next(
                (index for index, item in enumerate(messages) if item.get("role") == "tool"),
                None,
            )
            plan = next(
                (
                    item.get("content", "")
                    for item in messages
                    if item.get("role") == "assistant" and item.get("content")
                ),
                "",
            )
            calls = [
                item
                for item in messages
                if item.get("role") == "assistant" and item.get("tool_calls")
            ]
            first_call_message = calls[0] if calls else {}
            ordered_plan = bool(
                first_call_message.get("content")
                and first_tool is not None
                and messages.index(first_call_message) < first_tool
            )
            plan = str(first_call_message.get("content") or "")
            case_pass = bool(calls and ordered_plan)
            passed = passed and case_pass
            outcome = "PASS" if case_pass else "FAIL"
            print(f"candidate={candidate['name']} case={label} result={outcome}")
            print(f"plan_before_first_tool={ordered_plan} plan={plan[:500]}")
            for message in calls:
                for call in message["tool_calls"]:
                    name = call["function"]["name"]
                    args = call["function"]["arguments"]
                    call_id = call["id"]
                    result = next(
                        (
                            m["content"]
                            for m in messages
                            if m.get("role") == "tool"
                            and m.get("tool_call_id") == call_id
                        ),
                        "",
                    )
                    print(f"tool={name} arguments={args} result_size={len(result.encode())}")
            if label == "tampered" and calls:
                sequence = [
                    call["function"]["name"]
                    for message in calls
                    for call in message["tool_calls"]
                ]
                print(f"behavior_sequence_tampered={sequence}")
            print(f"answer={answer[:500]}")
            time.sleep(2)
        return passed
    finally:
        loop.get_agent_config, loop.model_entries = original_config, original_entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", help="candidate name, or run every candidate")
    args = parser.parse_args()
    load_env()
    candidates = _models()
    if args.candidate:
        candidates = [item for item in candidates if item["name"] == args.candidate]
        if not candidates:
            parser.error(f"unknown candidate: {args.candidate}")

    with tempfile.TemporaryDirectory(prefix="agent-smoke-") as temporary:
        root = Path(temporary)
        records = synthetic_records(root / "synthetic", n=1, size=256, seed=19)
        authentic = next(row for row in records if row[1] == AUTHENTIC)
        tampered = next(row for row in records if row[1] == TAMPERED)
        corrupt = root / "corrupt.jpg"
        corrupt.write_bytes(b"not an image")
        cases = [
            ("authentic", authentic[0], "Check this authentic synthetic test image."),
            ("tampered", tampered[0], "Check this tampered synthetic test image."),
            ("corrupt", str(corrupt), "Check this corrupt test file."),
        ]
        os.environ[loop.ENV_UPLOAD_ROOT] = str(root)
        results = [_run_candidate(candidate, cases) for candidate in candidates]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
