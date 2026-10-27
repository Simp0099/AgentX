"""Run the configured live agent on a balanced 20-image benchmark.

CASIA2 in this checkout has no ground-truth masks or declared test split, so
this runner uses the repository's synthetic generator and labels the report
accordingly. No prompt or verdict threshold is changed for evaluation.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import loop  # noqa: E402
from agent.env import load_env  # noqa: E402
from agent.schemas import Verdict  # noqa: E402
from src.data.dataset import AUTHENTIC, TAMPERED, synthetic_records  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "agent_eval.json"


def _tool_names(messages: list[dict[str, Any]]) -> list[str]:
    return [
        call["function"]["name"]
        for message in messages
        for call in message.get("tool_calls", [])
    ]


def main() -> int:
    load_env()
    config = loop.get_agent_config()
    endpoints = loop.model_entries(config)
    available = [entry for entry in endpoints if os.environ.get(entry["api_key_env"])]
    if not available:
        payload = {
            "model": config["model"],
            "date": date.today().isoformat(),
            "dataset_type": "synthetic",
            "dataset_size": 20,
            "status": "not_run",
            "reason": "No configured model credential is present.",
            "verdict_accuracy": None,
            "average_tool_calls_per_image": None,
            "uncertain_or_needs_review_cases": None,
            "failure_count": None,
            "behavior_checks": {
                "different_images_different_tool_sequences": "not_run",
                "later_tool_call_depended_on_prior_result": "not_run",
            },
        }
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("Benchmark not run: no configured model credential is present.")
        return 2

    with tempfile.TemporaryDirectory(prefix="agent-benchmark-") as temporary:
        root = Path(temporary)
        source = synthetic_records(root / "source", n=10, size=256, seed=2026)
        uploads = root / "uploads"
        uploads.mkdir()
        dataset = []
        for path, label, _bbox in source:
            destination = uploads / Path(path).name
            shutil.copy2(path, destination)
            dataset.append((destination, label))
        os.environ[loop.ENV_UPLOAD_ROOT] = str(uploads)
        outcomes = []
        for path, label in dataset:
            try:
                result = loop.run_agent_full(
                    "Determine whether this image appears manipulated, using available evidence.",
                    str(path),
                    use_memory=False,
                )
                verdict = result.grounded_verdict
                needs_review = verdict == Verdict.FLAGGED_FOR_REVIEW
                predicted = (
                    TAMPERED
                    if verdict in {
                        Verdict.FLAGGED_FOR_REVIEW,
                        Verdict.FLAGGED_REGION_UNAVAILABLE,
                    }
                    else AUTHENTIC
                    if verdict == Verdict.NOT_FLAGGED_BY_THIS_DETECTOR
                    else None
                )
                outcomes.append(
                    {
                        "label": label,
                        "predicted": predicted,
                        "correct": predicted == label,
                        "calls": len(_tool_names(result.messages)),
                        "uncertain": predicted is None,
                        "needs_review": needs_review,
                        "failed": result.stopped_reason != "final_answer"
                        or bool(result.errors),
                        "sequence": _tool_names(result.messages),
                        "messages": result.messages,
                    }
                )
                print(
                    f"{path.name}: label={label} predicted={predicted} "
                    f"tools={len(outcomes[-1]['sequence'])} "
                    f"status={result.stopped_reason}"
                )
            except Exception as exc:
                outcomes.append(
                    {
                        "label": label,
                        "predicted": None,
                        "correct": False,
                        "calls": 0,
                        "uncertain": True,
                        "needs_review": False,
                        "failed": True,
                        "sequence": [],
                        "error_type": type(exc).__name__,
                    }
                )

    accuracy = sum(bool(item["correct"]) for item in outcomes) / len(outcomes)
    sequences = [item["sequence"] for item in outcomes]
    distinct_sequences = len(set(tuple(item) for item in sequences)) > 1
    dependency = None
    for item in outcomes:
        messages = item.get("messages", [])
        for index, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            result_text = message.get("content", "")
            result = messages[index + 1] if index + 1 < len(messages) else {}
            calls = result.get("tool_calls", [])
            if calls:
                dependency = {
                    "prior_tool_message": {
                        "tool_call_id": message.get("tool_call_id"),
                        "content": result_text[:500],
                    },
                    "later_tool_message": calls[0],
                }
                break
        if dependency:
            break

    payload = {
        "model": config["model"],
        "date": date.today().isoformat(),
        "dataset_type": "synthetic",
        "dataset_size": len(outcomes),
        "verdict_accuracy": round(accuracy, 4),
        "average_tool_calls_per_image": round(
            sum(item["calls"] for item in outcomes) / len(outcomes), 4
        ),
        "uncertain_or_needs_review_cases": sum(
            item["uncertain"] or item["needs_review"] for item in outcomes
        ),
        "failure_count": sum(item["failed"] for item in outcomes),
        "behavior_checks": {
            "different_images_different_tool_sequences": {
                "passed": distinct_sequences,
                "sequences": sequences,
            },
            "later_tool_call_depended_on_prior_result": dependency,
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in payload.items() if key != "behavior_checks"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
