"""Offline coverage for Phase 8 failure handling and reflection."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from agent import loop
from agent.agent_tools import ToolContext
from agent.prompts import SYSTEM_PROMPT
from tests.fakes import FakeClient, make_message, make_tool_call, tool_messages


def _config(monkeypatch, tmp_path, *, reflection=False):
    config = loop.get_agent_config().copy()
    config["memory"] = {"enabled": False, "path": str(tmp_path / "memory.db")}
    config["reflection"] = reflection
    monkeypatch.setattr(loop, "get_agent_config", lambda: config)
    monkeypatch.setenv(loop.ENV_UPLOAD_ROOT, str(tmp_path))
    monkeypatch.setattr(loop, "BACKOFF_SECONDS", (0.0, 0.0))


def _image(tmp_path):
    path = tmp_path / "valid.png"
    Image.fromarray(np.full((20, 20, 3), 120, dtype=np.uint8)).save(path)
    return path


def test_tool_error_is_returned_to_model_and_reflected_in_next_turn(tmp_path, monkeypatch):
    _config(monkeypatch, tmp_path)
    missing = tmp_path / "missing.png"
    client = FakeClient(
        [
            make_message(
                tool_calls=[make_tool_call("inspect_image", {"image_path": str(missing)})]
            ),
            make_message(content="The inspect_image tool reported that the file was not found."),
        ]
    )

    result = loop.run_agent_full("quick_check", str(missing), client=client, use_memory=False)

    observed = tool_messages(client.requests[1])
    assert json.loads(observed[0]["content"])["error"]
    assert "file was not found" in result.answer
    assert result.confidence == "low"
    assert "Needs human review:" in result.answer


def test_localizer_failure_keeps_ela_and_metadata_available(tmp_path, monkeypatch):
    _config(monkeypatch, tmp_path)
    path = _image(tmp_path)

    class BrokenDetector:
        def predict_raw(self, _path):
            raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(
        loop, "ToolContext", lambda **kwargs: ToolContext(detector=BrokenDetector(), **kwargs)
    )
    client = FakeClient(
        [
            make_message(
                tool_calls=[make_tool_call("run_localizer", {"image_path": str(path)})]
            ),
            make_message(
                tool_calls=[
                    make_tool_call("generate_ela", {"image_path": str(path)}, "ela_call"),
                    make_tool_call(
                        "inspect_metadata", {"image_path": str(path)}, "metadata_call"
                    ),
                ]
            ),
            make_message(
                content=(
                    "run_localizer failed because the checkpoint was unavailable. "
                    "I could still run generate_ela and inspect_metadata."
                )
            ),
        ]
    )

    result = loop.run_agent_full("detailed", str(path), client=client, use_memory=False)

    assert result.record.observations[0].status == "error"
    assert {item.tool for item in result.record.observations} >= {
        "run_localizer", "generate_ela", "inspect_metadata"
    }
    assert "run_localizer failed" in result.answer
    assert "generate_ela" in result.answer and "inspect_metadata" in result.answer
    assert "run_localizer fails" in SYSTEM_PROMPT


def test_third_identical_failing_call_is_blocked_after_two_executions(tmp_path, monkeypatch):
    _config(monkeypatch, tmp_path)
    path = _image(tmp_path)

    class BrokenDetector:
        calls = 0

        def predict_raw(self, _path):
            self.calls += 1
            raise RuntimeError("checkpoint unavailable")

    detector = BrokenDetector()
    monkeypatch.setattr(
        loop, "ToolContext", lambda **kwargs: ToolContext(detector=detector, **kwargs)
    )
    repeated = [
        make_message(
            tool_calls=[make_tool_call("run_localizer", {"image_path": str(path)}, f"c{i}")]
        )
        for i in range(3)
    ]
    client = FakeClient(repeated)

    result = loop.run_agent_full("quick_check", str(path), client=client, use_memory=False)

    assert detector.calls == 2
    blocked = json.loads(result.messages[-1]["content"])
    assert blocked["error"] == "retry limit reached; change the arguments or use another tool"
    assert result.stopped_reason == "retry_limit"
    assert "retry limit" in result.answer
    assert len(client.requests) == 3


def test_borderline_score_adds_review_line_and_low_confidence(tmp_path, monkeypatch):
    _config(monkeypatch, tmp_path)
    path = _image(tmp_path)
    monkeypatch.setattr(
        loop,
        "run_tool",
        lambda name, _args, _ctx: {"tampered_prob": 0.5}
        if name == "run_localizer"
        else {"issues": []},
    )
    client = FakeClient(
        [
            make_message(
                tool_calls=[make_tool_call("run_localizer", {"image_path": str(path)})]
            ),
            make_message(content="The score is borderline; this remains inconclusive."),
        ]
    )

    result = loop.run_agent_full("quick_check", str(path), client=client, use_memory=False)

    assert result.confidence == "low"
    assert "Needs human review:" in result.answer
    assert "uncertain range" in result.answer
    assert "Confidence: low." in result.answer


def test_reflection_disabled_makes_one_model_call(tmp_path, monkeypatch):
    _config(monkeypatch, tmp_path, reflection=False)
    client = FakeClient([make_message(content="This image is tampered.")])

    loop.run_agent_full("quick_check", str(_image(tmp_path)), client=client, use_memory=False)

    assert len(client.requests) == 1


def test_reflection_accepts_only_explicit_corrected_answer(tmp_path, monkeypatch):
    _config(monkeypatch, tmp_path, reflection=True)
    client = FakeClient(
        [
            make_message(content="This image is tampered, though no tool evidence exists."),
            make_message(content="CORRECTED: Evidence is insufficient to call this tampered."),
        ]
    )

    result = loop.run_agent_full("quick_check", str(_image(tmp_path)), client=client, use_memory=False)

    assert len(client.requests) == 2
    assert result.answer.startswith("Evidence is insufficient")
    assert "This image is tampered" not in result.answer


def test_reflection_does_not_replace_answer_without_corrected_marker():
    original = "Original answer."
    client = FakeClient([make_message(content="A revised answer without the marker.")])

    assert loop.reflect(original, [], client, "test-model") == original
