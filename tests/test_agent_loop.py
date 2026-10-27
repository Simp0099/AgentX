"""Offline tests for agent/loop.py.

Every response is scripted by :class:`tests.fakes.FakeClient`; no network call,
no credential and no TensorFlow import happens here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import loop  # noqa: E402
from agent.agent_config import get_agent_config, model_entries  # noqa: E402
from agent.schemas import Verdict  # noqa: E402
from tests.fakes import FakeClient, make_message, make_tool_call, tool_messages  # noqa: E402


@pytest.fixture(autouse=True)
def _zero_backoff(monkeypatch, tmp_path):
    monkeypatch.setattr(loop, "BACKOFF_SECONDS", (0.0, 0.0))
    config = get_agent_config().copy()
    config["memory"] = {"enabled": True, "path": str(tmp_path / "memory.sqlite")}
    monkeypatch.setattr(loop, "get_agent_config", lambda: config)


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    directory = tmp_path / "uploads"
    directory.mkdir()
    monkeypatch.setenv(loop.ENV_UPLOAD_ROOT, str(directory))
    return directory


@pytest.fixture
def image(uploads):
    path = uploads / "photo.png"
    Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(path)
    return path


def _inspect_call(path, call_id="call_1"):
    return make_tool_call("inspect_image", {"image_path": str(path)}, call_id=call_id)


# ---------------------------------------------------------------------------
# (a) direct final answer
# ---------------------------------------------------------------------------


def test_direct_final_answer_returns_text_and_messages(image):
    client = FakeClient([make_message(content="Verdict: nothing to flag.")])

    answer, messages = loop.run_agent("quick_check", str(image), client=client)

    assert answer.startswith("Verdict: nothing to flag.")
    assert "Confidence: medium." in answer
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    user_turn = next(item for item in reversed(messages) if item["role"] == "user")
    assert "quick_check" in user_turn["content"]
    assert str(image) in user_turn["content"]
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == answer
    assert len(client.requests) == 1


# ---------------------------------------------------------------------------
# (b) tool call, then final answer
# ---------------------------------------------------------------------------


def test_tool_result_reaches_the_next_request(image):
    client = FakeClient(
        [
            make_message(tool_calls=[_inspect_call(image)]),
            make_message(content="Two of two."),
        ]
    )

    answer, _messages = loop.run_agent("quick_check", str(image), client=client)

    assert answer.startswith("Two of two.")
    assert len(client.requests) == 2

    # The full history is resent: system, user, assistant(tool_calls), tool.
    second = client.requests[1]
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]

    tool_msgs = tool_messages(second)
    assert len(tool_msgs) == 1
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["valid"] is True
    assert payload["width"] == 16
    assert tool_msgs[0]["tool_call_id"] == "call_1"
    # No optional `name` field: strict endpoints reject it.
    assert "name" not in tool_msgs[0]

    # The assistant turn is echoed back so the model can pair call and reply.
    echoed = second["messages"][-2]
    assert echoed["tool_calls"][0]["function"]["name"] == "inspect_image"


def test_record_is_built_from_the_real_tool_calls(image):
    client = FakeClient(
        [
            make_message(tool_calls=[_inspect_call(image)]),
            make_message(content="done"),
        ]
    )

    result = loop.run_agent_full("quick_check", str(image), client=client)

    assert len(result.record.observations) == 1
    observation = result.record.observations[0]
    assert observation.tool == "inspect_image"
    assert observation.status == "ok"
    assert observation.result["sha256"]
    assert result.record.image_id == observation.result["sha256"]


# ---------------------------------------------------------------------------
# (c) malformed tool call
# ---------------------------------------------------------------------------


def test_malformed_json_arguments_are_answered_and_the_loop_continues(image):
    broken = make_tool_call("inspect_image", "{not json", call_id="call_bad")
    client = FakeClient(
        [
            make_message(tool_calls=[broken]),
            make_message(content="recovered"),
        ]
    )

    answer, _messages = loop.run_agent("quick_check", str(image), client=client)

    assert answer.startswith("recovered")
    assert len(client.requests) == 2

    payload = json.loads(tool_messages(client.requests[1])[0]["content"])
    assert "error" in payload
    assert "hint" in payload
    assert "invalid JSON arguments" in payload["error"]


def test_non_object_json_arguments_are_treated_as_malformed(image):
    broken = make_tool_call("inspect_image", "[1, 2]")
    client = FakeClient(
        [
            make_message(tool_calls=[broken]),
            make_message(content="recovered"),
        ]
    )

    answer, _messages = loop.run_agent("quick_check", str(image), client=client)

    assert answer.startswith("recovered")
    payload = json.loads(tool_messages(client.requests[1])[0]["content"])
    assert "invalid JSON arguments" in payload["error"]


# ---------------------------------------------------------------------------
# (d) a fake that always asks for a tool
# ---------------------------------------------------------------------------


def test_always_tool_fake_stops_at_max_steps(image):
    responses = [
        make_message(
            tool_calls=[
                make_tool_call(
                    "inspect_image",
                    {"image_path": str(image.with_name(f"photo_{i}.png"))},
                    call_id=f"call_{i}",
                )
            ]
        )
        for i in range(10)
    ]
    client = FakeClient(responses)

    answer, messages = loop.run_agent("quick_check", str(image), max_steps=2, client=client)

    assert answer == loop.STEP_LIMIT_MESSAGE
    assert "step limit" in answer
    assert len(client.requests) == 2
    assert messages  # the partial conversation is still returned


def test_step_limit_comes_from_config_when_not_supplied(image):
    configured = int(get_agent_config()["max_steps"])
    responses = [
        make_message(
            tool_calls=[
                make_tool_call(
                    "inspect_image",
                    {"image_path": str(image.with_name(f"photo_{i}.png"))},
                    call_id=f"call_{i}",
                )
            ]
        )
        for i in range(20)
    ]
    client = FakeClient(responses)

    answer, _messages = loop.run_agent("quick_check", str(image), client=client)

    assert answer == loop.STEP_LIMIT_MESSAGE
    assert len(client.requests) == configured == 6


# ---------------------------------------------------------------------------
# (e) provider error, then the fallback model
# ---------------------------------------------------------------------------


def _provider_errors(count: int):
    return [RuntimeError("429 too many requests") for _ in range(count)]


def test_provider_errors_retries_then_falls_back_to_the_next_model(image):
    endpoints = model_entries(get_agent_config())
    primary = endpoints[0]["model"]
    fallback = endpoints[1]["model"]
    failures = _provider_errors(loop.MAX_PROVIDER_RETRIES + 1)  # initial attempt + 2 retries
    client = FakeClient([*failures, make_message(content="ok from the fallback")])

    answer, _messages = loop.run_agent("quick_check", str(image), client=client)

    assert answer.startswith("ok from the fallback")
    models = [request["model"] for request in client.requests]
    # Three failed attempts against the primary (attempt + two retries), then
    # the switch to the fallback model succeeds.
    assert models == [primary, primary, primary, fallback]
    assert len(models) == (loop.MAX_PROVIDER_RETRIES + 2)


def test_every_endpoint_failing_returns_a_structured_failure(image):
    endpoints = model_entries(get_agent_config())
    client = FakeClient(_provider_errors((loop.MAX_PROVIDER_RETRIES + 1) * len(endpoints)))

    answer, _messages = loop.run_agent("quick_check", str(image), client=client)

    assert answer.startswith("Model unavailable:")
    assert answer != loop.STEP_LIMIT_MESSAGE


def test_structured_failure_is_not_a_finished_answer(image):
    client = FakeClient([RuntimeError("connection reset") for _ in range(50)])

    result = loop.run_agent_full("quick_check", str(image), client=client)

    assert result.stopped_reason == "model_unavailable"
    assert result.answer.startswith("Model unavailable:")
    assert result.errors


# ---------------------------------------------------------------------------
# (f) logging
# ---------------------------------------------------------------------------


def test_one_log_line_per_tool_call(tmp_path, monkeypatch, image):
    log_path = tmp_path / "agent.log"
    monkeypatch.setattr(loop, "LOG_PATH", log_path)
    client = FakeClient(
        [
            make_message(tool_calls=[_inspect_call(image)]),
            make_message(content="done"),
        ]
    )

    loop.run_agent("quick_check", str(image), client=client)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, lines
    line = lines[0]
    assert "tool=inspect_image" in line
    assert "status=ok" in line
    assert "bytes=" in line
    assert "args=" in line
    assert "api_key" not in line
    # The full result is never written, only its size.
    assert "sha256" not in line


def test_no_final_answer_call_writes_no_log_line(tmp_path, monkeypatch, image):
    log_path = tmp_path / "agent.log"
    monkeypatch.setattr(loop, "LOG_PATH", log_path)
    client = FakeClient([make_message(content="straight to an answer")])

    loop.run_agent("quick_check", str(image), client=client)

    assert not log_path.exists()


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


def test_grounding_keeps_the_model_text_and_computes_the_verdict(image):
    client = FakeClient(
        [
            make_message(tool_calls=[_inspect_call(image)]),
            make_message(content="The image looks fine to me."),
        ]
    )

    result = loop.run_agent_full("quick_check", str(image), client=client)

    # The model's words are untouched; the verdict is computed alongside.
    assert result.answer.startswith("The image looks fine to me.")
    assert isinstance(result.grounded_verdict, Verdict)
    assert result.record.verdict == result.grounded_verdict
    assert result.stopped_reason == "final_answer"
    assert result.steps == 2
