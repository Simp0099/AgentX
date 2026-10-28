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
    loop._MODEL_COOLDOWNS.clear()


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
    assert "Confidence:" not in result.answer
    assert "Limits you must state rather than hide" in result.answer


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


def test_quota_skips_primary_without_retry_and_uses_first_fallback(tmp_path, monkeypatch):
    _config(monkeypatch, tmp_path)
    endpoints = loop.model_entries(loop.get_agent_config())
    client = FakeClient([
        RuntimeError("429 insufficient quota"),
        make_message(content="fallback response"),
    ])

    result = loop.run_agent_full("quick_check", str(_image(tmp_path)), client=client, use_memory=False)

    assert result.answer.startswith("fallback response")
    assert [request["model"] for request in client.requests] == [
        endpoints[0]["model"], endpoints[1]["model"]
    ]


def test_transient_retries_twice_then_falls_back(tmp_path, monkeypatch):
    _config(monkeypatch, tmp_path)
    endpoints = loop.model_entries(loop.get_agent_config())
    client = FakeClient([
        TimeoutError("request timed out"),
        ConnectionError("connection reset"),
        RuntimeError("503 service unavailable"),
        make_message(content="fallback response"),
    ])

    result = loop.run_agent_full("quick_check", str(_image(tmp_path)), client=client, use_memory=False)

    assert result.answer.startswith("fallback response")
    assert [request["model"] for request in client.requests] == [
        endpoints[0]["model"], endpoints[0]["model"], endpoints[0]["model"],
        endpoints[1]["model"],
    ]


def test_fatal_error_falls_back_to_next_provider(tmp_path, monkeypatch):
    _config(monkeypatch, tmp_path)
    endpoints = loop.model_entries(loop.get_agent_config())
    client = FakeClient([RuntimeError("400 invalid request: malformed tool schema")])

    result = loop.run_agent_full("quick_check", str(_image(tmp_path)), client=client, use_memory=False)

    assert result.stopped_reason == "model_unavailable"
    assert len(client.requests) == len(endpoints)
    assert endpoints[0]["model"] in result.errors[0]


def test_cooled_down_model_is_skipped_on_next_turn(tmp_path, monkeypatch):
    _config(monkeypatch, tmp_path)
    endpoints = loop.model_entries(loop.get_agent_config())
    path = str(_image(tmp_path))
    first_client = FakeClient([
        RuntimeError("429 quota exceeded"),
        make_message(content="fallback response"),
    ])
    loop.run_agent_full("quick_check", path, client=first_client, use_memory=False)

    next_client = FakeClient([make_message(content="next fallback response")])
    next_result = loop.run_agent_full("quick_check", path, client=next_client, use_memory=False)

    assert next_result.answer.startswith("next fallback response")
    assert [request["model"] for request in next_client.requests] == [endpoints[1]["model"]]


def test_all_models_exhausted_returns_model_unavailable(tmp_path, monkeypatch):
    _config(monkeypatch, tmp_path)
    endpoints = loop.model_entries(loop.get_agent_config())
    client = FakeClient([RuntimeError("429 quota exhausted") for _ in endpoints])

    result = loop.run_agent_full("quick_check", str(_image(tmp_path)), client=client, use_memory=False)

    assert result.stopped_reason == "model_unavailable"
    assert result.answer.startswith("Model unavailable:")
    assert all(item["model"] in result.answer for item in endpoints)
    assert len(client.requests) == len(endpoints)


def test_unconfigured_provider_keys_are_skipped_before_routing(tmp_path, monkeypatch):
    _config(monkeypatch, tmp_path)
    endpoints = loop.model_entries(loop.get_agent_config())
    monkeypatch.setattr(loop, "load_env", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")
    monkeypatch.delenv("OPENCODE_ZEN_API_KEY", raising=False)

    created = []

    def build(endpoint):
        created.append(endpoint)
        if endpoint["api_key_env"] != "OPENROUTER_API_KEY":
            return None
        return FakeClient([make_message(content="OpenRouter free fallback response")])

    monkeypatch.setattr(loop, "_build_client", build)
    result = loop.run_agent_full(
        "quick_check", str(_image(tmp_path)), client=None, use_memory=False
    )

    assert result.answer.startswith("OpenRouter free fallback response")
    assert created[-1]["model"] == endpoints[1]["model"]
    assert created[-1]["api_key_env"] == "OPENROUTER_API_KEY"


def test_no_configured_free_model_key_returns_clear_unavailable(tmp_path, monkeypatch):
    _config(monkeypatch, tmp_path)
    endpoints = loop.model_entries(loop.get_agent_config())
    monkeypatch.setattr(loop, "load_env", lambda: None)
    for name in {entry["api_key_env"] for entry in endpoints}:
        monkeypatch.delenv(name, raising=False)

    result = loop.run_agent_full(
        "quick_check", str(_image(tmp_path)), client=None, use_memory=False
    )

    assert result.stopped_reason == "model_unavailable"
    assert result.answer.startswith("Model unavailable: no configured free-model credentials")
    assert all(entry["model"] in result.answer for entry in endpoints)


def test_retry_after_is_capped_by_configured_maximum(tmp_path, monkeypatch):
    _config(monkeypatch, tmp_path)
    config = loop.get_agent_config()
    config["retry_after_max_seconds"] = 3
    sleeps = []
    monkeypatch.setattr(loop.time, "sleep", sleeps.append)

    class ProviderError(RuntimeError):
        status_code = 503
        headers = {"Retry-After": "90"}

    client = FakeClient([
        ProviderError("temporary service error"),
        make_message(content="retry succeeded"),
    ])
    result = loop.run_agent_full("quick_check", str(_image(tmp_path)), client=client, use_memory=False)

    assert result.answer.startswith("retry succeeded")
    assert sleeps == [3]
