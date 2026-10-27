"""Explicit live smoke tests; excluded from normal CI by the pytest config."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import loop  # noqa: E402
from agent.agent_config import model_entries  # noqa: E402
from agent.env import load_env  # noqa: E402


@pytest.mark.live
def test_live_model_calls_a_tool(tmp_path, monkeypatch):
    load_env()
    endpoints = model_entries()
    if not any(os.environ.get(item["api_key_env"]) for item in endpoints):
        pytest.skip("no provider credential is configured")
    image = tmp_path / "live.png"
    Image.new("RGB", (32, 32), (100, 120, 140)).save(image)
    monkeypatch.setenv(loop.ENV_UPLOAD_ROOT, str(tmp_path))

    result = loop.run_agent_full("Inspect this test image.", str(image), use_memory=False)

    assert any(message.get("tool_calls") for message in result.messages)
    first_tool = next(i for i, message in enumerate(result.messages) if message.get("role") == "tool")
    first_call = next(
        message for message in result.messages if message.get("role") == "assistant" and message.get("tool_calls")
    )
    first_call_index = result.messages.index(first_call)
    assert first_call.get("content"), "plan text should precede the first tool call"
    assert first_call_index < first_tool
