"""Tests for the LLM-driven agent layer (no network, no real LLM).

These tests drive ``run_llm_investigation`` with a scripted LLM client, so the
model checkpoint is never loaded and no network call is made. Tool callables
are monkeypatched with in-process fakes so the detector is never invoked.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import get_config  # noqa: E402
from agent.llm_client import ScriptedLLMClient, LLMNotConfigured  # noqa: E402
from agent.orchestrator import run_llm_investigation  # noqa: E402
from agent.schemas import ToolResult, Verdict  # noqa: E402
from agent.state import render_report_from_observations  # noqa: E402
from agent.tool_registry import tool_registry  # noqa: E402

MAX_STEPS = 5


def _valid_tool_targets(monkeypatch):
    """Point the approved tools at fast fakes so tests never load the model."""
    targets = {
        "inspect_image": lambda path, timeout=None: ToolResult(
            "inspect_image", "ok", {"valid": True, "width": 8, "height": 8, "sha256": "aaaa"}
        ),
        "run_localizer": lambda path, threshold, detector=None, timeout=None: ToolResult(
            "run_localizer",
            "ok",
            {"score": 0.2, "tampered_prob": 0.2, "bbox": [0.1, 0.1, 0.9, 0.9], "label": "fake"},
        ),
        "validate_prediction": lambda run_result, image_width, image_height, timeout=None: ToolResult(
            "validate_prediction", "ok", {"valid": True, "issues": []}
        ),
        "generate_ela": lambda path, quality=None, timeout=None: ToolResult(
            "generate_ela", "ok", {"ela_map": "x" * 100}
        ),
        "inspect_metadata": lambda path, timeout=None: ToolResult(
            "inspect_metadata", "ok", {"frames": 1, "camera": "synthetic"}
        ),
        "compare_evidence": lambda record, timeout=None: ToolResult(
            "compare_evidence", "ok", {"agreements": [], "conflicts": []}
        ),
    }
    for name, fn in targets.items():
        monkeypatch.setitem(tool_registry[name], "callable", fn)
    monkeypatch.setattr("agent.tool_registry._upload_root", lambda cfg=None: None)


def _image_file(tmp_path):
    import numpy as np
    from PIL import Image

    p = tmp_path / "img.png"
    Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8)).save(p)
    return p


class ScriptedLLMClient:
    """Deterministic LLM with no network. Returns scripted tool calls."""

    def __init__(self, responses):
        self._responses = list(responses)

    def choose_action(self, goal, observations, tool_schemas, timeout_seconds=None):
        if not self._responses:
            return SimpleNamespace(finished=True, tool=None, arguments=None)
        response = self._responses.pop(0)
        return SimpleNamespace(
            finished=bool(response.get("finished", False)),
            tool=response.get("tool"),
            arguments=response.get("arguments"),
        )


def _raise(exc):
    """Return a callable that always raises ``exc`` (for monkeypatching)."""

    def _f(*args, **kwargs):
        raise exc

    return _f


def test_scripted_llm_requests_unapproved_tool_and_is_rejected(image, monkeypatch):
    _valid_tool_targets(monkeypatch)
    client = ScriptedLLMClient(
        [
            {"tool": "run_shell", "arguments": {}, "finished": False},
            {"tool": "inspect_image", "arguments": {"path": str(image)}, "finished": False},
            {"finished": True},
        ]
    )
    record = run_llm_investigation(str(image), llm_client=client, max_steps=MAX_STEPS)

    assert len(record.observations) == 2
    assert record.observations[0].tool == "run_shell"
    assert record.observations[0].status == "rejected"
    assert "unapproved tool name" in record.observations[0].rationale
    assert record.observations[1].tool == "inspect_image"
    assert record.observations[1].status == "ok"
    assert any("unapproved tool name" in e for e in record.errors)
    assert record.verdict is not None


def test_scripted_llm_requests_invalid_args_and_loop_continues(image, monkeypatch):
    _valid_tool_targets(monkeypatch)
    client = ScriptedLLMClient(
        [
            {"tool": "generate_ela", "arguments": {"path": "/tmp/x", "quality": "high"}, "finished": False},
            {"tool": "generate_ela", "arguments": {"path": str(image), "quality": 90}, "finished": False},
            {"finished": True},
        ]
    )
    record = run_llm_investigation(str(image), llm_client=client, max_steps=MAX_STEPS)

    assert len(record.observations) == 2
    assert record.observations[0].tool == "generate_ela"
    assert record.observations[0].status == "rejected"
    assert "invalid argument" in record.observations[0].rationale
    assert record.observations[1].tool == "generate_ela"
    assert record.observations[1].status == "ok"
    assert any("invalid argument" in e for e in record.errors)
    assert record.verdict is not None


def test_scripted_llm_never_finishes_stops_at_max_steps(image, monkeypatch):
    _valid_tool_targets(monkeypatch)
    monkeypatch.setitem(
        tool_registry["run_localizer"],
        "callable",
        _raise(RuntimeError("boom")),
    )
    client = ScriptedLLMClient(
        [
            {"tool": "run_localizer", "arguments": {"path": str(image), "threshold": 0.5}, "finished": False}
            for _ in range(MAX_STEPS)
        ]
    )
    record = run_llm_investigation(str(image), llm_client=client, max_steps=MAX_STEPS)

    assert len(record.observations) == MAX_STEPS
    assert all(obs.status == "error" for obs in record.observations)
    assert all("boom" in e for e in record.errors)
    assert record.verdict == Verdict.INCONCLUSIVE


def test_tool_raises_and_loop_continues(image, monkeypatch):
    _valid_tool_targets(monkeypatch)
    monkeypatch.setitem(
        tool_registry["run_localizer"],
        "callable",
        _raise(RuntimeError("boom")),
    )
    client = ScriptedLLMClient(
        [
            {"tool": "run_localizer", "arguments": {"path": str(image), "threshold": 0.5}, "finished": False},
            {"tool": "inspect_image", "arguments": {"path": str(image)}, "finished": False},
            {"finished": True},
        ]
    )
    record = run_llm_investigation(str(image), llm_client=client, max_steps=MAX_STEPS)

    assert len(record.observations) == 2
    assert record.observations[0].tool == "run_localizer"
    assert record.observations[0].status == "error"
    assert "boom" in record.observations[0].result
    assert record.observations[1].tool == "inspect_image"
    assert record.observations[1].status == "ok"
    assert any("boom" in e for e in record.errors)
    assert record.verdict is not None


def test_tool_timeouts_and_loop_continues(image, monkeypatch):
    _valid_tool_targets(monkeypatch)

    def slow_inspect(path, timeout=None):
        import time

        time.sleep(0.2)
        return {"valid": True, "width": 8, "height": 8, "sha256": "bbbb"}

    # A tool that ignores the per-tool timeout and burns time: the dispatcher
    # wraps the resulting exception as an "error" and the loop keeps going.
    monkeypatch.setitem(tool_registry["inspect_image"], "callable", slow_inspect)
    client = ScriptedLLMClient(
        [
            {"tool": "inspect_image", "arguments": {"path": str(image)}, "finished": False},
            {"finished": True},
        ]
    )
    record = run_llm_investigation(str(image), llm_client=client, max_steps=MAX_STEPS, tool_timeout=0.05)

    assert len(record.observations) == 1
    assert record.observations[0].tool == "inspect_image"
    assert record.observations[0].status == "error"
    assert "timed out" in record.observations[0].result.lower()
    assert any("timed out" in e for e in record.errors)
    assert record.verdict is not None


def test_conflicting_observations_yeild_inconclusive(image, monkeypatch):
    _valid_tool_targets(monkeypatch)
    monkeypatch.setitem(
        tool_registry["run_localizer"],
        "callable",
        lambda path, threshold, detector=None, timeout=None: ToolResult(
            "run_localizer",
            "ok",
            {"score": 0.9, "tampered_prob": 0.9, "bbox": [0.8, 0.2, 0.1, 0.9], "label": "fake"},
        ),
    )
    monkeypatch.setitem(
        tool_registry["validate_prediction"],
        "callable",
        lambda run_result, image_width, image_height, timeout=None: {
            "tool": "validate_prediction",
            "status": "ok",
            "result": {"valid": False, "issues": ["box inverted"]},
        },
    )
    monkeypatch.setitem(
        tool_registry["compare_evidence"],
        "callable",
        lambda record, timeout=None: {
            "tool": "compare_evidence",
            "status": "ok",
            "result": {"agreements": [], "conflicts": ["score-box conflict"]},
        },
    )
    client = ScriptedLLMClient(
        [
            {"tool": "run_localizer", "arguments": {"path": str(image), "threshold": 0.5}, "finished": False},
            {
                "tool": "validate_prediction",
                "arguments": {"run_result": {"score": 0.9}, "image_width": 8, "image_height": 8},
                "finished": False,
            },
            {"tool": "compare_evidence", "arguments": {"record": {}}, "finished": False},
            {"finished": True},
        ]
    )
    record = run_llm_investigation(str(image), llm_client=client, max_steps=MAX_STEPS)

    assert record.verdict == Verdict.INCONCLUSIVE
    assert any("score-box conflict" in e for e in record.errors)
    assert record.verdict_reasons


def test_report_contains_only_facts_in_state(image, monkeypatch):
    _valid_tool_targets(monkeypatch)
    monkeypatch.setitem(
        tool_registry["run_localizer"],
        "callable",
        lambda path, threshold, detector=None, timeout=None: {
            "tool": "run_localizer",
            "status": "ok",
            "result": {"score": 0.2, "tampered_prob": 0.2, "bbox": [0.1, 0.1, 0.9, 0.9], "label": "fake"},
        },
    )
    monkeypatch.setitem(
        tool_registry["validate_prediction"],
        "callable",
        lambda run_result, image_width, image_height, timeout=None: {
            "tool": "validate_prediction",
            "status": "ok",
            "result": {"valid": True, "issues": []},
        },
    )
    client = ScriptedLLMClient(
        [
            {"tool": "run_localizer", "arguments": {"path": str(image), "threshold": 0.5}, "finished": False},
            {"tool": "validate_prediction", "arguments": {"run_result": {"score": 0.2}, "image_width": 8, "image_height": 8}, "finished": False},
            {"finished": True},
        ]
    )
    record = run_llm_investigation(str(image), llm_client=client, max_steps=MAX_STEPS)

    text = render_report_from_observations(
        record.observations, record.verdict.value, record.verdict_reasons, record.goal, record.filename
    )
    assert "0.2" in text
    assert "sample.png" in text
    assert "Not flagged by this detector" in text
    # Unstored numbers must not appear.
    assert "0.999" not in text
    assert "0.888" not in text
    # Conflict facts must not appear.
    assert "inconclusive" not in text.lower() or "Verdict:" in text


@pytest.fixture
def image(tmp_path):
    return _image_file(tmp_path)
