"""Offline tests for the Gradio-independent image chat handler."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from agent import loop
from agent.agent_config import get_agent_config
from agent.loop import AgentResult
from agent.schemas import InvestigationRecord, Observation, Verdict
from app import chat_handler
from tests.fakes import FakeClient, make_message, make_tool_call


def _config(path: Path) -> dict:
    config = get_agent_config().copy()
    config["memory"] = {"enabled": True, "path": str(path)}
    config["history"] = {"max_messages": 20, "max_chars": 12000}
    return config


def _record(path: str) -> InvestigationRecord:
    return InvestigationRecord(
        investigation_id="chat-fallback",
        image_id="hash",
        filename="<photo>.png",
        goal="quick_check",
        goal_text="",
        threshold=0.5,
        config_snapshot_hash="",
        model_version="stub",
        observations=[
            Observation(
                "inspect_image",
                {"path": path},
                "ok",
                {"valid": True, "width": 16, "height": 16},
            ),
            Observation(
                "run_localizer",
                {"path": path},
                "ok",
                {"tampered_prob": 0.8, "bbox": [0.1, 0.1, 0.8, 0.8]},
            ),
        ],
        errors=[],
        rationale="recorded check",
        verdict=Verdict.FLAGGED_FOR_REVIEW,
    )


def _setup(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(chat_handler, "UPLOAD_DIR", uploads)
    monkeypatch.setenv(loop.ENV_UPLOAD_ROOT, str(uploads))
    monkeypatch.setattr(loop, "get_agent_config", lambda: _config(tmp_path / "memory.db"))
    monkeypatch.setattr(
        chat_handler,
        "_ensure_ela",
        lambda _path: Image.new("RGB", (16, 16), "gray"),
    )
    return uploads


def test_handler_uses_fake_client_and_returns_tool_log_and_ela(tmp_path, monkeypatch):
    uploads = _setup(tmp_path, monkeypatch)
    original_run = loop.run_agent_full
    clients = []

    def run_with_fake(goal, image_path, history=None, client=None, **kwargs):
        assert Path(image_path).is_file()
        assert Path(image_path).parent == uploads.resolve()
        fake = FakeClient(
            [
                make_message(
                    content="I will inspect this image first.",
                    tool_calls=[
                        make_tool_call("inspect_image", {"image_path": image_path})
                    ],
                ),
                make_message(content="The inspection found an image.")
                if not history
                else make_message(content="I can continue from the prior inspection."),
            ]
        )
        clients.append(fake)
        return original_run(
            goal,
            image_path,
            history=history,
            client=fake,
            use_memory=False,
        )

    monkeypatch.setattr(chat_handler, "run_agent_full", run_with_fake)
    image = Image.fromarray(np.full((16, 16, 3), 120, dtype=np.uint8))
    image.filename = "sample.png"

    answer, calls, ela, state = chat_handler.agent_chat("Is this edited?", image, {})

    assert "The inspection found an image." in answer
    assert calls[0]["tool"] == "inspect_image"
    assert calls[0]["status"] == "ok"
    assert '"valid": true' in calls[0]["result_preview"]
    assert ela is not None and ela.size == image.size
    assert state["image_hash"] and Path(state["image_path"]).is_file()
    assert len(clients) == 1
    assert len(list(uploads.glob("chat-turn-*.png"))) == 0


def test_followup_reuses_history_and_needs_no_new_image(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    original_run = loop.run_agent_full
    seen_histories = []

    def run_with_fake(goal, image_path, history=None, **kwargs):
        seen_histories.append(list(history or []))
        if not history:
            responses = [
                make_message(
                    content="First I will inspect it.",
                    tool_calls=[
                        make_tool_call("inspect_image", {"image_path": image_path})
                    ],
                ),
                make_message(content="The initial image is readable."),
            ]
        else:
            responses = [make_message(content="The earlier inspection found a readable image.")]
        return original_run(
            goal,
            image_path,
            history=history,
            client=FakeClient(responses),
            use_memory=False,
        )

    monkeypatch.setattr(chat_handler, "run_agent_full", run_with_fake)
    image = Image.new("RGB", (16, 16), "blue")
    _, _calls, _ela, state = chat_handler.agent_chat("Check this image", image, {})
    followup, calls, _ela, next_state = chat_handler.agent_chat(
        "What did you find?", None, state
    )

    assert "The earlier inspection" in followup
    assert seen_histories[0] == []
    assert any(item["role"] == "tool" for item in seen_histories[1])
    assert len(calls) == 1
    assert calls[0]["tool"] == "inspect_image"
    assert next_state["image_path"] == state["image_path"]
    assert next_state["image_hash"] == state["image_hash"]


def test_no_model_uses_clearly_labeled_fixed_pipeline(tmp_path, monkeypatch):
    uploads = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        chat_handler,
        "run_agent_full",
        lambda **_kwargs: AgentResult(
            answer="Model unavailable",
            messages=[],
            record=_record("temporary.png"),
            steps=0,
            stopped_reason="model_unavailable",
        ),
    )
    monkeypatch.setattr(
        chat_handler,
        "run_investigation",
        lambda path, **_kwargs: _record(path),
    )

    image = Image.new("RGB", (16, 16), "green")
    image.filename = "<photo>.png"
    answer, calls, _ela, state = chat_handler.agent_chat("Is this edited?", image, {})

    assert "fixed pipeline (no LLM)" in answer
    assert "Flagged for review" in answer
    assert calls[0]["tool"] == "inspect_image"
    assert calls[0]["source"] == "fixed pipeline (no LLM)"
    assert "### &lt;photo&gt;.png" in answer
    assert Path(state["image_path"]).parent == uploads.resolve()


def test_memory_hit_is_marked_as_reused_saved_result(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    record = _record("temporary.png")
    monkeypatch.setattr(
        chat_handler,
        "run_agent_full",
        lambda **_kwargs: AgentResult(
            answer="The saved inspection found a review signal.",
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Is this edited?"},
                {"role": "assistant", "content": "The saved inspection found a review signal."},
            ],
            record=record,
            steps=0,
            stopped_reason="final_answer",
        ),
    )

    answer, calls, _ela, _state = chat_handler.agent_chat(
        "Is this edited?", Image.new("RGB", (16, 16), "gold"), {}
    )

    assert "Reused saved result" in answer
    assert calls[0]["tool"] == "inspect_image"


def test_untrusted_model_and_tool_text_is_escaped(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)

    def fake_run(**_kwargs):
        return type(
            "FakeResult",
            (),
            {
                "stopped_reason": "final_answer",
                "steps": 1,
                "answer": "<script>alert(1)</script>",
                "messages": [
                    {"role": "assistant", "content": "<script>alert(1)</script>"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "x",
                                "function": {
                                    "name": "inspect_image",
                                    "arguments": json.dumps({"image_path": "<unsafe>"}),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "x",
                        "content": json.dumps({"note": "<script>"}),
                    },
                ],
                "record": _record("safe.png"),
            },
        )()

    monkeypatch.setattr(chat_handler, "run_agent_full", fake_run)
    answer, calls, _ela, _state = chat_handler.agent_chat(
        "Explain", Image.new("RGB", (16, 16), "red"), {}
    )

    assert "&lt;script&gt;" in answer
    assert "&lt;unsafe&gt;" in calls[0]["arguments"]
    assert "&lt;script&gt;" in calls[0]["result_preview"]


def test_chat_handler_can_create_ela_with_the_registered_tool(tmp_path, monkeypatch):
    make_ela = chat_handler._ensure_ela
    uploads = _setup(tmp_path, monkeypatch)
    from agent.tools import generate_ela

    monkeypatch.setattr(
        generate_ela,
        "get_config",
        lambda: {
            "data": {"ela_quality": 90},
            "forensics": {"ela_temp_dir": str(tmp_path / "ela")},
        },
    )
    path = uploads / "ela-source.png"
    Image.new("RGB", (20, 18), "purple").save(path)

    monkeypatch.setattr(chat_handler, "_ensure_ela", make_ela)
    ela = chat_handler._ensure_ela(path)

    assert ela is not None
    assert ela.size == (20, 18)
