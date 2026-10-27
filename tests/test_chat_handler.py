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
    monkeypatch.setattr(
        chat_handler,
        "_ensure_ela_with_path",
        lambda _path: (Image.new("RGB", (16, 16), "gray"), None),
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

    result = chat_handler.agent_chat("Is this edited?", image, {})
    state = result["_state"]

    assert result["answer_text"].startswith("The inspection found an image.")
    assert result["mode"] == "agent"
    assert result["image"] == {"filename": "sample.png", "width": 16, "height": 16}
    calls = result["tool_log"]
    assert calls[0]["tool"] == "inspect_image"
    assert calls[0]["status"] == "ok"
    assert '"valid": true' in calls[0]["result_preview"]
    assert result["ela_image_path"] is None
    assert state["image_hash"] and Path(state["image_path"]).is_file()
    assert len(clients) == 1
    assert len(list(uploads.glob("chat-turn-*.png"))) == 0


def test_manipulated_image_returns_pixel_region_and_coverage(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    record = _record("image.png")
    monkeypatch.setattr(
        chat_handler,
        "run_agent_full",
        lambda **_kwargs: AgentResult(
            answer="The evidence flags a manipulated region.",
            messages=[{"role": "assistant", "content": "The evidence flags a manipulated region."}],
            record=record,
            steps=1,
            stopped_reason="final_answer",
        ),
    )

    reply = chat_handler.agent_chat("Check", Image.new("RGB", (16, 16)), {})

    assert reply["region"] == {"horizontal": [2, 13], "vertical": [2, 13]}
    assert reply["coverage_percent"] == 47.3
    assert reply["tampered_score_uncalibrated"] == 0.8


def test_authentic_image_has_no_region(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    record = _record("image.png")
    record.verdict = Verdict.NOT_FLAGGED_BY_THIS_DETECTOR
    record.observations[1].result = {"tampered_prob": 0.1, "bbox": None}
    monkeypatch.setattr(
        chat_handler,
        "run_agent_full",
        lambda **_kwargs: AgentResult(
            answer="No region was flagged.",
            messages=[{"role": "assistant", "content": "No region was flagged."}],
            record=record,
            steps=1,
            stopped_reason="final_answer",
        ),
    )

    reply = chat_handler.agent_chat("Check", Image.new("RGB", (16, 16)), {})

    assert reply["region"] is None
    assert reply["coverage_percent"] is None


def test_followup_reuses_history_and_needs_no_new_image(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    original_run = loop.run_agent_full
    seen_histories = []
    seen_paths = []

    def run_with_fake(goal, image_path, history=None, **kwargs):
        seen_histories.append(list(history or []))
        seen_paths.append(str(image_path))
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
    first = chat_handler.agent_chat("Check this image", image, {})
    state = first["_state"]
    followup = chat_handler.agent_chat("What did you find?", None, state)
    next_state = followup["_state"]

    assert "The earlier inspection" in followup["answer_text"]
    calls = followup["tool_log"]
    assert seen_histories[0] == []
    assert any(item["role"] == "tool" for item in seen_histories[1])
    assert len(calls) == 1
    assert calls[0]["tool"] == "inspect_image"
    assert len(seen_paths) == 2
    assert Path(state["image_path"]).is_file()
    assert next_state["image_path"] == state["image_path"]
    assert next_state["image_hash"] == state["image_hash"]


def test_no_model_uses_clearly_labeled_fixed_pipeline(tmp_path, monkeypatch, caplog):
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
            errors=["RuntimeError: environment variable OPENAI_API_KEY is not set"],
        ),
    )
    monkeypatch.setattr(
        chat_handler,
        "run_investigation",
        lambda path, **_kwargs: _record(path),
    )

    image = Image.new("RGB", (16, 16), "green")
    image.filename = "<photo>.png"
    result = chat_handler.agent_chat("Is this edited?", image, {})
    state = result["_state"]

    assert result["mode"] == "fixed_pipeline"
    assert "Flagged for review" in result["answer_text"]
    assert "(no LLM)" not in result["answer_text"]
    assert "fixed pipeline (no LLM)" not in result["answer_text"]
    assert "Language model unavailable (missing provider API key)" in result["answer_text"]
    assert "model chat unavailable: missing provider API key" in caplog.text
    calls = result["tool_log"]
    assert calls[0]["tool"] == "inspect_image"
    assert calls[0]["source"] == "fixed pipeline"
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

    result = chat_handler.agent_chat(
        "Is this edited?", Image.new("RGB", (16, 16), "gold"), {}
    )

    assert result["answer_text"] == "The saved inspection found a review signal."
    calls = result["tool_log"]
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
    result = chat_handler.agent_chat(
        "Explain", Image.new("RGB", (16, 16), "red"), {}
    )

    assert result["answer_text"] == "<script>alert(1)</script>"
    calls = result["tool_log"]
    assert "&lt;unsafe&gt;" in calls[0]["arguments"]
    assert "&lt;script&gt;" in calls[0]["result_preview"]


def test_chat_handler_can_create_ela_with_the_registered_tool(tmp_path, monkeypatch):
    make_ela = chat_handler._ensure_ela
    make_ela_with_path = chat_handler._ensure_ela_with_path
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
    monkeypatch.setattr(chat_handler, "_ensure_ela_with_path", make_ela_with_path)
    ela = chat_handler._ensure_ela(path)

    assert ela is not None
    assert ela.size == (20, 18)
