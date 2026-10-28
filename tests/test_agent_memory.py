"""Offline tests for short and long term investigation memory."""

from __future__ import annotations

import json
import sqlite3

from agent import loop
from agent.memory import trim_messages
from agent.memory_store import (
    clear_memory,
    image_hash,
    load_investigation,
    save_investigation,
)
from tests.fakes import FakeClient, make_message, make_tool_call, tool_messages

_get_config = loop.get_agent_config


def _tool_exchange(index: int) -> list[dict]:
    call_id = f"call_{index}"
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {}}],
        },
        {"role": "tool", "tool_call_id": call_id, "content": "{}"},
    ]


def test_trim_messages_preserves_system_and_complete_tool_units():
    messages = [{"role": "system", "content": "system stays"}]
    for index in range(30):
        messages.extend(
            [
                {"role": "user", "content": f"question {index}"},
                *_tool_exchange(index),
                {"role": "assistant", "content": f"answer {index}"},
            ]
        )

    trimmed = trim_messages(messages, max_messages=20, max_chars=4000)

    assert trimmed[0] == messages[0]
    assert len(trimmed) <= 20
    assert sum(len(json.dumps(item)) for item in trimmed) <= 4000
    by_id = {call["id"] for item in trimmed if item.get("tool_calls")
             for call in item["tool_calls"]}
    replies = {item["tool_call_id"] for item in trimmed if item["role"] == "tool"}
    assert by_id == replies


def test_trim_discards_orphan_tool_messages():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old"},
        *_tool_exchange(1),
        {"role": "user", "content": "new"},
        {"role": "tool", "tool_call_id": "missing", "content": "orphan"},
    ]
    trimmed = trim_messages(messages, 20, 1000)
    assert not any(item.get("tool_call_id") == "missing" for item in trimmed)


def _memory_config(path):
    config = _get_config().copy()
    config["memory"] = {"enabled": True, "path": str(path)}
    config["history"] = {"max_messages": 20, "max_chars": 12000}
    return config


def test_followup_reuses_history_without_another_tool_call(tmp_path, monkeypatch):
    image = tmp_path / "same.png"
    image.write_bytes(b"image bytes for hashing")
    monkeypatch.setenv(loop.ENV_UPLOAD_ROOT, str(tmp_path))
    monkeypatch.setattr(loop, "get_agent_config", lambda: _memory_config(tmp_path / "m.db"))
    first_client = FakeClient(
        [
            make_message(content="", tool_calls=[make_tool_call(
                "inspect_image", {"image_path": str(image)}
            )]),
            make_message(content="The returned score was 0.73."),
        ]
    )
    _answer, history = loop.run_agent("quick_check", str(image), client=first_client)
    assert any("0.73" in str(item.get("content", "")) for item in history), history
    second_client = FakeClient([make_message(content="The detector score was 0.73.")])

    answer, _ = loop.run_agent(
        "what was the detector score?", str(image), history=history, client=second_client,
        use_memory=False,
    )

    assert answer.startswith("The detector score was 0.73.")
    assert len(second_client.requests) == 1
    request_messages = second_client.requests[0]["messages"]
    assert any(
        "0.73" in str(item.get("content", ""))
        for item in request_messages
    ), request_messages
    previous_tools = len(tool_messages({"messages": history}))
    assert len(tool_messages(second_client.requests[0])) == previous_tools


def test_same_image_uses_saved_record_without_tools(tmp_path, monkeypatch):
    image = tmp_path / "same.png"
    image.write_bytes(b"private image bytes")
    db = tmp_path / "memory.sqlite"
    monkeypatch.setenv(loop.ENV_UPLOAD_ROOT, str(tmp_path))
    monkeypatch.setattr(loop, "get_agent_config", lambda: _memory_config(db))
    first = FakeClient(
        [
            make_message(tool_calls=[make_tool_call(
                "inspect_image", {"image_path": str(image)}
            )]),
            make_message(content="Checked image."),
        ]
    )
    initial = loop.run_agent_full("quick_check", str(image), client=first)
    second = FakeClient([])

    cached = loop.run_agent_full("quick_check", str(image), client=second)

    assert cached.steps == 0
    assert cached.answer == initial.answer
    assert cached.record.verdict == cached.grounded_verdict == initial.grounded_verdict
    assert cached.record.observations[0].result == initial.record.observations[0].result
    assert not second.requests
    with sqlite3.connect(db) as connection:
        serialized = repr(connection.execute("SELECT * FROM investigations").fetchone())
    assert b"private image bytes" not in serialized.encode()
    assert image_hash(image)
    clear_memory(db)


def test_cache_invalidates_when_model_threshold_or_config_changes(tmp_path):
    image = tmp_path / "input.bin"
    image.write_bytes(b"content")
    db = tmp_path / "cache.db"
    config_hash = "config-a"
    model = "model-a"
    threshold = 0.5
    key = image_hash(image)
    record = loop._new_record("quick_check", str(image))
    save_investigation(db, key, record, "saved", model, threshold, config_hash)

    assert load_investigation(db, key, model, threshold, config_hash) is not None
    assert load_investigation(db, key, "model-b", threshold, config_hash) is None
    assert load_investigation(db, key, model, 0.6, config_hash) is None
    assert load_investigation(db, key, model, threshold, "config-b") is None
