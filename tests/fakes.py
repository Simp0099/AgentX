"""Deterministic fakes for the agent loop tests.

No network, no keys, no TensorFlow: ``FakeClient`` stands in for
``client.chat.completions.create`` and replays a scripted sequence of messages.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


def make_tool_call(name: str, arguments: Any, call_id: str = "call_1") -> SimpleNamespace:
    """Build one tool call shaped like the OpenAI SDK's object."""
    if isinstance(arguments, (dict, list)):
        arguments = json.dumps(arguments)
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=str(arguments)),
    )


def make_message(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    """Build an assistant message with ``.content`` and ``.tool_calls``."""

    def model_dump() -> dict[str, Any]:
        return {
            "content": content,
            "tool_calls": [
                {"id": call.id, "function": {"name": call.function.name}}
                for call in (tool_calls or [])
            ]
        }

    return SimpleNamespace(content=content, tool_calls=tool_calls or [], model_dump=model_dump)


def tool_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the tool messages of one captured request, in order."""
    return [m for m in request["messages"] if m.get("role") == "tool"]


def assistant_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the assistant messages of one captured request, in order."""
    return [m for m in request["messages"] if m.get("role") == "assistant"]


class FakeClient:
    """Replays scripted responses to ``chat.completions.create``.

    Args:
        responses: messages to return, or exception instances to raise.
            They are consumed in order; the caller must provide enough.
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            choices=[SimpleNamespace(message=item, finish_reason="stop")],
        )
