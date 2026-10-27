"""Short-term conversation history utilities."""

from __future__ import annotations

import json
from typing import Any


def trim_messages(
    messages: list[dict[str, Any]], max_messages: int, max_chars: int
) -> list[dict[str, Any]]:
    """Keep the system prompt and newest complete exchanges within limits.

    Assistant tool-call messages and all of their replies are treated as one
    unit, so trimming cannot leave a provider-invalid orphan message.
    """
    if not messages:
        return []
    system = messages[0] if messages[0].get("role") == "system" else None
    body = messages[1:] if system else messages[:]
    units: list[list[dict[str, Any]]] = []
    pending: list[dict[str, Any]] = []
    expected: set[str] = set()
    for message in body:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            if pending:
                units.append(pending)
            pending = [message]
            expected = {
                str(call.get("id", "")) for call in message.get("tool_calls", [])
            }
        elif message.get("role") == "tool" and pending:
            call_id = str(message.get("tool_call_id", ""))
            if call_id not in expected:
                continue
            pending.append(message)
            expected.discard(call_id)
            if not expected:
                units.append(pending)
                pending = []
        elif message.get("role") == "tool":
            continue
        else:
            if pending:
                # An incomplete exchange is not safe to send to a provider.
                pending = []
                expected = set()
            units.append([message])
    # Incomplete trailing tool exchanges are omitted.
    max_messages = max(1, max_messages)
    max_chars = max(1, max_chars)
    kept: list[list[dict[str, Any]]] = []
    current_count = 1 if system else 0
    current_chars = len(json.dumps(system, default=str)) if system else 0
    for unit in reversed(units):
        unit_chars = len(json.dumps(unit, default=str))
        if current_count + len(unit) > max_messages or current_chars + unit_chars > max_chars:
            break
        kept.append(unit)
        current_count += len(unit)
        current_chars += unit_chars
    result = [system] if system else []
    for unit in reversed(kept):
        result.extend(unit)
    return result
