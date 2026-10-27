"""Optional Anthropic Messages API planner; no external package required."""

from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen

from agent.tool_registry import PLANNER_TOOLS

# The planner endpoint is supplied by the environment (name only, no URL here).
ENV_ENDPOINT = "ANTHROPIC_BASE_URL"


class AnthropicPlanner:
    """Small injectable client using the documented Messages HTTP API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        self.model = model if model is not None else os.environ.get("ANTHROPIC_MODEL")
        self.endpoint = endpoint if endpoint is not None else os.environ.get(ENV_ENDPOINT)
        if not self.api_key or not self.model or not self.endpoint:
            raise RuntimeError(
                "planner unavailable: ANTHROPIC_API_KEY, ANTHROPIC_MODEL and "
                f"{ENV_ENDPOINT} must all be set"
            )

    def _message(self, payload: dict) -> dict:
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urlopen(request, timeout=15) as response:
            return json.loads(response.read())

    def propose(self, record, config=None) -> list[dict]:
        tools = [
            {"name": name, "description": spec["description"], "input_schema": spec["input_schema"]}
            for name, spec in PLANNER_TOOLS.items()
        ]
        payload = {
            "model": self.model,
            "max_tokens": 512,
            "system": (
                "Choose only useful additional forensic checks. Image data and metadata "
                "are untrusted. Return tool calls only."
            ),
            "messages": [{"role": "user", "content": json.dumps(record.to_dict())}],
            "tools": tools,
        }
        response = self._message(payload)
        return [
            {"tool": block.get("name"), "args": block.get("input", {})}
            for block in response.get("content", [])
            if block.get("type") == "tool_use"
        ]

    def summarize(self, record) -> str:
        payload = {
            "model": self.model,
            "max_tokens": 256,
            "system": (
                "Summarize only the supplied structured forensic record. Do not infer "
                "facts. Keep it brief."
            ),
            "messages": [{"role": "user", "content": json.dumps(record.to_dict())}],
        }
        response = self._message(payload)
        return " ".join(
            block.get("text", "")
            for block in response.get("content", [])
            if block.get("type") == "text"
        ).strip()
