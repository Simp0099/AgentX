"""agent/llm_client.py: the single LLM SDK behind the agent's tool-calling loop.

LEGACY as of Phase 6: this single-shot ``choose_action`` interface is no
longer the primary path. The model-driven conversation now lives in
``agent.loop`` (``run_agent``), which sends the full message history, runs
several tool calls per turn and falls back across the configured models. This
module is retained because the deterministic pipeline and its tests still
depend on it, and nothing here may be deleted while ``agent/orchestrator.py``
remains the labelled fallback.

Only ONE LLM SDK is used: this module owns the provider HTTP call and exposes
a single interface, ``LLMClient.choose_action``. Nothing lower down may reach for
the provider directly, so the agent can be pointed at one model and one API.

Configuration lives in ``configs/agent.yaml`` (read through
:mod:`agent.agent_config`). No provider URL or model name is hardcoded here;
``DEEPFAKE_LLM_MODEL`` and ``DEEPFAKE_LLM_BASE_URL`` exist only as optional
per-process overrides. The credential is read from the environment variable
whose NAME ``api_key_env`` gives.

The provider is expected to speak the OpenAI-compatible ``chat/completions``
contract (tool calls, tool definition blocking). No third-party SDK is required;
the call is made with the standard library so the agent depends on nothing extra
for inference while remaining swappable with any compatible endpoint.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# Environment names (single source for where the key is looked up).
ENV_API_KEY = "DEEPFAKE_LLM_API_KEY"
ENV_MODEL = "DEEPFAKE_LLM_MODEL"
ENV_BASE_URL = "DEEPFAKE_LLM_BASE_URL"

# The chat/completions path, appended to the configured base URL.
_CHAT_PATH = "/chat/completions"


class LLMNotConfigured(Exception):
    """Raised when no API key is available for the LLM provider."""


class LLMResponseError(Exception):
    """Raised when the provider returns a non-success result."""


@dataclass
class Action:
    """A single structured decision returned by the LLM."""

    tool: str               # approved tool name, or ""
    arguments: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    finished: bool = False  # True means "stop the loop and build the report"


@dataclass
class LLMClientConfig:
    api_key: str
    model: str
    base_url: str
    timeout_seconds: float


class LLMClient:
    """Single LLM SDK behind the ``choose_action`` interface.

    ``choose_action`` is the ONLY place an LLM model is queried. It receives
    (1) the goal, (2) the recorded investigation state as a plain list of
    observations (not raw images), and (3) the JSON schemas of the approved
    tools, and returns a single structured action.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        from agent.agent_config import get_agent_config

        config = get_agent_config()
        self.api_key_env = str(config["api_key_env"])
        self.api_key = api_key
        if self.api_key is None:
            self.api_key = os.environ.get(ENV_API_KEY) or os.environ.get(self.api_key_env)
        self.model = model if model is not None else (os.environ.get(ENV_MODEL) or config["model"])
        self.base_url = base_url
        if self.base_url is None:
            self.base_url = os.environ.get(ENV_BASE_URL) or config["base_url"]
        self.timeout_seconds = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else float(config["request_timeout_seconds"])
        )
        if not self.api_key:
            raise LLMNotConfigured(
                f"LLM provider key not found in environment variable {self.api_key_env!r}; "
                "the agent will use the deterministic fixed pipeline."
            )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _post(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if not self.is_configured:
            raise LLMNotConfigured(
                f"LLM provider key not found in environment variable "
                f"{getattr(self, 'api_key_env', ENV_API_KEY)!r}; "
                "the agent will use the deterministic fixed pipeline."
            )
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url.rstrip("/") + _CHAT_PATH,
            data=body,
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
                "accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=timeout or self.timeout_seconds
        ) as response:
            raw = json.loads(response.read().decode("utf-8"))
        if raw.get("error") is not None or raw.get(
            "choices", [{}]
        )[0].get("finish_reason") == "length":
            raise LLMResponseError(raw.get("error", {}).get("message", "LLM call failed"))
        return raw

    def choose_action(
        self,
        goal: str,
        observations: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        timeout_seconds: float | None = None,
    ) -> Action:
        """Ask the LLM for the next action and return it as an ``Action``.

        Args:
            goal: the user's investigation goal (constant for the whole run).
            observations: the recorded state so far (a list of dicts); raw images
                are never sent to the model, only these structured facts.
            tool_schemas: the approved tool definitions (JSON Schema objects), so
                the model can call one of the registered tools.
            timeout_seconds: per-call timeout; defaults to the client level.

        Returns:
            An ``Action`` describing the tool to call (or ``finish=True``).
            Unknown / unparseable provider output is turned into a safe no-op
            action (tool="" / finished=False) rather than crashing the loop.
        """
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a bounded forensic investigator. "
                            "Decide ONE tool to run next, or say 'finish' when you have enough. "
                            "Return ONLY a tool call: name and arguments matching the provided "
                            "JSON schemas. Do not invent results, do not call unapproved tools, "
                            "and do not output anything besides the tool call."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "goal": goal,
                                "observations": observations,
                            },
                            default=str,
                        ),
                    },
                ],
                "tools": tool_schemas,
                "tool_choice": "auto",
                "max_tokens": 512,
            }
            response = self._post(payload, timeout=timeout_seconds)
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls") or []
            finish_reason = choice.get("finish_reason")

            if finish_reason == "stop" and not tool_calls:
                return Action(
                    tool="",
                    arguments={},
                    rationale="LLM indicated finishing the investigation.",
                    finished=True,
                )

            if tool_calls:
                block = tool_calls[0]
                name = block.get("function", {}).get("name") or ""
                try:
                    args = json.loads(block.get("function", {}).get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                rationale = message.get("content") or ""
                return Action(
                    tool=name,
                    arguments=args if isinstance(args, dict) else {},
                    rationale=rationale,
                    finished=False,
                )

            # No tool call and no stop: treat as "finish" so the loop can produce
            # a report from the recorded state (the LLM must not invent results).
            return Action(
                tool="",
                arguments={},
                rationale="LLM returned no tool call and no explicit stop.",
                finished=True,
            )

        except LLMNotConfigured:
            # Never present fallback output as agentic; the caller chooses the
            # deterministic path and labels it accordingly.
            raise
        except Exception as exc:  # network, timeouts, parse errors, provider errors
            return Action(
                tool="",
                arguments={},
                rationale=f"LLM call failed ({type(exc).__name__}); fallback active.",
                finished=True,
            )


class ScriptedLLMClient(LLMClient):
    """A deterministic "fake LLM" used by tests. No network, no keys.

    ``responses`` is a list of Action objects (or dicts). Each call to
    ``choose_action`` consumes the next one. When the list is exhausted the
    scripted client returns a default "finish" action so tests can drive the
    loop to its natural stop point.
    """

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        super().__init__(api_key="SCRIPTED", model="scripted-test", base_url="local")
        self._responses = list(responses)
        self._index = 0

    def choose_action(
        self,
        goal: str,
        observations: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        timeout_seconds: float | None = None,
    ) -> Action:
        if self._responses:
            raw = self._responses.pop(0)
            if isinstance(raw, dict):
                return Action(
                    tool=str(raw.get("tool", "")),
                    arguments=raw.get("arguments", {}) or {},
                    rationale=str(raw.get("rationale", "")),
                    finished=bool(raw.get("finished", False)),
                )
            return raw
        # Exhausted: finish deterministically so tests can assert the loop bound.
        return Action(tool="", arguments={}, rationale="scripted client exhausted", finished=True)


def get_llm_client() -> LLMClient | None:
    """Return a configured LLMClient, or None when no API key is set.

    The orchestrator consults this before running the deterministic fixed
    pipeline, and labels its output 'fixed pipeline (no LLM)'.
    """
    try:
        return LLMClient()
    except LLMNotConfigured:
        return None
