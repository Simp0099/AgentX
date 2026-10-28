"""agent/loop.py: the model-driven tool loop.

``run_agent`` replaces the single-shot ``choose_action`` design for the primary
path. One conversation, repeatedly::

    user -> assistant(tool_calls) -> tool results -> ... -> final text

Each pass sends the FULL message history, so the model always sees what it asked
for and what came back. Tool calls are executed through
:func:`agent.agent_tools.run_tool`, which cannot raise: a bad path, a missing
checkpoint or malformed arguments come back as an ``{"error", "hint"}`` payload
that is fed straight back to the model as a ``tool`` message.

Three outcomes exist: a final answer, a tool call, and a malformed tool call
(invalid JSON arguments), which is answered with an error result for that
``call_id`` so the conversation continues instead of crashing.

Provider failures are never laundered into an answer. A 429, timeout, 400 or 5xx
is retried twice with backoff, then the next model in ``fallback_models`` is
tried, and only then does the loop return a structured failure whose text starts
with ``Model unavailable:``.

``run_investigation`` in ``agent/orchestrator.py`` remains the deterministic,
labelled fallback and is untouched by this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.agent_config import get_agent_config, model_entries
from agent.agent_tools import TOOL_SCHEMAS, ToolContext, run_tool
from agent.env import load_env
from agent.memory import trim_messages
from agent.memory_store import image_hash, load_investigation, save_investigation
from agent.prompts import SYSTEM_PROMPT, build_user_message
from agent.schemas import InvestigationRecord, Observation, Verdict
from agent.verdict import decide

# Exact wording required when the step budget runs out.
STEP_LIMIT_MESSAGE = (
    "Stopped: the step limit was reached before a final answer. "
    "Partial results are in the tool messages above."
)

# Seconds to wait before each retry. Tests set this to zero.
BACKOFF_SECONDS = (1.0, 2.0)

# Attempts against one model before moving to the next one.
MAX_PROVIDER_RETRIES = 2
_MODEL_COOLDOWNS: dict[str, float] = {}
LAST_USED_MODELS: dict[str, str] = {}
_LOGGER = logging.getLogger(__name__)

ENV_UPLOAD_ROOT = "AGENT_UPLOAD_ROOT"
DEFAULT_UPLOAD_ROOT = "artifacts/uploads"

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "logs" / "agent.log"


@dataclass(frozen=True)
class ProviderFailure:
    category: str
    cause: str
    retry_after: float | None = None


def _provider_failure(exc: Exception) -> ProviderFailure:
    """Classify an SDK/provider exception using status, headers and safe text."""
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    headers = getattr(exc, "headers", None) or getattr(response, "headers", None) or {}
    try:
        retry_after_value = headers.get("Retry-After") or headers.get("retry-after")
    except AttributeError:
        retry_after_value = None
    try:
        retry_after = max(0.0, float(retry_after_value)) if retry_after_value is not None else None
    except (TypeError, ValueError):
        try:
            retry_after = (
                max(
                    0.0,
                    (
                        parsedate_to_datetime(str(retry_after_value)) - datetime.now(timezone.utc)
                    ).total_seconds(),
                )
                if retry_after_value is not None
                else None
            )
        except (TypeError, ValueError, OverflowError):
            retry_after = None
    text = str(exc)
    lowered = text.casefold()
    if status is None:
        match = re.search(r"\b([45]\d\d)\b", text)
        status = int(match.group(1)) if match else None
    cause = f"{type(exc).__name__}: {text[:300]}"
    if (
        status == 429
        or (retry_after is not None and status is None)
        or any(
            term in lowered
            for term in (
                "quota",
                "rate limit",
                "rate-limit",
                "too many requests",
                "token exhausted",
                "token exhaustion",
            )
        )
    ):
        return ProviderFailure("quota", cause, retry_after)
    if status in (400, 401) or "invalid request" in lowered or "authentication" in lowered:
        return ProviderFailure("fatal", cause)
    if status is not None and 500 <= int(status) <= 599:
        return ProviderFailure("transient", cause, retry_after)
    if isinstance(exc, (TimeoutError, ConnectionError)) or any(
        term in lowered
        for term in ("timeout", "timed out", "connection", "temporarily unavailable")
    ):
        return ProviderFailure("transient", cause)
    if status is not None and 400 <= int(status) <= 499:
        return ProviderFailure("fatal", cause)
    return ProviderFailure("fatal", cause)


def _log_switch(model: str, next_model: str | None, reason: str) -> None:
    """Log a provider route change without including request credentials."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    destination = next_model or "none"
    line = f"{stamp} model_switch from={model} to={destination} reason={reason[:300]}"
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass
    _LOGGER.warning("%s", line)


@dataclass
class AgentResult:
    """Everything one run produced.

    ``answer`` is the model's own text and is never rewritten here;
    ``grounded_verdict`` is the independent ``decide(record)`` verdict, kept
    separate so a later step can compare the two.
    """

    answer: str
    messages: list[dict[str, Any]]
    record: InvestigationRecord
    steps: int
    stopped_reason: str  # final_answer | max_steps | model_unavailable | retry_limit
    grounded_verdict: Verdict | None = None
    errors: list[str] = field(default_factory=list)
    confidence: str = "low"


# ---------------------------------------------------------------------------
# Request plumbing
# ---------------------------------------------------------------------------


def _upload_root() -> Path:
    """Directory every image path must resolve inside."""
    return Path(os.environ.get(ENV_UPLOAD_ROOT, DEFAULT_UPLOAD_ROOT)).expanduser()


def _default_threshold() -> float:
    from agent.config import get_config

    return float(get_config().get("output", {}).get("threshold", 0.5))


def _build_client(endpoint: dict[str, Any]) -> Any:
    """Construct an OpenAI SDK client for one configured endpoint.

    The key is read from the environment variable NAMED in the config; its
    value never appears here or in any log.
    """
    env_name = str(endpoint["api_key_env"])
    load_env()
    key = os.environ.get(env_name, "").strip()
    if not key or key.casefold() in {"your_key_here", "replace_me", "changeme", "placeholder"}:
        return None
    config = get_agent_config()
    if str(endpoint["provider"]).lower() == "anthropic":
        return _AnthropicCompatClient(
            key,
            str(endpoint["base_url"]),
            float(config["request_timeout_seconds"]),
        )
    from openai import OpenAI

    return OpenAI(
        api_key=key,
        base_url=endpoint["base_url"],
        timeout=float(config["request_timeout_seconds"]),
        max_retries=0,  # retries are ours, so they are counted and bounded
    )


class _AnthropicCompatClient:
    """Translate the loop's OpenAI-shaped tool calls to Anthropic Messages."""

    def __init__(self, api_key: str, base_url: str, timeout: float):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = 0.0,
    ) -> Any:
        system = "\n\n".join(
            str(item.get("content") or "") for item in messages if item.get("role") == "system"
        )
        converted: list[dict[str, Any]] = []
        for item in messages:
            role = item.get("role")
            if role == "system":
                continue
            if role == "tool":
                result = {
                    "type": "tool_result",
                    "tool_use_id": item.get("tool_call_id", ""),
                    "content": str(item.get("content", "")),
                }
                if converted and converted[-1]["role"] == "user":
                    converted[-1]["content"].append(result)
                else:
                    converted.append({"role": "user", "content": [result]})
                continue
            blocks: list[dict[str, Any]] = []
            if role == "assistant" and item.get("tool_calls"):
                if item.get("content"):
                    blocks.append({"type": "text", "text": str(item["content"])})
                for call in item["tool_calls"]:
                    function = call.get("function", {})
                    try:
                        arguments = json.loads(function.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.get("id", ""),
                            "name": function.get("name", ""),
                            "input": arguments,
                        }
                    )
            elif isinstance(item.get("content"), str):
                blocks.append({"type": "text", "text": item["content"]})
            else:
                blocks = item.get("content") or []
            converted.append(
                {"role": "assistant" if role == "assistant" else "user", "content": blocks}
            )

        anthropic_tools = []
        for tool in tools:
            function = tool.get("function", {})
            anthropic_tools.append(
                {
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "input_schema": function.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": 2048,
            "messages": converted,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system
        if anthropic_tools:
            payload["tools"] = anthropic_tools
            payload["tool_choice"] = {"type": "auto" if tool_choice == "auto" else "any"}
        request = urllib.request.Request(
            f"{self.base_url}/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderHTTPError(exc.code) from None
        blocks = raw.get("content", [])
        text = "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        tool_calls = [
            SimpleNamespace(
                id=block.get("id", ""),
                type="function",
                function=SimpleNamespace(
                    name=block.get("name", ""),
                    arguments=json.dumps(block.get("input", {})),
                ),
            )
            for block in blocks
            if block.get("type") == "tool_use"
        ]
        message = SimpleNamespace(content=text, tool_calls=tool_calls or None)
        finish_reason = "length" if raw.get("stop_reason") == "max_tokens" else "stop"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
        )


class ProviderHTTPError(RuntimeError):
    """Status-only provider error that cannot disclose request credentials."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code} from Anthropic")
        self.status_code = status_code


def _initial_messages(
    goal: str, image_path: str, history: list | None, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """System prompt, prior exchanges, then this investigation's user turn."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(dict(item) for item in history if item.get("role") != "system")
    messages.append({"role": "user", "content": build_user_message(goal, image_path)})
    limits = config.get("history", {})
    return trim_messages(
        messages,
        int(limits.get("max_messages", 20)),
        int(limits.get("max_chars", 12000)),
    )


def _call_id(tool_call: Any, index: int) -> str:
    return str(getattr(tool_call, "id", "") or f"call_{index}")


def _tool_name(tool_call: Any) -> str:
    function = getattr(tool_call, "function", None)
    return str(getattr(function, "name", "") or "")


def _tool_arguments(tool_call: Any) -> str:
    function = getattr(tool_call, "function", None)
    return str(getattr(function, "arguments", "") or "")


def _assistant_message(message: Any) -> dict[str, Any]:
    """Serialise an assistant turn back into the wire format.

    The optional ``name`` field on ``tool`` messages is never sent at all, and
    the assistant turn carries only ``role``/``content``/``tool_calls``: that is
    the smallest shape every OpenAI-compatible endpoint accepts. See
    ``docs/model_selection.md``.
    """
    calls = list(getattr(message, "tool_calls", None) or [])
    content = getattr(message, "content", None)
    if not calls:
        return {"role": "assistant", "content": content or ""}
    body: dict[str, Any] = {
        "role": "assistant",
        "content": content if content else None,
        "tool_calls": [
            {
                "id": _call_id(call, index),
                "type": "function",
                "function": {
                    "name": _tool_name(call),
                    "arguments": _tool_arguments(call),
                },
            }
            for index, call in enumerate(calls)
        ],
    }
    return body


# ---------------------------------------------------------------------------
# Record, observations and logging
# ---------------------------------------------------------------------------


def _new_record(goal: str, image_path: str) -> InvestigationRecord:
    config = get_agent_config()
    snapshot = {"agent": config}
    return InvestigationRecord(
        investigation_id=str(uuid.uuid4()),
        image_id=str(uuid.uuid4()),
        filename=os.path.basename(image_path),
        goal=goal,
        goal_text="",
        threshold=_default_threshold(),
        config_snapshot_hash=hashlib.sha256(str(snapshot).encode()).hexdigest()[:16],
        model_version="unknown",
        observations=[],
        errors=[],
        rationale="",
        image_path=image_path,
        planner_status="available",
        config_snapshot=snapshot,
    )


def _observe(
    record: InvestigationRecord,
    name: str,
    args: dict[str, Any],
    payload: dict[str, Any],
    status: str,
) -> None:
    """Append one observation, keeping failed payloads out of ``result``.

    ``decide`` reads ``result`` expecting tool facts, so an error payload goes
    to ``rationale`` instead; the error keeps the record interpretable.
    """
    if status == "ok":
        observation = Observation(tool=name, args=args, status="ok", result=payload)
        if name == "inspect_image" and isinstance(payload.get("sha256"), str):
            record.image_id = payload["sha256"]
            record.sha256 = payload["sha256"]
    else:
        observation = Observation(
            tool=name, args=args, status="error", rationale=str(payload.get("error", status))
        )
        record.errors.append(f"{name}: {payload.get('error', status)}")
    record.observations.append(observation)


def _log_tool_call(name: str, args: dict[str, Any], payload: dict[str, Any], status: str) -> None:
    """Append one line to ``logs/agent.log``: time, tool, args, size, status.

    Never the full result and never a credential; a logging failure must not
    break an investigation, so OSError is swallowed.
    """
    try:
        encoded_args = json.dumps(args, default=str, sort_keys=True)
        encoded_payload = json.dumps(payload, default=str)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = (
            f"{stamp} tool={name} status={status} bytes={len(encoded_payload)} "
            f"args={encoded_args[:400]}"
        )
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _execute(tool_call: Any, ctx: ToolContext, record: InvestigationRecord) -> dict[str, Any]:
    """Run one tool call, recording and logging it. Never raises."""
    name = _tool_name(tool_call)
    try:
        arguments = json.loads(_tool_arguments(tool_call) or "{}")
    except (json.JSONDecodeError, TypeError):
        arguments = None
    if not isinstance(arguments, dict):
        payload = {
            "error": f"invalid JSON arguments for tool {name or '(unnamed)'}",
            "hint": "send arguments as a JSON object matching the tool's schema",
        }
        _observe(record, name, {}, payload, "error")
        _log_tool_call(name, {}, payload, "error")
        return payload

    payload = run_tool(name, arguments, ctx)
    status = "error" if "error" in payload else "ok"
    _observe(record, name, arguments, payload, status)
    _log_tool_call(name, arguments, payload, status)
    return payload


def _reflection_enabled(config: dict[str, Any]) -> bool:
    """Return whether the optional answer review is enabled."""
    return bool(config.get("reflection", False))


def reflect(answer: str, messages: list[dict[str, Any]], client: Any, model: str) -> str:
    """Review the answer against tool evidence; accept only explicit corrections."""
    check = (
        "Review your verdict against the tool results above. If it contradicts the evidence, "
        "reply with CORRECTED: followed by the corrected answer. Otherwise reply with OK."
    )
    review_messages = [*copy.deepcopy(messages), {"role": "user", "content": check}]
    response = client.chat.completions.create(model=model, messages=review_messages)
    text = str(getattr(response.choices[0].message, "content", None) or "")
    if "CORRECTED:" not in text:
        return answer
    return text.split("CORRECTED:", 1)[1].strip() or answer


def _model_verdict(answer: str) -> Verdict | None:
    """Read only clear verdict language from the answer, without guessing."""
    text = answer.casefold()
    if any(
        term in text
        for term in ("not tampered", "not manipulated", "not flagged", "no manipulation")
    ):
        return Verdict.NOT_FLAGGED_BY_THIS_DETECTOR
    if "inconclusive" in text or "unable to determine" in text:
        return Verdict.INCONCLUSIVE
    if "tampered" in text or "manipulated" in text or "flagged" in text:
        return Verdict.FLAGGED_FOR_REVIEW
    return None


def _assess_answer(answer: str, record: InvestigationRecord, grounded: Verdict) -> tuple[str, str]:
    """Append computed review/confidence guidance from recorded evidence."""
    from agent.config import get_config

    clean_lines = [
        line
        for line in answer.splitlines()
        if not line.startswith("Needs human review:")
        and line not in {"Confidence: low.", "Confidence: medium.", "Confidence: high."}
    ]
    answer = "\n".join(clean_lines).rstrip()
    settings = get_config().get("forensics", {})
    low = float(settings.get("uncertain_low", 0.4))
    high = float(settings.get("uncertain_high", 0.6))
    reasons: list[str] = []
    score = None
    validation_issues = False
    for observation in record.observations:
        result = observation.result or {}
        if observation.tool == "run_localizer":
            value = result.get("tampered_prob", result.get("score"))
            if isinstance(value, (int, float)):
                score = float(value)
        elif observation.tool == "validate_prediction" and result.get("issues"):
            validation_issues = True
    if score is not None and low <= score <= high:
        reasons.append(f"the localizer score {score:.4f} is in the uncertain range")
    if any(item.status in ("error", "timeout") for item in record.observations):
        reasons.append("a required tool failed")
    if validation_issues:
        reasons.append("validate_prediction reported issues")
    stated = _model_verdict(answer)
    if stated is not None and stated != grounded:
        reasons.append("the stated verdict disagrees with the evidence-based verdict")
    confidence = "low" if reasons else "high" if score is not None else "medium"
    if reasons:
        answer += "\n\nNeeds human review: " + "; ".join(reasons) + "."
    answer += f"\n\nConfidence: {confidence}."
    return answer, confidence


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def run_agent_full(
    goal: str,
    image_path: str,
    history: list[dict[str, Any]] | None = None,
    max_steps: int | None = None,
    client: Any | None = None,
    use_memory: bool = True,
    threshold: float | None = None,
) -> AgentResult:
    """Run one conversation to completion and return everything it produced."""
    config = get_agent_config()
    endpoints = model_entries(config)
    limit = int(config["max_steps"] if max_steps is None else max_steps)
    temperature = float(config["temperature"])

    messages = _initial_messages(goal, image_path, history, config)
    record = _new_record(goal, image_path)
    load_env()
    ctx = ToolContext(
        upload_root=_upload_root(),
        threshold=_default_threshold() if threshold is None else float(threshold),
    )

    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    memory = config.get("memory", {})
    memory_path = str(memory.get("path", "artifacts/memory/investigations.db"))
    # A follow-up belongs to the active conversation and should be answered
    # from its history, not short-circuited by the per-image cache.
    memory_enabled = use_memory and bool(memory.get("enabled", True)) and not history
    try:
        key = image_hash(image_path) if memory_enabled else ""
    except OSError:
        key = ""
    if key:
        saved = load_investigation(
            memory_path, key, endpoints[0]["model"], record.threshold, config_hash
        )
        if saved is not None:
            verdict, reasons = decide(saved.record)
            saved.record.verdict = verdict
            saved.record.verdict_reasons = reasons
            cached_answer, confidence = _assess_answer(saved.answer, saved.record, verdict)
            messages.append({"role": "assistant", "content": cached_answer})
            return AgentResult(
                answer=cached_answer,
                messages=copy.deepcopy(messages),
                record=saved.record,
                steps=0,
                stopped_reason="final_answer",
                grounded_verdict=verdict,
                errors=list(saved.record.errors),
                confidence=confidence,
            )

    endpoint_index = 0
    retries = 0
    steps = 0
    answer = ""
    stopped_reason = "final_answer"
    active = client
    clients: dict[int, Any] = {0: client} if client is not None else {}
    identical_calls: dict[str, int] = {}
    retry_limit_reached = False

    tried: list[str] = []
    unavailable: list[str] = []

    def select_usable(start: int) -> int | None:
        for candidate in range(start, len(endpoints)):
            candidate_model = str(endpoints[candidate]["model"])
            if _MODEL_COOLDOWNS.get(candidate_model, 0.0) > time.monotonic():
                continue
            if candidate not in clients:
                clients[candidate] = _build_client(endpoints[candidate])
            if clients[candidate] is None:
                env_name = str(endpoints[candidate]["api_key_env"])
                detail = f"skipped {candidate_model}: {env_name} not set"
                record.errors.append(detail)
                unavailable.append(detail)
                continue
            return candidate
        return None

    selected = select_usable(0)
    if selected is None:
        cooling = [str(item["model"]) for item in endpoints]
        detail = "no configured free-model credentials"
        if unavailable:
            detail += "; skipped: " + "; ".join(unavailable)
        return _unavailable(record, messages, steps, detail, cooling)
    endpoint_index = selected
    if active is None:
        active = clients[endpoint_index]

    while True:
        if steps >= limit:
            answer = STEP_LIMIT_MESSAGE
            stopped_reason = "max_steps"
            break

        endpoint = endpoints[endpoint_index]
        model_name = str(endpoint["model"])
        if not tried or tried[-1] != model_name:
            tried.append(model_name)
        try:
            response = active.chat.completions.create(
                model=endpoint["model"],
                messages=copy.deepcopy(messages),
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=temperature,
            )
        except Exception as exc:
            failure = _provider_failure(exc)
            if failure.category == "transient" and retries < MAX_PROVIDER_RETRIES:
                retries += 1
                delay = BACKOFF_SECONDS[min(retries - 1, len(BACKOFF_SECONDS) - 1)]
                if failure.retry_after is not None:
                    maximum = float(config.get("retry_after_max_seconds", 60))
                    delay = min(failure.retry_after, maximum)
                if delay:
                    time.sleep(delay)
                continue
            if failure.category == "fatal":
                detail = f"{model_name} failed ({failure.category}): {failure.cause}"
                unavailable.append(detail)
                record.errors.append(detail)
                next_index = select_usable(endpoint_index + 1)
                if next_index is None:
                    return _unavailable(record, messages, steps, "; ".join(unavailable), tried)
                endpoint_index = next_index
                if client is None:
                    active = clients[endpoint_index]
                retries = 0
                continue
            if failure.category == "quota":
                cooldown = float(config.get("cooldown_seconds", 900))
                _MODEL_COOLDOWNS[model_name] = time.monotonic() + cooldown
            detail = f"{model_name} failed ({failure.category}): {failure.cause}"
            unavailable.append(detail)
            record.errors.append(detail)
            next_index = select_usable(endpoint_index + 1)
            retries = 0
            if next_index is None:
                _log_switch(model_name, None, failure.category)
                models = ", ".join(tried)
                cause_text = "; ".join(unavailable)
                return _unavailable(
                    record,
                    messages,
                    steps,
                    f"{cause_text}; models tried: {models}",
                    tried,
                )
            next_model = str(endpoints[next_index]["model"])
            _log_switch(model_name, next_model, failure.category)
            record.errors.append(
                f"provider switched from {model_name} to {next_model}: {failure.category}"
            )
            endpoint_index = next_index
            if client is None:
                active = clients[endpoint_index]
            continue

        LAST_USED_MODELS[str(endpoint["provider"])] = model_name
        steps += 1
        retries = 0
        message = response.choices[0].message
        calls = list(getattr(message, "tool_calls", None) or [])

        if not calls:
            answer = str(getattr(message, "content", None) or "")
            messages.append(_assistant_message(message))
            if _reflection_enabled(config):
                try:
                    answer = reflect(answer, messages, active, endpoint["model"])
                except Exception as exc:
                    record.errors.append(f"reflection failed: {type(exc).__name__}")
            stopped_reason = "final_answer"
            break

        messages.append(_assistant_message(message))
        for index, call in enumerate(calls):
            try:
                parsed = json.loads(_tool_arguments(call) or "{}")
            except (json.JSONDecodeError, TypeError):
                parsed = None
            key_call = None
            if isinstance(parsed, dict):
                key_call = json.dumps(
                    [_tool_name(call), parsed], sort_keys=True, separators=(",", ":")
                )
                identical_calls[key_call] = identical_calls.get(key_call, 0) + 1
            retry_cap = int(config.get("max_identical_retries", 2))
            if key_call is not None and identical_calls[key_call] > retry_cap:
                payload = {
                    "error": "retry limit reached; change the arguments or use another tool",
                    "hint": "select a different tool or provide different arguments",
                }
                _observe(record, _tool_name(call), parsed, payload, "error")
                _log_tool_call(_tool_name(call), parsed, payload, "error")
                retry_limit_reached = True
            else:
                payload = _execute(call, ctx, record)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _call_id(call, index),
                    "content": json.dumps(payload, default=str),
                }
            )
        if retry_limit_reached:
            answer = "Stopped: the identical tool-call retry limit was reached."
            stopped_reason = "retry_limit"
            break

    verdict, reasons = decide(record)
    record.verdict = verdict
    record.verdict_reasons = reasons
    record.rationale = "; ".join(reasons)
    confidence = "low"
    if stopped_reason == "final_answer":
        answer, confidence = _assess_answer(answer, record, verdict)
        if messages and messages[-1].get("role") == "assistant":
            messages[-1]["content"] = answer
    if key and stopped_reason == "final_answer":
        save_investigation(
            memory_path,
            key,
            record,
            answer,
            str(endpoints[0]["model"]),
            record.threshold,
            config_hash,
        )
    return AgentResult(
        answer=answer,
        messages=messages,
        record=record,
        steps=steps,
        stopped_reason=stopped_reason,
        grounded_verdict=verdict,
        errors=list(record.errors),
        confidence=confidence,
    )


def _unavailable(
    record: InvestigationRecord,
    messages: list[dict[str, Any]],
    steps: int,
    detail: str,
    tried: list[str] | None = None,
) -> AgentResult:
    """Build the structured failure. Never a fabricated final answer."""
    record.errors.append(f"model unavailable: {detail}")
    verdict, reasons = decide(record)
    record.verdict = verdict
    record.verdict_reasons = reasons
    record.rationale = "; ".join(reasons)
    return AgentResult(
        answer=(
            f"Model unavailable: {detail}"
            + (
                f". Models tried: {', '.join(tried)}"
                if tried and "models tried:" not in detail
                else ""
            )
        ),
        messages=messages,
        record=record,
        steps=steps,
        stopped_reason="model_unavailable",
        grounded_verdict=verdict,
        errors=list(record.errors),
    )


def run_agent(
    goal: str,
    image_path: str,
    history: list[dict[str, Any]] | None = None,
    max_steps: int | None = None,
    client: Any | None = None,
    use_memory: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Run one conversation and return ``(answer_text, messages)``."""
    result = run_agent_full(
        goal=goal,
        image_path=image_path,
        history=history,
        max_steps=max_steps,
        client=client,
        use_memory=use_memory,
    )
    return result.answer, result.messages
