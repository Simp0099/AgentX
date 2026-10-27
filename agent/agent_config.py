"""agent/agent_config.py: validated accessor for ``configs/agent.yaml``.

This is the single place the agent's provider, model, base URL and credential
NAME come from. Nothing here holds a credential value: ``api_key_env`` is the
*name* of an environment variable, resolved through :mod:`agent.env`.

Every validation error names the offending field or environment variable so a
misconfiguration is a one-line fix rather than a stack trace:

    AgentConfigError: configs/agent.yaml: missing required field 'model'
    AgentConfigError: environment variable 'OPENROUTER_API_KEY' is not set
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from agent.env import load_env

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "configs" / "agent.yaml"

# Top-level fields every agent config must define.
REQUIRED_FIELDS = (
    "provider",
    "model",
    "base_url",
    "api_key_env",
    "fallback_models",
    "max_steps",
    "temperature",
    "request_timeout_seconds",
    "retry_after_max_seconds",
    "cooldown_seconds",
    "max_identical_retries",
    "reflection",
    "memory",
    "history",
)

# Fields that identify one model endpoint (used for the primary and fallbacks).
MODEL_FIELDS = ("provider", "model", "base_url", "api_key_env")

REQUIRED_MEMORY_FIELDS = ("enabled", "path")
REQUIRED_HISTORY_FIELDS = ("max_messages", "max_chars")

_cache: dict[str, Any] | None = None


class AgentConfigError(RuntimeError):
    """Raised when the agent config is missing a field or a credential NAME."""


def _require_fields(section: Any, fields: tuple[str, ...], where: str) -> None:
    """Raise naming the first missing or empty field in ``section``."""
    if not isinstance(section, dict):
        raise AgentConfigError(f"{where}: expected a mapping, got {type(section).__name__}")
    for field_name in fields:
        if field_name not in section:
            raise AgentConfigError(f"{where}: missing required field '{field_name}'")
        if section[field_name] is None or section[field_name] == "":
            raise AgentConfigError(f"{where}: required field '{field_name}' is empty")


def _validate(config: Any, source: str) -> dict[str, Any]:
    """Validate the parsed YAML and return the ``agent:`` block."""
    if not isinstance(config, dict) or "agent" not in config:
        raise AgentConfigError(f"{source}: missing required section 'agent'")
    agent = config["agent"]
    _require_fields(agent, REQUIRED_FIELDS, source)

    fallbacks = agent["fallback_models"]
    if not isinstance(fallbacks, list) or not fallbacks:
        raise AgentConfigError(f"{source}: 'fallback_models' must be a non-empty list")
    for index, fallback in enumerate(fallbacks):
        _require_fields(fallback, MODEL_FIELDS, f"{source}: fallback_models[{index}]")

    _require_fields(agent["memory"], REQUIRED_MEMORY_FIELDS, f"{source}: memory")
    _require_fields(agent["history"], REQUIRED_HISTORY_FIELDS, f"{source}: history")

    # Numeric bounds: a zero or negative budget would make the loop a no-op.
    for field_name in (
        "max_steps", "request_timeout_seconds", "retry_after_max_seconds", "cooldown_seconds"
    ):
        value = agent[field_name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
            or (field_name != "cooldown_seconds" and value == 0)
        ):
            requirement = "non-negative" if field_name == "cooldown_seconds" else "positive"
            raise AgentConfigError(f"{source}: '{field_name}' must be a {requirement} number")
    if not 0.0 <= float(agent["temperature"]) <= 2.0:
        raise AgentConfigError(f"{source}: 'temperature' must be between 0.0 and 2.0")

    return agent


def get_agent_config(
    path: str | Path | None = None,
    *,
    require_credentials: bool = False,
    reload: bool = False,
) -> dict[str, Any]:
    """Load and validate ``configs/agent.yaml`` (cached after the first call).

    Args:
        path: override the config file location (tests only).
        require_credentials: also assert the credential environment variable is
            actually present. Off by default so configuration can be inspected
            without keys; the runtime client turns it on.
        reload: bypass the cache.

    Returns:
        The ``agent:`` mapping from the config file.

    Raises:
        AgentConfigError: naming the missing field, or the missing environment
            variable NAME (never its value).
    """
    global _cache
    source = str(path or _DEFAULT_PATH)
    if _cache is not None and not reload and path is None:
        agent = _cache
    else:
        config_path = Path(path) if path is not None else _DEFAULT_PATH
        if not config_path.is_file():
            raise AgentConfigError(f"{source}: agent config file not found")
        with open(config_path, encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
        agent = _validate(parsed, source)
        if path is None:
            _cache = agent

    if require_credentials:
        _assert_credentials(agent)

    return agent


def _assert_credentials(agent: dict[str, Any]) -> None:
    """Raise naming the missing environment variable, never its value."""
    load_env()
    env_names = [agent["api_key_env"]]
    env_names.extend(fallback["api_key_env"] for fallback in agent["fallback_models"])
    # At least one usable key is enough: the primary may be rate-limited while
    # a fallback on another provider still works.
    if any(os.environ.get(name) for name in env_names):
        return
    raise AgentConfigError(
        "no model credential is set; configure one of these environment "
        f"variables: {', '.join(sorted(set(env_names)))}"
    )


def model_entries(agent: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return the primary followed by every fallback, as model endpoint dicts."""
    config = agent if agent is not None else get_agent_config()
    primary = {field: config[field] for field in MODEL_FIELDS}
    ordered = [primary]
    for fallback in config["fallback_models"]:
        ordered.append({field: fallback[field] for field in MODEL_FIELDS})
    return ordered


def reset_cache() -> None:
    """Clear the cached config (tests only)."""
    global _cache
    _cache = None
