"""Tests for agent/agent_config.py and the "config lives in configs/" rule.

Nothing here needs a credential: the config is validated structurally, and the
one credential test points at an environment variable name that cannot exist on
any machine.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.agent_config import (  # noqa: E402
    AgentConfigError,
    get_agent_config,
    model_entries,
    reset_cache,
)

CONFIG_PATH = ROOT / "configs" / "agent.yaml"

# A variable name that is guaranteed not to be set anywhere.
UNSET_KEY = "AGENT_CONFIG_TEST_UNSET_KEY"

# Model names and provider endpoints must live in configs/, never in agent/*.py.
FORBIDDEN_IN_AGENT_SOURCES = (
    # provider endpoints
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "openrouter.ai",
    "opencode.ai",
    # model names / families
    "gpt-4",
    "gpt-3",
    "gemini-",
    "claude-",
    "nemotron",
    "big-pickle",
    "local-fixed",
)


def _mutated_config(tmp_path: Path, mutate) -> Path:
    """Write a copy of the real config after ``mutate`` edits the dict."""
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(data)
    target = tmp_path / "agent.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_cache()
    yield
    reset_cache()


def test_config_loads_with_every_required_field():
    config = get_agent_config()

    for field in (
        "provider",
        "model",
        "base_url",
        "api_key_env",
        "max_steps",
        "temperature",
        "request_timeout_seconds",
        "max_identical_retries",
        "reflection",
        "memory",
        "history",
        "fallback_models",
    ):
        assert field in config, field

    assert config["max_steps"] == 6
    assert config["temperature"] == 0.0
    assert config["max_identical_retries"] == 2
    assert config["reflection"] is False
    assert config["memory"]["enabled"] is True
    assert config["memory"]["path"].startswith("artifacts/")
    assert config["history"]["max_messages"] > 0
    assert config["history"]["max_chars"] > 0
    assert config["fallback_models"], "at least one fallback is required"


def test_primary_is_the_phase_one_selection():
    config = get_agent_config()

    assert config["provider"] == "openrouter"
    assert config["model"] == "nvidia/nemotron-3.5-lightning:free"
    assert config["api_key_env"] == "OPENROUTER_API_KEY"


@pytest.mark.parametrize(
    ("provider", "key_name", "base_url"),
    [
        ("anthropic", "ANTHROPIC_API_KEY", "https://api.anthropic.com/v1"),
        ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1"),
    ],
)
def test_env_provider_settings_override_primary_without_fallbacks(
    monkeypatch, provider, key_name, base_url
):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "17")

    config = get_agent_config()
    endpoints = model_entries(config)

    assert config["provider"] == provider
    assert config["api_key_env"] == key_name
    assert config["base_url"] == base_url
    assert config["request_timeout_seconds"] == 17
    assert config["fallback_models"] == []
    assert len(endpoints) == 1


def test_model_entries_are_primary_first_then_fallbacks():
    config = get_agent_config()

    entries = model_entries(config)

    assert entries[0]["model"] == config["model"]
    assert len(entries) == 1 + len(config["fallback_models"])
    for entry in entries:
        assert set(entry) == {"provider", "model", "base_url", "api_key_env"}


def test_missing_field_names_the_field(tmp_path):
    target = _mutated_config(tmp_path, lambda data: data["agent"].pop("model"))

    with pytest.raises(AgentConfigError) as excinfo:
        get_agent_config(target)

    message = str(excinfo.value)
    assert "model" in message
    assert "missing required field" in message


def test_missing_section_is_reported(tmp_path):
    target = tmp_path / "empty.yaml"
    target.write_text("something_else: {}\n", encoding="utf-8")

    with pytest.raises(AgentConfigError) as excinfo:
        get_agent_config(target)

    assert "agent" in str(excinfo.value)


def test_fallback_missing_a_field_names_its_index(tmp_path):
    def mutate(data):
        data["agent"]["fallback_models"][1].pop("base_url")

    target = _mutated_config(tmp_path, mutate)

    with pytest.raises(AgentConfigError) as excinfo:
        get_agent_config(target)

    message = str(excinfo.value)
    assert "fallback_models[1]" in message
    assert "base_url" in message


def test_missing_credential_names_the_variable_not_a_value(tmp_path):
    def mutate(data):
        data["agent"]["api_key_env"] = UNSET_KEY
        for fallback in data["agent"]["fallback_models"]:
            fallback["api_key_env"] = UNSET_KEY

    target = _mutated_config(tmp_path, mutate)

    with pytest.raises(AgentConfigError) as excinfo:
        get_agent_config(target, require_credentials=True)

    message = str(excinfo.value)
    assert UNSET_KEY in message


def test_zero_step_budget_is_rejected(tmp_path):
    target = _mutated_config(tmp_path, lambda data: data["agent"].__setitem__("max_steps", 0))

    with pytest.raises(AgentConfigError) as excinfo:
        get_agent_config(target)

    assert "max_steps" in str(excinfo.value)


def test_no_model_name_or_provider_url_appears_in_agent_sources():
    offenders = []
    for source in sorted((ROOT / "agent").glob("*.py")):
        text = source.read_text(encoding="utf-8")
        for token in FORBIDDEN_IN_AGENT_SOURCES:
            if token in text:
                offenders.append(f"{source.name} contains {token!r}")

    assert not offenders, (
        "model names and provider URLs belong in configs/, not agent/*.py: "
        + "; ".join(offenders)
    )


def test_config_keeps_credentials_by_name_only():
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)

    names = [data["agent"]["api_key_env"]]
    names.extend(item["api_key_env"] for item in data["agent"]["fallback_models"])
    for name in names:
        # An environment variable NAME, never a credential value.
        assert name.isupper(), f"{name} should be an environment variable NAME"

    # No long opaque token (a pasted key) anywhere in the file. Model ids and
    # URLs are excluded by charset: they carry '/', ':' or '.'.
    opaque = re.compile(r"^[A-Za-z0-9_-]{32,}$")
    for line in raw.splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        value = line.split(":", 1)[-1].strip().strip('"').strip("'")
        assert not opaque.match(value), f"suspicious value in configs/agent.yaml: {line}"
