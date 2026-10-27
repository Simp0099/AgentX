"""Tests for agent/env.py: credential loading that never exposes a value.

Nothing here touches a real credential. The variables exercised are invented
names written into temporary env files; the only assertion about a value is
that ``describe`` does not return it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import env as agent_env  # noqa: E402

DUMMY = "AGENT_ENV_TEST_DUMMY"
EXTERNAL = "AGENT_ENV_TEST_EXTERNAL"


def _write_env(path: Path, name: str, value: str) -> Path:
    path.write_text(f"# test fixture\n{name}={value}\n", encoding="utf-8")
    return path


def test_describe_missing(monkeypatch):
    monkeypatch.delenv(DUMMY, raising=False)
    assert agent_env.describe(DUMMY) == "missing"


def test_describe_reports_length_not_value(monkeypatch):
    monkeypatch.setenv(DUMMY, "abcdefg")
    described = agent_env.describe(DUMMY)
    assert described == "set (length 7)"
    assert "abcdefg" not in described


def test_load_env_reads_the_file_named_by_the_shell(monkeypatch, tmp_path):
    external = _write_env(tmp_path / "external.env", EXTERNAL, "from-file")
    monkeypatch.delenv(EXTERNAL, raising=False)
    monkeypatch.setenv(agent_env.ENV_FILE_VAR, str(external))

    loaded = agent_env.load_env(force=True)

    assert os.environ.get(EXTERNAL) == "from-file"
    assert str(external) in loaded


def test_load_env_never_overrides_a_shell_variable(monkeypatch, tmp_path):
    external = _write_env(tmp_path / "external.env", EXTERNAL, "from-file")
    monkeypatch.setenv(agent_env.ENV_FILE_VAR, str(external))
    monkeypatch.setenv(EXTERNAL, "from-shell")

    agent_env.load_env(force=True)

    assert os.environ[EXTERNAL] == "from-shell"


def test_load_env_ignores_a_missing_path(monkeypatch, tmp_path):
    missing = tmp_path / "nope.env"
    monkeypatch.setenv(agent_env.ENV_FILE_VAR, str(missing))

    loaded = agent_env.load_env(force=True)

    assert all("nope.env" not in path for path in loaded)


def test_get_secret_and_require(monkeypatch):
    monkeypatch.delenv(DUMMY, raising=False)
    assert agent_env.get_secret(DUMMY) is None
    with pytest.raises(RuntimeError) as excinfo:
        agent_env.require(DUMMY)
    # The error names the variable and never carries a value.
    assert DUMMY in str(excinfo.value)

    monkeypatch.setenv(DUMMY, "present")
    assert agent_env.get_secret(DUMMY) == "present"
    assert agent_env.require(DUMMY) == "present"
