"""Tests for scripts/probe_models.py helpers and configs/agent_candidates.yaml.

No network call and no credential is involved: the probe's parsing and
selection rules are pure functions, and the candidate file is only inspected
for structure and for accidentally embedded secrets.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "probe_models", ROOT / "scripts" / "probe_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


def _message(name: str | None = None, arguments: str | None = None):
    """Build a minimal assistant message shaped like the OpenAI client's."""
    if name is None:
        return SimpleNamespace(tool_calls=[])
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(tool_calls=[SimpleNamespace(function=function)])


def test_extract_call_accepts_exact_tool_call():
    passed, detail = probe._extract_call(_message("get_number", '{"n": 7}'))
    assert passed is True
    assert "get_number" in detail


def test_extract_call_rejects_missing_tool_calls():
    passed, detail = probe._extract_call(_message())
    assert passed is False
    assert "no tool_calls" in detail


def test_extract_call_rejects_wrong_function_name():
    passed, _detail = probe._extract_call(_message("get_number_v2", '{"n": 7}'))
    assert passed is False


def test_extract_call_rejects_invalid_json():
    passed, detail = probe._extract_call(_message("get_number", "{n: 7}"))
    assert passed is False
    assert "valid JSON" in detail


def test_extract_call_rejects_wrong_argument_value():
    passed, detail = probe._extract_call(_message("get_number", '{"n": 6}'))
    assert passed is False
    assert "wrong n" in detail


def test_extract_call_rejects_non_object_arguments():
    passed, _detail = probe._extract_call(_message("get_number", "[7]"))
    assert passed is False


def _result(name: str, passes: int, latency: float | None, status: str = "ok") -> dict:
    return {
        "name": name,
        "passes": passes,
        "median_latency_ms": latency,
        "status": status,
        "trials_planned": 5,
        "provider": "p",
        "model": "m",
        "base_url": "https://example.invalid/v1",
        "api_key_env": "SOME_KEY",
        "key_status": "set (length 3)",
        "http_errors": [],
        "rate_limit_messages": [],
    }


def test_select_ranks_by_passes_then_latency():
    results = [
        _result("slow_but_perfect", 5, 900.0),
        _result("fast_but_perfect", 5, 100.0),
        _result("partial", 3, 50.0),
        _result("dead", 0, None, status="unreachable"),
    ]

    primary, fallbacks = probe.select(results)

    assert primary["name"] == "fast_but_perfect"
    assert [item["name"] for item in fallbacks] == ["slow_but_perfect", "partial"]


def test_select_returns_none_when_nothing_passes():
    primary, fallbacks = probe.select([_result("dead", 0, None, status="unreachable")])
    assert primary is None
    assert fallbacks == []


def test_verdict_labels_every_outcome():
    assert probe.verdict_for(_result("a", 5, 1.0)) == "passed tool-calling"
    assert probe.verdict_for(_result("b", 0, 1.0)) == "failed tool-calling"
    assert probe.verdict_for(_result("c", 2, 1.0)) == "partial tool-calling (2/5)"
    assert probe.verdict_for(_result("d", 0, None, status="unreachable")) == "unreachable"
    assert probe.verdict_for(_result("e", 0, None, status="missing key")) == "missing key"


def test_markdown_table_covers_every_candidate():
    results = [_result("one", 5, 10.0), _result("two", 0, None, status="unreachable")]
    table = probe.markdown_table(results)
    assert "`one`" in table
    assert "`two`" in table
    assert table.count("\n") >= len(results) + 1


def test_candidate_config_is_structural_and_secret_free():
    data = yaml.safe_load((ROOT / "configs" / "agent_candidates.yaml").read_text(encoding="utf-8"))
    candidates = data["candidates"]
    assert len(candidates) >= 9

    key_pattern = re.compile(r"(AIza[0-9A-Za-z_-]{10,}|sk-[0-9A-Za-z]{10,}|ghp_[0-9A-Za-z]{10,})")
    for candidate in candidates:
        assert candidate["name"]
        assert candidate["base_url"].startswith("https://")
        assert candidate["model"]
        # Credentials are referenced by NAME only, as an upper-case env var.
        assert candidate["api_key_env"].isupper()
        for value in candidate.values():
            assert not key_pattern.search(str(value)), candidate["name"]

    # Every provider block likewise names its key rather than holding one.
    for provider in data["providers"].values():
        assert provider["api_key_env"].isupper()
        assert known_url(provider["base_url"])


def known_url(url: str) -> bool:
    """Return True for a documented provider endpoint (no invented hosts)."""
    allowed = (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "https://openrouter.ai/api/v1",
        "https://opencode.ai/zen/v1",
    )
    return url in allowed
