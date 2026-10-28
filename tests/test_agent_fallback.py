from __future__ import annotations

from agent import loop
from agent.agent_config import get_agent_config
from tests.fakes import FakeClient, make_message


def setup(monkeypatch, tmp_path):
    config = get_agent_config().copy()
    config["memory"] = {"enabled": False, "path": str(tmp_path / "mem.db")}
    monkeypatch.setattr(loop, "get_agent_config", lambda: config)
    monkeypatch.setattr(loop, "load_env", lambda: None)
    monkeypatch.setattr(loop, "BACKOFF_SECONDS", (0, 0))
    loop._MODEL_COOLDOWNS.clear()
    from PIL import Image
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image)
    return loop.model_entries(config), image


def test_missing_primary_key_falls_through(tmp_path, monkeypatch):
    endpoints, image = setup(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    clients = {endpoints[1]["model"]: FakeClient([make_message(content="Gemini answer")])}
    monkeypatch.setattr(loop, "_build_client", lambda endpoint: clients.get(endpoint["model"]))
    result = loop.run_agent_full("quick_check", str(image), use_memory=False)
    assert result.answer.startswith("Gemini answer")
    assert f"skipped {endpoints[0]['model']}: OPENROUTER_API_KEY not set" in result.errors


def test_429_on_primary_uses_next_provider(tmp_path, monkeypatch):
    endpoints, image = setup(monkeypatch, tmp_path)
    clients = {
        endpoints[0]["model"]: FakeClient([RuntimeError("429 quota exceeded")]),
        endpoints[1]["model"]: FakeClient([make_message(content="Gemini answer")]),
    }
    monkeypatch.setattr(loop, "_build_client", lambda endpoint: clients[endpoint["model"]])
    result = loop.run_agent_full("quick_check", str(image), use_memory=False)
    assert result.answer.startswith("Gemini answer")


def test_all_missing_reports_variable_names(tmp_path, monkeypatch):
    endpoints, image = setup(monkeypatch, tmp_path)
    for name in {endpoint["api_key_env"] for endpoint in endpoints}:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(loop, "_build_client", lambda _endpoint: None)
    result = loop.run_agent_full("quick_check", str(image), use_memory=False)
    assert result.stopped_reason == "model_unavailable"
    assert all(endpoint["api_key_env"] in result.answer for endpoint in endpoints)


def test_env_loader_is_called_before_key_lookup(tmp_path, monkeypatch):
    endpoints, _image = setup(monkeypatch, tmp_path)
    loaded = []
    def load_file():
        loaded.append(True)
        monkeypatch.setenv("OPENROUTER_API_KEY", "file-key")
    monkeypatch.setattr(loop, "load_env", load_file)
    monkeypatch.setattr(loop, "get_agent_config", lambda: {"request_timeout_seconds": 1})
    import sys
    class SDK:
        class OpenAI:
            def __init__(self, **kwargs):
                self.key = kwargs["api_key"]
    monkeypatch.setitem(sys.modules, "openai", SDK())
    client = loop._build_client(endpoints[0])
    assert loaded and client.key == "file-key"


def test_shell_variable_overrides_env_loader(tmp_path, monkeypatch):
    endpoints, _image = setup(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "shell-key")
    monkeypatch.setattr(loop, "load_env", lambda: None)
    monkeypatch.setattr(loop, "get_agent_config", lambda: {"request_timeout_seconds": 1})
    import sys
    class SDK:
        class OpenAI:
            def __init__(self, **kwargs):
                self.key = kwargs["api_key"]
    monkeypatch.setitem(sys.modules, "openai", SDK())
    assert loop._build_client(endpoints[0]).key == "shell-key"
