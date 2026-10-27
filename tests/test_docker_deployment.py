"""Static safety checks for the Docker deployment configuration."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_has_no_provider_key_or_key_build_argument():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "API_KEY" not in dockerfile
    assert not any(
        line.lstrip().startswith("ARG ") and "KEY" in line.upper()
        for line in dockerfile.splitlines()
    )


def test_dockerignore_excludes_secrets_and_runtime_data():
    ignored = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())

    assert {".env", ".env.*", ".git", "artifacts/", "logs/"}.issubset(ignored)


def test_deployment_docs_only_pass_provider_names_at_runtime():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    checklist = (ROOT / "docs" / "deploy_checklist.md").read_text(encoding="utf-8")

    assert "-e OPENROUTER_API_KEY -e GEMINI_API_KEY" in readme
    assert "[ ] **OpenRouter**" in checklist
    assert "[ ] **Gemini API**" in checklist
    assert "[x]" not in checklist.lower()
