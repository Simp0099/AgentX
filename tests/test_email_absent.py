"""The optional email phase is explicitly deleted rather than stubbed."""

from pathlib import Path


def test_email_not_implemented_and_no_module_exists():
    root = Path(__file__).resolve().parents[1]
    readme = " ".join((root / "README.md").read_text().lower().split())
    assert "email is not implemented" in readme
    assert not list(root.glob("agent/email*.py"))
