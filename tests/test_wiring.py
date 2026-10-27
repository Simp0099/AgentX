"""Static app wiring contract tests that do not import Gradio."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_app_imports_orchestrator_and_is_single_gradio_entrypoint():
    app_text = (ROOT / "app" / "app.py").read_text()
    assert "from agent import orchestrator" in app_text
    gradio_apps = []
    for path in ROOT.rglob("*.py"):
        if any(part.startswith(".") for part in path.parts):
            continue
        source = path.read_text(errors="ignore")
        if "gr." + "Blocks(" in source or "gr." + "Interface(" in source:
            gradio_apps.append(path.relative_to(ROOT).as_posix())
    assert gradio_apps == ["app/app.py"]
