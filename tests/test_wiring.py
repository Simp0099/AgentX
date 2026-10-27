"""Static app wiring contract tests that do not import Gradio."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_app_imports_orchestrator_and_is_single_gradio_entrypoint():
    app_text = (ROOT / "app" / "app.py").read_text()
    handler_text = (ROOT / "app" / "handler.py").read_text()
    assert "from app.handler import analyze" in app_text
    assert "from agent.orchestrator import run_investigation" in handler_text
    gradio_apps = []
    for path in ROOT.rglob("*.py"):
        if any(part.startswith(".") for part in path.parts):
            continue
        source = path.read_text(errors="ignore")
        if "gr." + "Blocks(" in source or "gr." + "Interface(" in source:
            gradio_apps.append(path.relative_to(ROOT).as_posix())
    assert gradio_apps == ["app/app.py"]


def test_gradio_app_wires_batch_zip_processor():
    app_text = (ROOT / "app" / "app.py").read_text()
    assert "from agent.batch import process_batch" in app_text
    assert "def analyze_batch(zip_file):" in app_text
    assert "batch_button.click(analyze_batch" in app_text
