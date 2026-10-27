"""Handler-level test without importing or starting Gradio."""

from pathlib import Path

from PIL import Image

from agent.schemas import InvestigationRecord, Observation, Verdict
from app import handler


def test_analyze_uses_orchestrator_record_and_renders_verdict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(handler, "UPLOAD_DIR", Path("uploads"))
    calls = []

    def fake_investigation(path, goal, config, detector):
        calls.append((path, goal, config, detector))
        assert Path(path).exists()
        return InvestigationRecord(
            investigation_id="handler-test",
            image_id=path,
            filename="random.png",
            goal=goal,
            goal_text="",
            threshold=config["threshold"],
            config_snapshot_hash="",
            model_version="stub",
            observations=[
                Observation("inspect_image", {"path": path}, "ok", {"valid": True}),
                Observation("run_localizer", {"path": path}, "ok",
                            {"tampered_prob": 0.83, "bbox": [0.1, 0.1, 0.8, 0.8]}),
                Observation("validate_prediction", {}, "ok", {"valid": True, "issues": []}),
                Observation("generate_ela", {}, "ok", {"artifact": "ela.png"}),
            ],
            errors=[],
            rationale="",
            verdict=Verdict.FLAGGED_FOR_REVIEW,
            verdict_reasons=[],
            created_at_utc="2026-01-01T00:00:00Z",
        )

    monkeypatch.setattr(handler, "run_investigation", fake_investigation)
    image = Image.new("RGB", (20, 16), "blue")
    image.filename = "sample.png"
    ela, scores, overlay, detail = handler.analyze(image, 0.5, detector=object())
    assert len(calls) == 1
    assert calls[0][1] == "quick_check"
    assert "Flagged for review" in detail
    assert "tampered score (uncalibrated)" in detail
    assert "inspect_image" in detail and "generate_ela" in detail
    assert "authentic" not in detail.lower()
    assert scores["tampered score (uncalibrated)"] == 0.83
    assert ela.size == image.size
    assert overlay.size == image.size
