"""Handler-level test without importing or starting Gradio."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from PIL import Image

from agent.schemas import InvestigationRecord, Observation, Verdict
from app import handler


def test_analyze_uses_orchestrator_record_and_renders_verdict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(handler, "UPLOAD_DIR", Path("uploads"))
    ela_path = tmp_path / "ela.png"
    Image.new("RGB", (20, 16), "white").save(ela_path)
    calls = []

    def fake_investigation(path, goal, config, detector, goal_text=""):
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
                Observation(
                    "run_localizer",
                    {"path": path},
                    "ok",
                    {"tampered_prob": 0.83, "bbox": [0.1, 0.1, 0.8, 0.8]},
                ),
                Observation("validate_prediction", {}, "ok", {"valid": True, "issues": []}),
                Observation("generate_ela", {}, "ok", {"artifact": str(ela_path)}),
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
    assert "Checks not run" in detail and "Errors" in detail
    assert "Model-predicted tampered region (ELA-based)" not in detail
    assert "authentic" not in detail.lower()
    assert scores["tampered score (uncalibrated)"] == 0.83
    assert ela.size == image.size
    assert overlay.size == image.size


def test_simultaneous_analyze_calls_keep_thresholds_isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(handler, "UPLOAD_DIR", Path("uploads"))
    simultaneous = Barrier(2)

    def fake_investigation(path, goal, config, detector, goal_text=""):
        simultaneous.wait(timeout=5)
        score = 0.6
        verdict = (
            Verdict.FLAGGED_FOR_REVIEW
            if score >= config["threshold"]
            else Verdict.NOT_FLAGGED_BY_THIS_DETECTOR
        )
        return InvestigationRecord(
            investigation_id=f"threshold-{config['threshold']}",
            image_id=path,
            filename=Path(path).name,
            goal=goal,
            goal_text="",
            threshold=config["threshold"],
            config_snapshot_hash="",
            model_version="stub",
            observations=[
                Observation("inspect_image", {"path": path}, "ok", {"valid": True}),
                Observation(
                    "run_localizer",
                    {"path": path},
                    "ok",
                    {"tampered_prob": score, "bbox": [0.1, 0.1, 0.7, 0.7]},
                ),
                Observation("validate_prediction", {}, "ok", {"valid": True, "issues": []}),
            ],
            errors=[],
            rationale="",
            verdict=verdict,
            verdict_reasons=[],
            created_at_utc="2026-01-01T00:00:00Z",
        )

    monkeypatch.setattr(handler, "run_investigation", fake_investigation)

    def analyze_at(threshold):
        image = Image.new("RGB", (24, 20), "green")
        image.filename = f"threshold-{threshold}.png"
        return handler.analyze(image, threshold, detector=object())[3]

    with ThreadPoolExecutor(max_workers=2) as executor:
        low = executor.submit(analyze_at, 0.3)
        high = executor.submit(analyze_at, 0.8)
        low_detail, high_detail = low.result(timeout=10), high.result(timeout=10)

    assert "Flagged for review" in low_detail
    assert "Not flagged by this detector" in high_detail


def test_multiple_upload_component_outputs_one_image_without_error(tmp_path, monkeypatch):
    from app import app as gradio_app

    monkeypatch.setattr(gradio_app, "DETECTOR", object())
    monkeypatch.setattr(
        gradio_app,
        "_analyze",
        lambda image, threshold, **kwargs: (
            Image.new("RGB", image.size, "white"),
            {"tampered score (uncalibrated)": 0.2},
            Image.new("RGB", image.size, "blue"),
            "### Verdict: **Not flagged by this detector**",
        ),
    )
    source = tmp_path / "one.png"
    Image.new("RGB", (20, 16), "green").save(source)
    original, ela, overlay, details = gradio_app.analyze_uploads(
        [str(source)],
        0.5,
        "Quick check",
        "",
    )
    assert len(original) == len(ela) == len(overlay) == 1
    assert "one.png" in original[0][1]
    assert "Model-predicted tampered region (ELA-based)" in overlay[0][1]
    assert "Not flagged by this detector" in details
