"""Offline tests for the typed, bounded forensic tool contracts."""

import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from agent.schemas import ToolResult
from agent.tools.compare_evidence import compare_evidence
from agent.tools.generate_ela import generate_ela
from agent.tools.inspect_image import inspect_image
from agent.tools.inspect_metadata import inspect_metadata
from agent.tools.run_localizer import run_localizer
from agent.tools.validate_prediction import validate_prediction


def _assert_result(result, status):
    assert isinstance(result, ToolResult)
    assert result.status == status


def _slow(*args, **kwargs):
    time.sleep(0.1)


class SlowExecutor(ThreadPoolExecutor):
    def submit(self, fn, /, *args, **kwargs):
        return super().submit(_slow)


def test_inspect_image_success_failure_timeout(tmp_path):
    path = tmp_path / "valid.png"
    Image.new("RGB", (8, 8), "red").save(path)
    good = inspect_image(path)
    _assert_result(good, "ok")
    assert good.result["sha256"]
    assert good.result["width"] == 8
    assert good.result["issues"] == []
    _assert_result(inspect_image(tmp_path / "missing.png"), "error")
    from unittest.mock import patch

    with patch("agent.tools.common.ThreadPoolExecutor", SlowExecutor):
        _assert_result(inspect_image(path, timeout=0.001), "timeout")


def test_validate_prediction_success_failure_timeout():
    _assert_result(validate_prediction(0.8, [0.1, 0.1, 0.5, 0.5], 100, 100), "ok")
    bad = validate_prediction(float("nan"), [0, 0, 1, 1], 100, 100)
    _assert_result(bad, "ok")
    assert bad.result["valid"] is False
    from unittest.mock import patch

    with patch("agent.tools.common.ThreadPoolExecutor", SlowExecutor):
        _assert_result(validate_prediction(0.8, None, 100, 100, timeout=0.001), "timeout")


def test_generate_ela_success_failure_timeout(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "blue").save(path)
    result = generate_ela(path)
    _assert_result(result, "ok")
    assert (tmp_path.parent / result.result["artifact"]).exists() or __import__("pathlib").Path(
        result.result["artifact"]
    ).exists()
    _assert_result(generate_ela(tmp_path / "missing.png"), "error")
    from unittest.mock import patch

    with patch("agent.tools.common.ThreadPoolExecutor", SlowExecutor):
        _assert_result(generate_ela(path, timeout=0.001), "timeout")


def test_inspect_metadata_success_failure_timeout(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (8, 8)).save(path)
    result = inspect_metadata(path)
    _assert_result(result, "ok")
    assert result.result["exif_status"] == "absent"
    assert result.result["present_fields"] == {}
    assert "Make" in result.result["absent_fields"]
    assert result.result["parse_errors"] == []
    _assert_result(inspect_metadata(tmp_path / "missing.png"), "error")
    from unittest.mock import patch

    with patch("agent.tools.common.ThreadPoolExecutor", SlowExecutor):
        _assert_result(inspect_metadata(path, timeout=0.001), "timeout")


def test_inspect_metadata_reports_present_exif_fields(tmp_path):
    path = tmp_path / "with-exif.jpg"
    image = Image.new("RGB", (8, 8), "white")
    exif = image.getexif()
    exif[271] = "Test Camera"
    image.save(path, exif=exif.tobytes())

    result = inspect_metadata(path)
    _assert_result(result, "ok")
    assert result.result["present_fields"]["Make"] == "Test Camera"
    assert "Model" in result.result["absent_fields"]
    assert result.result["parse_errors"] == []


def test_run_localizer_success_failure_timeout(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(path)

    class StubDetector:
        def predict_raw(self, image_path):
            assert str(image_path) == str(path)
            return {
                "tampered_prob": 0.8,
                "bbox_raw": [0.1, 0.1, 0.8, 0.8],
                "image_size": (8, 8),
                "model_version": "stub",
            }

    result = run_localizer(path, detector=StubDetector())
    _assert_result(result, "ok")
    assert result.result["tampered_prob"] == 0.8
    assert result.result["label"] == "tampered"
    assert result.duration_ms >= 0
    _assert_result(run_localizer(path, detector=object()), "error")
    from unittest.mock import patch

    with patch("agent.tools.common.ThreadPoolExecutor", SlowExecutor):
        _assert_result(run_localizer(path, detector=StubDetector(), timeout=0.001), "timeout")


def test_malformed_model_result_fails_prediction_validation():
    malformed = {
        "tampered_prob": float("nan"),
        "bbox": [0.1, 0.1, 0.8, 0.8],
        "image_size": (100, 100),
    }
    result = validate_prediction(malformed)
    _assert_result(result, "ok")
    assert result.result["valid"] is False
    assert "score must be finite" in result.result["issues"]


def test_tools_use_configured_default_timeout(tmp_path):
    from unittest.mock import patch

    path = tmp_path / "valid.png"
    Image.new("RGB", (8, 8), "red").save(path)
    config = {"forensics": {"tool_timeout_seconds": 0.001}}
    with patch("agent.config.get_config", return_value=config):
        with patch("agent.tools.common.ThreadPoolExecutor", SlowExecutor):
            _assert_result(inspect_image(path), "timeout")


def test_compare_evidence_success_failure_timeout():
    record = {
        "threshold": 0.5,
        "observations": [
            {"tool": "run_localizer", "result": {"tampered_prob": 0.8}},
            {"tool": "validate_prediction", "result": {"valid": False, "issues": ["bad box"]}},
        ],
    }
    # dict observations are normalized to the same fields used by the record API.
    from types import SimpleNamespace

    record["observations"] = [SimpleNamespace(**item) for item in record["observations"]]
    result = compare_evidence(record)
    _assert_result(result, "ok")
    assert result.result["conflicts"]
    _assert_result(compare_evidence(None), "error")
    from unittest.mock import patch

    with patch("agent.tools.common.ThreadPoolExecutor", SlowExecutor):
        _assert_result(compare_evidence(record, timeout=0.001), "timeout")


def test_absent_metadata_is_not_a_conflict():
    from types import SimpleNamespace

    record = {
        "threshold": 0.5,
        "observations": [
            SimpleNamespace(tool="run_localizer", result={"tampered_prob": 0.2}),
            SimpleNamespace(tool="validate_prediction", result={"valid": True, "issues": []}),
            SimpleNamespace(
                tool="inspect_metadata",
                result={"exif_status": "absent", "present_fields": {}, "absent_fields": ["Make"]},
            ),
        ],
    }
    result = compare_evidence(record)
    _assert_result(result, "ok")
    assert result.result["conflicts"] == []
