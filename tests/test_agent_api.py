"""Tests for agent/api.py using FastAPI's TestClient.

The detector is a stub injected through ``set_detector``, so TensorFlow is never
imported and the real checkpoint is never opened. Upload paths are pointed at a
temporary directory via ``AGENT_UPLOAD_ROOT``.
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import api  # noqa: E402
from agent.agent_tools import TOOL_SCHEMAS, TOOLS  # noqa: E402
from agent.loop import AgentResult  # noqa: E402


class StubDetector:
    """Stand-in for TamperingDetector; returns a plausible raw payload."""

    def predict_raw(self, path):
        return {
            "tampered_prob": 0.4321,
            "bbox_raw": [0.1, 0.12, 0.88, 0.9],
            "image_size": [16, 16],
            "model_version": "stub-api",
        }


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    directory = tmp_path / "uploads"
    directory.mkdir()
    monkeypatch.setenv(api.ENV_UPLOAD_ROOT, str(directory))
    monkeypatch.delenv(api.ENV_API_TOKEN, raising=False)
    return directory


@pytest.fixture
def image(upload_dir):
    path = upload_dir / "photo.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def client(upload_dir):
    api.set_detector(StubDetector())
    with TestClient(api.app) as test_client:
        yield test_client
    api.reset_detector()


# ---------------------------------------------------------------------------
# Read-only endpoints
# ---------------------------------------------------------------------------


def test_health_reports_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_health_reports_llm_availability_without_credential_values(client, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["llm"]["available"] is False
    assert "OPENROUTER_API_KEY" in response.json()["llm"]["reason"]


def test_root_serves_standalone_dashboard(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>Deepfake Localizer</title>" in response.text
    assert "data-theme=\"dark\"" in response.text
    assert 'storageWrite("deepfake-localizer.theme", next)' in response.text


def test_browser_upload_is_saved_inside_upload_root(client, upload_dir):
    buffer = upload_dir / "source.png"
    Image.new("RGB", (8, 6), "navy").save(buffer)

    response = client.post(
        "/api/upload?filename=browser.png",
        content=buffer.read_bytes(),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    body = response.json()
    saved = Path(body["image_path"])
    assert body["filename"] == "browser.png"
    assert saved.is_file() and saved.parent == upload_dir.resolve()


def test_zip_upload_extracts_valid_images_and_skips_macos_and_non_images(client, upload_dir):
    image_buffer = BytesIO()
    Image.new("RGB", (9, 7), "navy").save(image_buffer, format="PNG")
    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("__MACOSX/._photo.png", b"resource fork")
        archive.writestr("folder/.DS_Store", b"metadata")
        archive.writestr("notes.txt", b"not an image")
        archive.writestr("folder/photo.png", image_buffer.getvalue())
        archive.writestr("broken.jpg", b"not a jpeg")

    response = client.post(
        "/api/upload?filename=photos.zip",
        content=archive_buffer.getvalue(),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["filename"] for item in body["images"]] == ["folder/photo.png"]
    extracted = Path(body["images"][0]["image_path"])
    assert extracted.is_file() and extracted.is_relative_to(upload_dir.resolve())
    assert {item["filename"] for item in body["warnings"]} == {"notes.txt", "broken.jpg"}


def test_zip_upload_rejects_slip_paths_and_requires_one_valid_image(client, upload_dir):
    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("../outside.png", b"not an image")
        archive.writestr("notes.txt", b"not an image")

    response = client.post(
        "/api/upload?filename=invalid.zip",
        content=archive_buffer.getvalue(),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["message"] == "archive contains no valid supported images"
    assert not (upload_dir.parent / "outside.png").exists()


def test_tools_returns_the_same_schemas(client):
    response = client.get("/tools")

    assert response.status_code == 200
    assert response.json() == TOOL_SCHEMAS
    names = [entry["function"]["name"] for entry in response.json()]
    assert sorted(names) == sorted(TOOLS)


def test_chat_returns_structured_evidence_card(client, image, monkeypatch):
    from app import chat_handler

    # Force the model-unavailable condition so this test deterministically
    # exercises the real fixed-pipeline fallback without making an LLM request.
    monkeypatch.setattr(
        chat_handler,
        "run_agent_full",
        lambda **_kwargs: AgentResult(
            answer="Model unavailable",
            messages=[],
            record=None,
            steps=0,
            stopped_reason="model_unavailable",
            errors=["RuntimeError: no model available"],
        ),
    )

    response = client.post(
        "/api/chat",
        json={"message": "Check this image", "image_path": str(image)},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["image"]["filename"] == image.name
    assert "tampered_score_uncalibrated" in body
    assert "answer_text" in body
    assert "tool_log" in body
    assert body["mode"] == "fixed_pipeline"
    saved_path = body["conversation_state"]["image_path"]
    followup = client.post(
        "/api/chat",
        json={
            "message": "What did you find?",
            "image_path": saved_path,
            "history": body["conversation_state"]["history"],
        },
    )
    assert followup.status_code == 200
    assert followup.json()["conversation_state"]["image_path"] == saved_path
    assert not body["answer_text"].startswith("Model unavailable:")


def test_analyze_image_endpoint_returns_results_and_ela(client, image):
    response = client.post(
        "/api/analyze-image",
        json={
            "image_path": str(image),
            "goal": "quick_check",
            "threshold": 0.5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["image"]["filename"] == image.name
    assert "tampered_score_uncalibrated" in payload
    assert payload["tool_log"]
    assert payload["conversation_state"]["image_path"] == str(image.resolve())


def test_chat_requires_key_and_sends_analysis_context(client, image, monkeypatch):
    from agent import env
    from app import chat_handler

    monkeypatch.setattr(env, "get_secret", lambda _name: None)
    missing = client.post(
        "/api/chat",
        json={"message": "What do you see?", "image_path": str(image), "require_llm": True},
    )
    assert missing.status_code == 503
    assert "OPENROUTER_API_KEY" in missing.json()["detail"]

    captured = {}

    def fake_chat(message, _image, _state, **_kwargs):
        captured["message"] = message
        return {"answer_text": "context used", "_state": {"history": [], "image_path": str(image)}}

    monkeypatch.setattr(chat_handler, "agent_chat", fake_chat)
    monkeypatch.setattr(env, "get_secret", lambda _name: "offline-test-key")
    response = client.post(
        "/api/chat",
        json={
            "message": "What do you see?",
            "image_path": str(image),
            "require_llm": True,
            "context": {
                "filename": "sample.jpg",
                "verdict": "Inconclusive",
                "tampered_score_uncalibrated": 0.4,
                "ela_available": True,
            },
        },
    )
    assert response.status_code == 200
    assert "sample.jpg" in captured["message"]
    assert "tampered_score_uncalibrated" in captured["message"]


def test_batch_route_returns_per_file_zip_results(client, upload_dir):
    from zipfile import ZIP_DEFLATED, ZipFile

    image_buffer = BytesIO()
    Image.new("RGB", (10, 8), "navy").save(image_buffer, format="JPEG")
    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr(".DS_Store", b"junk")
        archive.writestr("__MACOSX/._hidden.jpg", b"junk")
        archive.writestr("photo.jpg", image_buffer.getvalue())
        archive.writestr("notes.txt", b"unsupported")
    uploaded = client.post(
        "/api/upload?filename=photos.zip",
        content=archive_buffer.getvalue(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert uploaded.status_code == 200
    result = client.post("/api/batch", json={"zip_path": uploaded.json()["image_path"]})
    assert result.status_code == 200
    rows = result.json()["results"]
    assert any(row["filename"] == "photo.jpg" for row in rows)
    skipped = next(row for row in rows if row["filename"] == "notes.txt")
    assert skipped["status"] == "Skipped" and skipped["reason"] == "not an image"


def test_report_export_returns_generated_pdf(client, image):
    response = client.post(
        "/api/report",
        json={"image_paths": [str(image)], "threshold": 0.5},
    )

    assert response.status_code == 200
    report_path = Path(response.json()["report_path"])
    try:
        assert report_path.is_file()
        assert report_path.read_bytes().startswith(b"%PDF-")
    finally:
        report_path.unlink(missing_ok=True)


def test_app_imports_without_loading_the_model():
    # Importing the module must not have imported TensorFlow.
    assert "tensorflow" not in sys.modules or api._detector_loaded is False


# ---------------------------------------------------------------------------
# Validation and routing
# ---------------------------------------------------------------------------


def test_missing_image_path_is_a_422(client):
    response = client.post("/tools/inspect_image", json={})

    assert response.status_code == 422


def test_wrong_type_is_a_422(client):
    response = client.post("/tools/inspect_image", json={"image_path": 123})

    assert response.status_code == 422


def test_unknown_field_is_a_422(client, image):
    response = client.post(
        "/tools/inspect_image", json={"image_path": str(image), "nonsense": 1}
    )

    assert response.status_code == 422


def test_out_of_range_threshold_is_a_422(client, image):
    response = client.post(
        "/tools/run_localizer", json={"image_path": str(image), "threshold": 5}
    )

    assert response.status_code == 422


def test_unknown_tool_is_a_404_listing_the_available_tools(client):
    response = client.post("/tools/does_not_exist", json={})

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["available"] == sorted(TOOLS)


# ---------------------------------------------------------------------------
# Tool-level failures are normal results (200)
# ---------------------------------------------------------------------------


def test_bad_path_is_a_200_with_an_error_dict(client, upload_dir):
    response = client.post(
        "/tools/inspect_image", json={"image_path": str(upload_dir / "missing.png")}
    )

    assert response.status_code == 200
    body = response.json()
    assert "error" in body
    assert "hint" in body


def test_path_outside_upload_root_is_refused(client):
    response = client.post("/tools/inspect_image", json={"image_path": "/etc/passwd"})

    assert response.status_code == 200
    assert "outside the allowed upload directory" in response.json()["error"]


def test_api_rejects_parent_traversal_and_external_symlink(client, upload_dir):
    outside = upload_dir.parent / "outside.png"
    Image.new("RGB", (8, 8), "white").save(outside)
    symlink = upload_dir / "outside-link.png"
    symlink.symlink_to(outside)
    try:
        paths = [upload_dir / ".." / outside.name, symlink]
        for path in paths:
            response = client.post("/tools/inspect_image", json={"image_path": str(path)})
            assert response.status_code == 200
            assert "error" in response.json()
            assert "outside the allowed upload directory" in response.json()["error"]
    finally:
        symlink.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Happy path through the tools
# ---------------------------------------------------------------------------


def test_inspect_then_localize_then_validate(client, image):
    inspected = client.post("/tools/inspect_image", json={"image_path": str(image)})
    assert inspected.status_code == 200
    assert inspected.json()["valid"] is True

    localized = client.post(
        "/tools/run_localizer", json={"image_path": str(image), "threshold": 0.5}
    )
    assert localized.status_code == 200
    assert localized.json()["tampered_prob"] == pytest.approx(0.4321)


def test_tools_that_do_not_need_the_model_never_load_it(upload_dir):
    api.reset_detector()
    try:
        # TestClient without set_detector: nothing may trigger a model load.
        with TestClient(api.app) as test_client:
            response = test_client.post("/tools/inspect_image", json={"image_path": "/etc/hosts"})
        assert response.status_code == 200
        assert api._detector_loaded is False
    finally:
        api.reset_detector()


def test_exactly_one_log_line_per_request_without_argument_values(client, image):
    import logging

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    api.logger.addHandler(handler)
    try:
        client.post("/tools/inspect_image", json={"image_path": str(image)})
    finally:
        api.logger.removeHandler(handler)

    messages = [record.getMessage() for record in records]
    assert len(messages) == 1
    line = messages[0]
    assert "tool=inspect_image" in line
    assert "status=ok" in line
    assert "duration_ms=" in line
    # The path must never be logged: it is user data.
    assert str(image) not in line


def test_compare_evidence_accepts_an_empty_body(client, image):
    response = client.post("/tools/compare_evidence", json={})

    # No evidence recorded in this fresh context -> a normal error result.
    assert response.status_code == 200
    assert "error" in response.json()


# ---------------------------------------------------------------------------
# Optional shared-secret mode
# ---------------------------------------------------------------------------


def test_token_check_is_off_by_default(client, image):
    response = client.post("/tools/inspect_image", json={"image_path": str(image)})

    assert response.status_code == 200


def test_token_check_rejects_a_missing_header(upload_dir, monkeypatch):
    monkeypatch.setenv(api.ENV_API_TOKEN, "test-token-value")
    api.set_detector(StubDetector())
    try:
        with TestClient(api.app) as test_client:
            response = test_client.post("/tools/compare_evidence", json={})
        assert response.status_code == 401
    finally:
        api.reset_detector()


def test_token_check_accepts_the_right_header(upload_dir, monkeypatch):
    monkeypatch.setenv(api.ENV_API_TOKEN, "test-token-value")
    api.set_detector(StubDetector())
    try:
        with TestClient(api.app) as test_client:
            response = test_client.post(
                "/tools/compare_evidence", json={}, headers={api.TOKEN_HEADER: "test-token-value"}
            )
        assert response.status_code == 200
    finally:
        api.reset_detector()
