"""Tests for agent/api.py using FastAPI's TestClient.

The detector is a stub injected through ``set_detector``, so TensorFlow is never
imported and the real checkpoint is never opened. Upload paths are pointed at a
temporary directory via ``AGENT_UPLOAD_ROOT``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import api  # noqa: E402
from agent.agent_tools import TOOL_SCHEMAS, TOOLS  # noqa: E402


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


def test_tools_returns_the_same_schemas(client):
    response = client.get("/tools")

    assert response.status_code == 200
    assert response.json() == TOOL_SCHEMAS
    names = [entry["function"]["name"] for entry in response.json()]
    assert sorted(names) == sorted(TOOLS)


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
    response = client.post("/tools/inspect_image", json={"image_path": "/etc/hosts"})

    assert response.status_code == 200
    assert "outside the allowed upload directory" in response.json()["error"]


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
