"""Tests for agent/agent_tools.py: the model-facing tool API.

Everything runs offline on a tiny synthetic image with a stub detector, so the
checkpoint is never loaded and no network call is made.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.agent_tools import TOOL_SCHEMAS, TOOLS, ToolContext, run_tool  # noqa: E402
from agent.orchestrator import build_tool_schemas  # noqa: E402


class StubDetector:
    """Minimal stand-in for TamperingDetector.predict_raw."""

    def predict_raw(self, path):
        return {
            "tampered_prob": np.float32(0.4321),  # numpy type on purpose
            "bbox_raw": (0.1, 0.12, 0.88, 0.9),  # tuple and floats
            "image_size": (16, 16),
            "model_version": "stub-1",
        }


def _schema_names() -> list[str]:
    return [entry["function"]["name"] for entry in TOOL_SCHEMAS]


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "photo.png"
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return path


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(detector=StubDetector(), upload_root=tmp_path, threshold=0.5)


# ---------------------------------------------------------------------------
# Schema shape
# ---------------------------------------------------------------------------


def test_every_schema_is_openai_function_format():
    for entry in TOOL_SCHEMAS:
        assert entry["type"] == "function"
        function = entry["function"]
        assert set(function) == {"name", "description", "parameters"}
        assert isinstance(function["description"], str) and function["description"]
        assert isinstance(function["parameters"], dict)


def test_every_parameters_object_is_a_valid_json_schema():
    for entry in TOOL_SCHEMAS:
        parameters = entry["function"]["parameters"]
        # Raises SchemaError if the parameters block is not a valid schema.
        jsonschema.Draft202012Validator.check_schema(parameters)
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False
        assert "required" in parameters
        for name in parameters["required"]:
            assert name in parameters["properties"]


def test_tool_names_match_between_tools_and_schemas():
    schema_names = _schema_names()
    assert all(name in TOOLS for name in schema_names)
    assert sorted(schema_names) == sorted(TOOLS)
    assert len(set(_schema_names())) == len(_schema_names())


def test_orchestrator_exposes_the_same_schemas():
    built = build_tool_schemas()

    assert built
    for entry in built:
        assert entry["type"] == "function"
        assert set(entry["function"]) == {"name", "description", "parameters"}
    assert sorted(entry["function"]["name"] for entry in built) == sorted(TOOLS)


def test_descriptions_state_limitations():
    by_name = {entry["function"]["name"]: entry["function"]["description"] for entry in TOOL_SCHEMAS}

    localizer = by_name["run_localizer"].lower()
    assert "uncalibrated" in localizer
    assert "casia v2" in localizer
    assert "cannot detect fully ai-generated images" in localizer
    assert "re-compressed" in localizer


def test_unknown_tool_lists_the_available_tools():
    result = run_tool("nonexistent", {})

    assert "error" in result
    assert result["available"] == sorted(TOOLS)


# ---------------------------------------------------------------------------
# Behaviour on a real synthetic image
# ---------------------------------------------------------------------------


def test_inspect_image_returns_a_json_safe_digest(image, ctx):
    result = run_tool("inspect_image", {"image_path": str(image)}, ctx)

    assert "error" not in result
    assert result["valid"] is True
    assert result["width"] == 16 and result["height"] == 16
    assert len(result["sha256"]) == 64
    json.dumps(result)


def test_every_tool_returns_json_serialisable_output(image, ctx):
    calls = [
        ("inspect_image", {"image_path": str(image)}),
        ("inspect_metadata", {"image_path": str(image)}),
        ("run_localizer", {"image_path": str(image), "threshold": 0.5}),
        ("validate_prediction", {"image_path": str(image)}),
        ("generate_ela", {"image_path": str(image), "quality": 90}),
        ("compare_evidence", {}),
    ]

    for name, args in calls:
        result = run_tool(name, args, ctx)
        assert isinstance(result, dict), name
        assert "error" not in result, f"{name} failed: {result}"
        # json.dumps is the contract: no numpy, no PIL, no tuples.
        encoded = json.dumps(result)
        assert encoded
        assert not any(isinstance(value, np.generic) for value in result.values()), name


def test_floats_are_rounded_and_tuples_become_lists(image, ctx):
    run_tool("run_localizer", {"image_path": str(image)}, ctx)
    localizer = ctx.recall("run_localizer")

    assert localizer is not None
    assert isinstance(localizer["tampered_prob"], float)
    # 0.4321 already has 4 decimals; the tuple box must be a list of floats.
    assert localizer["tampered_prob"] == 0.4321
    assert isinstance(localizer["bbox"], list)
    assert all(isinstance(value, float) for value in localizer["bbox"])


def test_validate_prediction_reads_the_recorded_localizer_result(image, ctx):
    first = run_tool("validate_prediction", {"image_path": str(image)}, ctx)
    assert "error" in first
    assert "hint" in first

    run_tool("inspect_image", {"image_path": str(image)}, ctx)
    run_tool("run_localizer", {"image_path": str(image)}, ctx)
    result = run_tool("validate_prediction", {"image_path": str(image)}, ctx)

    assert "error" not in result
    assert result["valid"] is True
    assert result["issues"] == []


def test_compare_evidence_needs_recorded_evidence(image, ctx):
    empty = run_tool("compare_evidence", {}, ctx)
    assert "error" in empty and "hint" in empty

    run_tool("inspect_image", {"image_path": str(image)}, ctx)
    run_tool("run_localizer", {"image_path": str(image)}, ctx)
    run_tool("validate_prediction", {"image_path": str(image)}, ctx)
    result = run_tool("compare_evidence", {}, ctx)

    assert "error" not in result
    assert isinstance(result["agreements"], list)
    assert isinstance(result["conflicts"], list)


def test_missing_file_returns_an_error_with_a_hint(tmp_path, ctx):
    result = run_tool("inspect_image", {"image_path": str(tmp_path / "nope.png")}, ctx)

    assert "error" in result
    assert "hint" in result
    assert "not found" in result["error"]


def test_path_outside_the_upload_root_is_rejected(tmp_path, ctx):
    outside = tmp_path.parent / "outside.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(outside)
    try:
        result = run_tool("inspect_image", {"image_path": str(outside)}, ctx)
    finally:
        outside.unlink(missing_ok=True)

    assert "error" in result
    assert "hint" in result
    assert "outside the allowed upload directory" in result["error"]


def test_bad_path_is_an_error_dict_not_an_exception(ctx):
    # Empty string, None and a directory must all come back as dicts.
    for bad in ("", None, str(Path("/"))):
        result = run_tool("inspect_image", {"image_path": bad}, ctx)
        assert isinstance(result, dict)
        assert "error" in result
        assert "hint" in result


def test_unexpected_arguments_are_reported_not_raised(image, ctx):
    result = run_tool("inspect_image", {"image_path": str(image), "nonsense": 1}, ctx)

    assert "error" in result
    assert "hint" in result


def test_compare_evidence_never_requires_model_supplied_objects():
    # The model cannot pass a record or a result object: both tools that need
    # them take no such argument in their schema.
    by_name = {entry["function"]["name"]: entry for entry in TOOL_SCHEMAS}
    assert by_name["compare_evidence"]["function"]["parameters"]["properties"] == {}
    assert by_name["validate_prediction"]["function"]["parameters"]["properties"] == {
        "image_path": by_name["validate_prediction"]["function"]["parameters"]["properties"][
            "image_path"
        ]
    }
