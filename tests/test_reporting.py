"""Phase 7 tests for PDF report generation."""

from pathlib import Path

from agent.reporting import _escape_text, generate_pdf
from agent.schemas import InvestigationRecord, Observation, Verdict


def test_pdf_300_char_filename_script_escape():
    """PDF generates cleanly with a 300-character filename containing <script> and &."""

    # Create a record with a 300-char filename containing HTML special chars
    long_name = "A" * 290 + '<script>alert("xss")</script>' + "&special"
    # 300+ char filename with HTML special chars

    record = InvestigationRecord(
        investigation_id="test-filename",
        image_id="sha256test",
        filename=long_name,
        goal="quick_check",
        goal_text="",
        threshold=0.5,
        config_snapshot_hash="abc123",
        model_version="08b5f1cee3f94204f92ae77e13a3b466e8ceacc5739cd651af97597ce518add4",
        observations=[
            Observation(tool="inspect_image", args={"path": long_name}, status="ok",
                        result={"valid": True, "format": "PNG", "width": 64, "height": 64, "mode": "RGB", "byte_size": 5120, "sha256": "faksha"},
                        rationale="baseline", duration_ms=10.0),
            Observation(tool="run_localizer", args={"path": long_name, "threshold": 0.5}, status="ok",
                        result={"tampered_prob": 0.8, "label": "tampered", "bbox": [0.1, 0.1, 0.5, 0.5]},
                        rationale="baseline", duration_ms=50.0),
        ],
        errors=[],
        rationale="No issues found",
        verdict=Verdict.FLAGGED_FOR_REVIEW,
        verdict_reasons=["Score above threshold; validation valid"],
        created_at_utc="2026-01-15T10:30:00Z",
    )

    output_path = Path("/tmp/test_filename_report.pdf")
    generate_pdf(record, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0  # non-empty PDF

    # The filename must be escaped before it reaches the PDF markup:
    # no raw '<', '>' or unescaped '&' may survive.
    escaped_name = _escape_text(record.filename)
    assert "<" not in escaped_name and ">" not in escaped_name
    assert "&lt;script&gt;" in escaped_name
    assert "&amp;" in escaped_name

    # Clean up
    output_path.unlink()


def test_pdf_no_exif_reports_absent():
    """PDF reports 'absent' for no-EXIF image."""

    record = InvestigationRecord(
        investigation_id="test-no-exif",
        image_id="sha256test",
        filename="test_image.jpg",
        goal="quick_check",
        goal_text="",
        threshold=0.5,
        config_snapshot_hash="abc123",
        model_version="08b5f1cee3f94204f92ae77e13a3b466e8ceacc5739cd651af97597ce518add4",
        observations=[
            Observation(tool="inspect_image", args={"path": "test_image.jpg"}, status="ok",
                        result={"valid": True, "format": "PNG", "width": 64, "height": 64, "mode": "RGB", "byte_size": 5120, "sha256": "faksha"},
                        rationale="baseline", duration_ms=10.0),
            # No EXIF data in result
        ],
        errors=[],
        rationale="No issues found",
        verdict=Verdict.NOT_FLAGGED_BY_THIS_DETECTOR,
        verdict_reasons=["score below threshold; validation valid"],
        created_at_utc="2026-01-15T10:30:00Z",
    )

    output_path = Path("/tmp/test_no_exif_report.pdf")
    generate_pdf(record, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Clean up
    output_path.unlink()


def test_pdf_thumbnail_for_large_image():
    """PDF handles a 6000x4000 image producing a thumbnail."""

    # Create a record with large image dimensions from inspect_image
    record = InvestigationRecord(
        investigation_id="test-large-image",
        image_id="sha256test",
        filename="large_image.jpg",
        goal="quick_check",
        goal_text="",
        threshold=0.5,
        config_snapshot_hash="abc123",
        model_version="08b5f1cee3f94204f92ae77e13a3b466e8ceacc5739cd651af97597ce518add4",
        observations=[
            Observation(tool="inspect_image", args={"path": "large_image.jpg"}, status="ok",
                        result={"valid": True, "format": "PNG", "width": 6000, "height": 4000, "mode": "RGB", "byte_size": 5000000, "sha256": "faksha"},
                        rationale="baseline", duration_ms=10.0),
            Observation(tool="run_localizer", args={"path": "large_image.jpg", "threshold": 0.5}, status="ok",
                        result={"tampered_prob": 0.8, "label": "tampered", "bbox": [0.1, 0.1, 0.5, 0.5]},
                        rationale="baseline", duration_ms=50.0),
        ],
        errors=[],
        rationale="No issues found",
        verdict=Verdict.FLAGGED_FOR_REVIEW,
        verdict_reasons=["Score above threshold; validation valid"],
        created_at_utc="2026-01-15T10:30:00Z",
    )

    output_path = Path("/tmp/test_large_image_report.pdf")
    generate_pdf(record, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Clean up
    output_path.unlink()


def test_pdf_multi_page_record():
    """PDF for a multi-page record breaks cleanly."""

    # Create a record with multiple observations (enough to generate multiple pages)
    record = InvestigationRecord(
        investigation_id="test-multi-page",
        image_id="sha256test",
        filename="multi.jpg",
        goal="quick_check",
        goal_text="",
        threshold=0.5,
        config_snapshot_hash="abc123",
        model_version="08b5f1cee3f94204f92ae77e13a3b466e8ceacc5739cd651af97597ce518add4",
        observations=[
            Observation(tool="inspect_image", args={"path": "multi.jpg"}, status="ok",
                        result={"valid": True, "format": "PNG", "width": 64, "height": 64, "mode": "RGB", "byte_size": 5120, "sha256": "faksha"},
                        rationale="baseline", duration_ms=10.0),
            Observation(tool="run_localizer", args={"path": "multi.jpg", "threshold": 0.5}, status="ok",
                        result={"tampered_prob": 0.8, "label": "tampered", "bbox": [0.1, 0.1, 0.5, 0.5]},
                        rationale="baseline", duration_ms=50.0),
            Observation(tool="validate_prediction", args={"raw_score": 0.8, "bbox": [0.1, 0.1, 0.5, 0.5], "image_width": 64, "image_height": 64}, status="ok",
                        result={"valid": True, "issues": []},
                        rationale="baseline", duration_ms=5.0),
            Observation(tool="generate_ela", args={"path": "multi.jpg"}, status="ok",
                        result={"ela_valid": True, "ela_score": 0.15},
                        rationale="baseline", duration_ms=20.0),
            Observation(tool="inspect_metadata", args={"path": "multi.jpg"}, status="ok",
                        result={"camera": "DSLR", "flash": "fired"},
                        rationale="baseline", duration_ms=5.0),
        ],
        errors=[],
        rationale="No issues found",
        verdict=Verdict.FLAGGED_FOR_REVIEW,
        verdict_reasons=["Score above threshold; validation valid"],
        created_at_utc="2026-01-15T10:30:00Z",
    )

    output_path = Path("/tmp/test_multi_page_report.pdf")
    generate_pdf(record, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Clean up
    output_path.unlink()


def test_pdf_identical_record_same_text():
    """Identical record produces identical extracted text from PDF."""

    # Create the same record twice
    record = InvestigationRecord(
        investigation_id="test-identical",
        image_id="sha256test",
        filename="identical.jpg",
        goal="quick_check",
        goal_text="",
        threshold=0.5,
        config_snapshot_hash="abc123",
        model_version="08b5f1cee3f94204f92ae77e13a3b466e8ceacc5739cd651af97597ce518add4",
        observations=[
            Observation(tool="inspect_image", args={"path": "identical.jpg"}, status="ok",
                        result={"valid": True, "format": "PNG", "width": 64, "height": 64, "mode": "RGB", "byte_size": 5120, "sha256": "faksha"},
                        rationale="baseline", duration_ms=10.0),
            Observation(tool="run_localizer", args={"path": "identical.jpg", "threshold": 0.5}, status="ok",
                        result={"tampered_prob": 0.8, "label": "tampered", "bbox": [0.1, 0.1, 0.5, 0.5]},
                        rationale="baseline", duration_ms=50.0),
        ],
        errors=[],
        rationale="No issues found",
        verdict=Verdict.FLAGGED_FOR_REVIEW,
        verdict_reasons=["Score above threshold; validation valid"],
        created_at_utc="2026-01-15T10:30:00Z",
    )

    output1 = Path("/tmp/test_identical_1.pdf")
    output2 = Path("/tmp/test_identical_2.pdf")

    generate_pdf(record, output1)
    generate_pdf(record, output2)

    # Read and compare file sizes and modification times
    import os
    size1 = os.path.getsize(output1)
    size2 = os.path.getsize(output2)
    assert size1 == size2, f"PDF sizes differ: {size1} vs {size2}"

    # Clean up
    output1.unlink()
    output2.unlink()


def test_escape_text():
    """Test the _escape_text helper function."""
    from agent.reporting import _escape_text

    # HTML characters must be escaped to entities
    assert _escape_text("test <script>") == "test &lt;script&gt;"
    assert _escape_text("test > something") == "test &gt; something"
    assert _escape_text("test & more") == "test &amp; more"

    # A hostile filename with all three specials is fully escaped
    hostile = '<script>alert("xss")</script> & special'
    escaped = _escape_text(hostile)
    assert "<" not in escaped and ">" not in escaped
    assert "&lt;script&gt;" in escaped and "&amp;" in escaped

    # Empty and None
    assert _escape_text("") == ""
    assert _escape_text(None) == ""
