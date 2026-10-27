"""Phase 7 tests for PDF report generation."""

from PIL import Image as PILImage

from agent.reporting import _escape_text, generate_batch_pdf, generate_pdf
from agent.schemas import InvestigationRecord, Observation, Verdict


def test_pdf_300_char_filename_script_escape(tmp_path):
    """PDF generates cleanly with a 300-character filename containing <script> and &."""

    # Create a record with a 300-char filename containing HTML special chars
    hostile_suffix = "<script>&</script>"
    long_name = "A" * (300 - len(hostile_suffix)) + hostile_suffix
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
            Observation(
                tool="inspect_image",
                args={"path": long_name},
                status="ok",
                result={
                    "valid": True,
                    "format": "PNG",
                    "width": 64,
                    "height": 64,
                    "mode": "RGB",
                    "byte_size": 5120,
                    "sha256": "faksha",
                },
                rationale="baseline",
                duration_ms=10.0,
            ),
            Observation(
                tool="run_localizer",
                args={"path": long_name, "threshold": 0.5},
                status="ok",
                result={"tampered_prob": 0.8, "label": "tampered", "bbox": [0.1, 0.1, 0.5, 0.5]},
                rationale="baseline",
                duration_ms=50.0,
            ),
        ],
        errors=[],
        rationale="No issues found",
        verdict=Verdict.FLAGGED_FOR_REVIEW,
        verdict_reasons=["Score above threshold; validation valid"],
        created_at_utc="2026-01-15T10:30:00Z",
    )

    output_path = tmp_path / "filename_report.pdf"
    generate_pdf(record, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0  # non-empty PDF

    # The filename must be escaped before it reaches the PDF markup:
    # no raw '<', '>' or unescaped '&' may survive.
    escaped_name = _escape_text(record.filename)
    assert "<" not in escaped_name and ">" not in escaped_name
    assert "&lt;script&gt;" in escaped_name
    assert "&amp;" in escaped_name

    from pypdf import PdfReader

    extracted = "\n".join(page.extract_text() or "" for page in PdfReader(output_path).pages)
    assert "Image:" in extracted
    assert "&lt;script&gt;&amp;&lt;/script&gt;" in extracted


def test_pdf_no_exif_reports_absent(tmp_path):
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
            Observation(
                tool="inspect_image",
                args={"path": "test_image.jpg"},
                status="ok",
                result={
                    "valid": True,
                    "format": "PNG",
                    "width": 64,
                    "height": 64,
                    "mode": "RGB",
                    "byte_size": 5120,
                    "sha256": "faksha",
                },
                rationale="baseline",
                duration_ms=10.0,
            ),
            Observation(
                tool="inspect_metadata",
                args={"path": "test_image.jpg"},
                status="ok",
                result={"exif_status": "absent", "present_fields": [], "absent_fields": ["Make"]},
            ),
        ],
        errors=[],
        rationale="No issues found",
        verdict=Verdict.NOT_FLAGGED_BY_THIS_DETECTOR,
        verdict_reasons=["score below threshold; validation valid"],
        created_at_utc="2026-01-15T10:30:00Z",
    )

    output_path = tmp_path / "no_exif_report.pdf"
    generate_pdf(record, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    from pypdf import PdfReader

    extracted = "\n".join(page.extract_text() or "" for page in PdfReader(output_path).pages)
    assert "EXIF: absent" in extracted


def test_pdf_thumbnail_for_large_image(tmp_path):
    """PDF handles a 6000x4000 image producing a thumbnail."""

    # Create a record with large image dimensions from inspect_image
    image_path = tmp_path / "large_image.jpg"
    PILImage.new("RGB", (6000, 4000), "blue").save(image_path, quality=80)
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
            Observation(
                tool="inspect_image",
                args={"path": "large_image.jpg"},
                status="ok",
                result={
                    "valid": True,
                    "format": "PNG",
                    "width": 6000,
                    "height": 4000,
                    "mode": "RGB",
                    "byte_size": 5000000,
                    "sha256": "faksha",
                },
                rationale="baseline",
                duration_ms=10.0,
            ),
            Observation(
                tool="run_localizer",
                args={"path": "large_image.jpg", "threshold": 0.5},
                status="ok",
                result={"tampered_prob": 0.8, "label": "tampered", "bbox": [0.1, 0.1, 0.5, 0.5]},
                rationale="baseline",
                duration_ms=50.0,
            ),
        ],
        errors=[],
        rationale="No issues found",
        verdict=Verdict.FLAGGED_FOR_REVIEW,
        verdict_reasons=["Score above threshold; validation valid"],
        created_at_utc="2026-01-15T10:30:00Z",
        image_path=str(image_path),
        sha256="large-image-sha",
        config_snapshot={
            "forensics": {"report_asset_roots": [str(tmp_path)], "report_limitations": []}
        },
    )

    output_path = tmp_path / "large_image_report.pdf"
    generate_pdf(record, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    from pypdf import PdfReader

    reader = PdfReader(output_path)
    assert any(page.get("/Resources", {}).get("/XObject") for page in reader.pages)
    assert "Image thumbnail" in "\n".join(page.extract_text() or "" for page in reader.pages)


def test_pdf_multi_page_record(tmp_path):
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
            Observation(
                tool="inspect_image",
                args={"path": "multi.jpg"},
                status="ok",
                result={
                    "valid": True,
                    "format": "PNG",
                    "width": 64,
                    "height": 64,
                    "mode": "RGB",
                    "byte_size": 5120,
                    "sha256": "faksha",
                },
                rationale="baseline",
                duration_ms=10.0,
            ),
            Observation(
                tool="run_localizer",
                args={"path": "multi.jpg", "threshold": 0.5},
                status="ok",
                result={"tampered_prob": 0.8, "label": "tampered", "bbox": [0.1, 0.1, 0.5, 0.5]},
                rationale="baseline",
                duration_ms=50.0,
            ),
            Observation(
                tool="validate_prediction",
                args={
                    "raw_score": 0.8,
                    "bbox": [0.1, 0.1, 0.5, 0.5],
                    "image_width": 64,
                    "image_height": 64,
                },
                status="ok",
                result={"valid": True, "issues": []},
                rationale="baseline",
                duration_ms=5.0,
            ),
            Observation(
                tool="generate_ela",
                args={"path": "multi.jpg"},
                status="ok",
                result={"ela_valid": True, "ela_score": 0.15},
                rationale="baseline",
                duration_ms=20.0,
            ),
            Observation(
                tool="inspect_metadata",
                args={"path": "multi.jpg"},
                status="ok",
                result={"camera": "DSLR", "flash": "fired"},
                rationale="baseline",
                duration_ms=5.0,
            ),
        ],
        errors=[],
        rationale="No issues found",
        verdict=Verdict.FLAGGED_FOR_REVIEW,
        verdict_reasons=["Score above threshold; validation valid"],
        created_at_utc="2026-01-15T10:30:00Z",
    )

    output_path = tmp_path / "multi_page_report.pdf"
    generate_pdf(record, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    from pypdf import PdfReader

    assert len(PdfReader(output_path).pages) >= 4


def test_pdf_identical_record_same_text(tmp_path):
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
            Observation(
                tool="inspect_image",
                args={"path": "identical.jpg"},
                status="ok",
                result={
                    "valid": True,
                    "format": "PNG",
                    "width": 64,
                    "height": 64,
                    "mode": "RGB",
                    "byte_size": 5120,
                    "sha256": "faksha",
                },
                rationale="baseline",
                duration_ms=10.0,
            ),
            Observation(
                tool="run_localizer",
                args={"path": "identical.jpg", "threshold": 0.5},
                status="ok",
                result={"tampered_prob": 0.8, "label": "tampered", "bbox": [0.1, 0.1, 0.5, 0.5]},
                rationale="baseline",
                duration_ms=50.0,
            ),
        ],
        errors=[],
        rationale="No issues found",
        verdict=Verdict.FLAGGED_FOR_REVIEW,
        verdict_reasons=["Score above threshold; validation valid"],
        created_at_utc="2026-01-15T10:30:00Z",
    )

    output1 = tmp_path / "identical_1.pdf"
    output2 = tmp_path / "identical_2.pdf"

    generate_pdf(record, output1)
    generate_pdf(record, output2)

    from pypdf import PdfReader

    text1 = "\n".join(page.extract_text() or "" for page in PdfReader(output1).pages)
    text2 = "\n".join(page.extract_text() or "" for page in PdfReader(output2).pages)
    assert text1 == text2


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


def test_pdf_extracted_text_escaping_absent_exif_and_repeatability(tmp_path):
    """Extracted PDF text preserves the hostile filename, absent EXIF, and repeatability."""
    from pypdf import PdfReader

    filename = "sample_<script>alert(1)</script>&record.jpg"
    record = InvestigationRecord(
        investigation_id="extract-text-check",
        image_id="synthetic-image",
        filename=filename,
        goal="quick_check",
        goal_text="",
        threshold=0.5,
        config_snapshot_hash="fixed-hash",
        model_version="stub",
        observations=[
            Observation(
                tool="inspect_image",
                args={"path": filename},
                status="ok",
                result={
                    "valid": True,
                    "format": "JPEG",
                    "width": 64,
                    "height": 64,
                    "mode": "RGB",
                    "exif": {},
                },
            ),
            Observation(
                tool="run_localizer",
                args={"path": filename},
                status="ok",
                result={"tampered_prob": 0.25, "bbox": None},
            ),
            Observation(
                tool="inspect_metadata",
                args={"path": filename},
                status="ok",
                result={"exif_status": "absent", "present_fields": [], "absent_fields": ["Make"]},
            ),
        ],
        errors=[],
        rationale="",
        verdict=Verdict.NOT_FLAGGED_BY_THIS_DETECTOR,
        verdict_reasons=["score below threshold"],
        created_at_utc="2026-01-01T00:00:00Z",
    )

    first_pdf = tmp_path / "first.pdf"
    second_pdf = tmp_path / "second.pdf"
    generate_pdf(record, first_pdf)
    first_text = "\n".join(page.extract_text() or "" for page in PdfReader(first_pdf).pages)
    generate_pdf(record, second_pdf)
    second_text = "\n".join(page.extract_text() or "" for page in PdfReader(second_pdf).pages)

    assert "Image: sample_&lt;script&gt;alert(1)&lt;/script&gt;&amp;record.jpg" in first_text
    assert "EXIF: absent" in first_text
    assert first_text == second_text


def test_batch_pdf_contains_required_summary_columns(tmp_path):
    from pypdf import PdfReader

    record = InvestigationRecord(
        investigation_id="batch-row",
        image_id="image-id",
        filename="splice.png",
        goal="quick_check",
        goal_text="",
        threshold=0.5,
        config_snapshot_hash="",
        model_version="stub",
        observations=[Observation("run_localizer", {}, "ok", {"tampered_prob": 0.75})],
        errors=[],
        rationale="",
        verdict=Verdict.FLAGGED_FOR_REVIEW,
    )
    output = tmp_path / "batch-summary.pdf"
    generate_batch_pdf([record], output)
    text = "\n".join(page.extract_text() or "" for page in PdfReader(output).pages)
    for column in ("Filename", "Status", "Raw score", "Verdict category"):
        assert column in text
    assert "splice.png" in text and "0.75" in text and "Flagged for review" in text


def test_pdf_metadata_parse_error_is_not_reported_as_absent(tmp_path):
    from pypdf import PdfReader

    record = InvestigationRecord(
        investigation_id="metadata-error",
        image_id="image-id",
        filename="metadata.png",
        goal="quick_check",
        goal_text="",
        threshold=0.5,
        config_snapshot_hash="",
        model_version="stub",
        observations=[
            Observation(
                "inspect_metadata",
                {},
                "ok",
                {
                    "exif_status": "parse_error",
                    "present_fields": [],
                    "absent_fields": [],
                    "parse_errors": ["bad EXIF block"],
                },
            ),
        ],
        errors=[],
        rationale="",
        verdict=Verdict.INCONCLUSIVE,
    )
    output = tmp_path / "metadata-error.pdf"
    generate_pdf(record, output)
    text = "\n".join(page.extract_text() or "" for page in PdfReader(output).pages)
    assert "Metadata: parse error" in text
    assert "EXIF: absent" not in text
