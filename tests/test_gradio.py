"""Phase 6 tests for Gradio UI rendering and verdict strings."""

from agent.schemas import InvestigationRecord, Observation

# We test the analyze function logic directly rather than launching the Gradio app
from agent.verdict import Verdict, decide


def test_verdict_no_authentic():
    """Verify that no verdict string contains 'authentic'."""
    all_verdicts = [
        Verdict.FLAGGED_FOR_REVIEW,
        Verdict.FLAGGED_REGION_UNAVAILABLE,
        Verdict.NOT_FLAGGED_BY_THIS_DETECTOR,
        Verdict.INCONCLUSIVE,
    ]
    for v in all_verdicts:
        assert "authentic" not in v.value.lower(), f"Verdict {v.value} contains 'authentic'"


def test_all_four_verdicts_present():
    """Verify all 4 allowed verdicts exist and are distinct."""
    values = [v.value for v in Verdict]
    expected = [
        "Flagged for review",
        "Flagged, region unavailable",
        "Not flagged by this detector",
        "Inconclusive",
    ]
    for e in expected:
        assert e in values, f"Expected verdict '{e}' not found in {values}"
    assert len(values) == 4, f"Expected exactly 4 verdicts, got {len(values)}"


def test_summary_contains_uncalibrated_label():
    """Test that summary contains 'tampered score (uncalibrated)' label."""
    from agent.summary import SummaryBuilder

    record = InvestigationRecord(
        investigation_id='test-002',
        image_id='sha256test',
        filename='test_image.jpg',
        goal='quick_check',
        goal_text='',
        threshold=0.5,
        config_snapshot_hash='abc123',
        model_version='08b5f1cee3f94204f92ae77e13a3b466e8ceacc5739cd651af97597ce518add4',
        observations=[
            Observation(tool='inspect_image', args={'path': 'test_image.jpg'}, status='ok',
                        result={'valid': True, 'format': 'PNG', 'width': 64, 'height': 64, 'mode': 'RGB', 'byte_size': 5120, 'sha256': 'faksha'},
                        rationale='baseline', duration_ms=10.0),
            Observation(tool='run_localizer', args={'path': 'test_image.jpg', 'threshold': 0.5}, status='ok',
                        result={'tampered_prob': 0.8, 'label': 'tampered', 'bbox': [0.1, 0.1, 0.5, 0.5]},
                        rationale='baseline', duration_ms=50.0),
            Observation(tool='validate_prediction', args={'raw_score': 0.8, 'bbox': [0.1, 0.1, 0.5, 0.5], 'image_width': 64, 'image_height': 64}, status='ok',
                        result={'valid': True, 'issues': []},
                        rationale='baseline', duration_ms=5.0),
        ],
        errors=[],
        rationale='No issues found',
        verdict=Verdict.FLAGGED_FOR_REVIEW,
        verdict_reasons=['Score above threshold; validation valid'],
        created_at_utc='2026-01-15T10:30:00Z',
    )

    builder = SummaryBuilder(record)
    summary = builder.build()

    # Should contain the uncalibrated label
    assert "tampered score (uncalibrated)" in summary.lower() or "Tampered score (uncalibrated)" in summary, \
        f"Summary should contain 'tampered score (uncalibrated)', got: {summary}"
    assert builder.check_numbers_against_record()


def test_summary_numeric_parser_rejects_unstored_number():
    from agent.summary import SummaryBuilder

    record = InvestigationRecord(
        investigation_id="numeric-check", image_id="id", filename="x.png", goal="quick_check",
        goal_text="", threshold=0.5, config_snapshot_hash="", model_version="",
        observations=[Observation("run_localizer", {}, "ok",
                                  {"tampered_prob": 0.8, "bbox": [0.1, 0.1, 0.5, 0.5]})],
        errors=[], rationale="", verdict=Verdict.FLAGGED_FOR_REVIEW,
        created_at_utc="2026-01-01T00:00:00Z")
    builder = SummaryBuilder(record)
    assert builder.check_numbers_against_record()
    builder.build = lambda: "Score 0.8 threshold 0.5 with stray 999.0"
    assert not builder.check_numbers_against_record()


def test_record_renderer_escapes_untrusted_content_and_shows_verdict():
    from agent.rendering import render_record

    record = InvestigationRecord(
        investigation_id="render", image_id="id", filename='<script>alert("x")</script>&.png',
        goal="origin", goal_text="camera <script>&", threshold=0.5, config_snapshot_hash="",
        model_version="", observations=[Observation("run_localizer", {}, "ok",
                                                      {"tampered_prob": 0.75})],
        errors=[], rationale="", verdict=Verdict.FLAGGED_FOR_REVIEW,
        created_at_utc="2026-01-01T00:00:00Z")
    output = render_record(record)
    assert "Flagged for review" in output
    assert "tampered score (uncalibrated)" in output
    assert "authentic" not in output.lower()
    assert "&lt;script&gt;" in output and "&amp;" in output
    assert "<script>" not in output


def test_escaping_handles_script_tags():
    """The app's own untrusted-string escaping handles script tags and ampersands."""
    from agent.rendering import escape_untrusted

    # Test with script tag: '<' and '>' must be escaped, not passed through
    dangerous = '<script>alert("xss")</script>'
    escaped = escape_untrusted(dangerous)
    assert "<script>" not in escaped, f"Script tag not escaped: {escaped}"
    assert "&lt;script&gt;" in escaped, f"Expected entities in: {escaped}"

    # Test with ampersand
    ampersand = "test & more"
    escaped_amp = escape_untrusted(ampersand)
    assert "&amp;" in escaped_amp, f"Ampersand not escaped: {escaped_amp}"
    assert escaped_amp == "test &amp; more"


def test_verdict_decision_scenarios():
    """Test the decide function with different scenarios to verify all 4 verdicts."""
    from agent.schemas import InvestigationRecord

    # Scenario 1: score above threshold, validation valid -> Flagged for review
    record1 = InvestigationRecord(
        investigation_id='s1',
        image_id='i1',
        filename='img1.jpg',
        goal='quick_check',
        goal_text='',
        threshold=0.5,
        config_snapshot_hash='',
        model_version='',
        observations=[
            Observation(tool='inspect_image', args={'path': 'img1.jpg'}, status='ok',
                        result={'valid': True}, rationale='baseline', duration_ms=10.0),
            Observation(tool='run_localizer', args={'path': 'img1.jpg', 'threshold': 0.5}, status='ok',
                        result={'tampered_prob': 0.8, 'label': 'tampered', 'bbox': [0.1, 0.1, 0.5, 0.5]},
                        rationale='baseline', duration_ms=50.0),
            Observation(tool='validate_prediction', args={'raw_score': 0.8, 'bbox': [0.1, 0.1, 0.5, 0.5], 'image_width': 64, 'image_height': 64}, status='ok',
                        result={'valid': True, 'issues': []}, rationale='baseline', duration_ms=5.0),
        ],
        errors=[],
        rationale='',
        verdict=Verdict.INCONCLUSIVE,
        verdict_reasons=[],
        created_at_utc='',
    )
    v1, r1 = decide(record1)
    assert v1 == Verdict.FLAGGED_FOR_REVIEW, f"Expected FLAGGED_FOR_REVIEW, got {v1}"

    # Scenario 2: score below threshold, validation valid -> Not flagged
    record2 = InvestigationRecord(
        investigation_id='s2',
        image_id='i2',
        filename='img2.jpg',
        goal='quick_check',
        goal_text='',
        threshold=0.5,
        config_snapshot_hash='',
        model_version='',
        observations=[
            Observation(tool='inspect_image', args={'path': 'img2.jpg'}, status='ok',
                        result={'valid': True}, rationale='baseline', duration_ms=10.0),
            Observation(tool='run_localizer', args={'path': 'img2.jpg', 'threshold': 0.5}, status='ok',
                        result={'tampered_prob': 0.3, 'label': 'authentic', 'bbox': None},
                        rationale='baseline', duration_ms=50.0),
            Observation(tool='validate_prediction', args={'raw_score': 0.3, 'bbox': None, 'image_width': 64, 'image_height': 64}, status='ok',
                        result={'valid': True, 'issues': []}, rationale='baseline', duration_ms=5.0),
        ],
        errors=[],
        rationale='',
        verdict=Verdict.INCONCLUSIVE,
        verdict_reasons=[],
        created_at_utc='',
    )
    v2, r2 = decide(record2)
    assert v2 == Verdict.NOT_FLAGGED_BY_THIS_DETECTOR, f"Expected NOT_FLAGGED_BY_THIS_DETECTOR, got {v2}"

    # Scenario 3: score above threshold, validation invalid -> Flagged, region unavailable
    record3 = InvestigationRecord(
        investigation_id='s3',
        image_id='i3',
        filename='img3.jpg',
        goal='quick_check',
        goal_text='',
        threshold=0.5,
        config_snapshot_hash='',
        model_version='',
        observations=[
            Observation(tool='inspect_image', args={'path': 'img3.jpg'}, status='ok',
                        result={'valid': True}, rationale='baseline', duration_ms=10.0),
            Observation(tool='run_localizer', args={'path': 'img3.jpg', 'threshold': 0.5}, status='ok',
                        result={'tampered_prob': 0.8, 'label': 'tampered', 'bbox': [0.5, 0.1, 0.1, 0.5]},  # inverted bbox
                        rationale='baseline', duration_ms=50.0),
            Observation(tool='validate_prediction', args={'raw_score': 0.8, 'bbox': [0.5, 0.1, 0.1, 0.5], 'image_width': 64, 'image_height': 64}, status='ok',
                        result={'valid': False, 'issues': ['inverted box']}, rationale='baseline', duration_ms=5.0),
        ],
        errors=[],
        rationale='',
        verdict=Verdict.INCONCLUSIVE,
        verdict_reasons=[],
        created_at_utc='',
    )
    v3, r3 = decide(record3)
    assert v3 == Verdict.FLAGGED_REGION_UNAVAILABLE, f"Expected FLAGGED_REGION_UNAVAILABLE, got {v3}"

    # Scenario 4: error in required check -> Inconclusive
    record4 = InvestigationRecord(
        investigation_id='s4',
        image_id='i4',
        filename='img4.jpg',
        goal='quick_check',
        goal_text='',
        threshold=0.5,
        config_snapshot_hash='',
        model_version='',
        observations=[
            Observation(tool='inspect_image', args={'path': 'img4.jpg'}, status='error',
                        result={}, rationale='baseline', duration_ms=10.0),
        ],
        errors=['inspect_image error'],
        rationale='',
        verdict=Verdict.INCONCLUSIVE,
        verdict_reasons=[],
        created_at_utc='',
    )
    v4, r4 = decide(record4)
    assert v4 == Verdict.INCONCLUSIVE, f"Expected INCONCLUSIVE, got {v4}"

    print('All Phase 6 tests PASSED')
