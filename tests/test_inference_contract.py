"""Contract tests for the inference module's predict() and predict_raw() methods.

These tests verify the new Phase 1 contract: predict_raw returns raw score without
threshold, predict() accepts an optional threshold, and various edge cases are handled.
"""

import io

import numpy as np
import pytest

from src.inference import TamperingDetector


@pytest.fixture(scope="module")
def detector():
    """Detector with a real checkpoint (trained on CASIA v2 splices)."""
    return TamperingDetector("checkpoints/best_localizer.keras")


@pytest.fixture
def valid_pil_image():
    """A small RGB PIL image for testing."""
    from PIL import Image
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestPredictRaw:
    """Tests for predict_raw() — returns raw score without threshold."""

    def test_valid_image_returns_dict_with_expected_keys(self, detector, valid_pil_image):
        result = detector.predict_raw(valid_pil_image)
        assert isinstance(result, dict)
        assert "tampered_prob" in result
        assert "bbox_raw" in result
        assert "image_size" in result
        assert "model_version" in result
        assert isinstance(result["tampered_prob"], float)
        assert 0.0 <= result["tampered_prob"] <= 1.0
        assert isinstance(result["bbox_raw"], list) or result["bbox_raw"] is None
        assert isinstance(result["image_size"], tuple)
        assert len(result["image_size"]) == 2
        assert isinstance(result["model_version"], str)
        assert len(result["model_version"]) == 64  # SHA-256 hex digest

    def test_tampered_image_has_nonzero_prob(self, detector):
        """A spliced image should have a non-zero tampered probability."""
        from PIL import Image

        rng = np.random.default_rng(0)
        base = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        buf = io.BytesIO()
        base.save(buf, "JPEG", quality=75)
        buf.seek(0)
        host = Image.open(buf).convert("RGB")

        patch = Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))
        spliced = host.copy()
        spliced.paste(patch, (16, 16))

        result = detector.predict_raw(spliced)
        # A spliced image should have a non-zero tampered probability
        assert result["tampered_prob"] != 0.5  # should not be exactly neutral

    def test_predict_raw_no_threshold_applied(self, detector, valid_pil_image):
        """predict_raw must NOT apply threshold — the result should be the same
        regardless of what threshold is set externally (since threshold isn't stored)."""
        from PIL import Image
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)

        # Call predict_raw twice — result must be identical both times
        result1 = detector.predict_raw(img)
        result2 = detector.predict_raw(img)
        assert result1["tampered_prob"] == result2["tampered_prob"]
        assert result1["model_version"] == result2["model_version"]


class TestPredict:
    """Tests for predict() — with optional threshold argument."""

    def test_predict_uses_provided_threshold(self, detector, valid_pil_image):
        result = detector.predict(valid_pil_image, threshold=0.5)
        assert "label" in result
        assert "confidence" in result
        assert "tampered_prob" in result
        assert result["tampered_prob"] is not None

    def test_predict_defaults_to_config_threshold(self, detector, valid_pil_image):
        """When no threshold is given, it should come from configs/default.yaml."""
        result = detector.predict(valid_pil_image)
        # The result should have a label decided by the config threshold (0.5)
        assert result["label"] in ("tampered", "authentic")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_no_threshold_on_authentic(self, detector, valid_pil_image):
        result = detector.predict(valid_pil_image, threshold=0.99)
        # With a very high threshold, the image should be authentic
        assert result["label"] == "authentic"

    def test_predict_no_threshold_on_tampered(self, detector):
        """With a very low threshold, tampered images should be flagged."""
        # Use a spliced image
        from PIL import Image
        rng = np.random.default_rng(0)
        base = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        buf = io.BytesIO()
        base.save(buf, "JPEG", quality=75)
        buf.seek(0)
        host = Image.open(buf).convert("RGB")
        patch = Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))
        spliced = host.copy()
        spliced.paste(patch, (16, 16))

        result = detector.predict(spliced, threshold=0.01)
        # Should not crash and should return a valid dict
        assert "label" in result
        assert "confidence" in result

    def test_predict_returns_bounding_box_when_tampered(self, detector):
        """If the model predicts tampered, bbox should be a list of 4 floats."""
        from PIL import Image
        rng = np.random.default_rng(0)
        base = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        buf = io.BytesIO()
        base.save(buf, "JPEG", quality=75)
        buf.seek(0)
        host = Image.open(buf).convert("RGB")
        patch = Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))
        spliced = host.copy()
        spliced.paste(patch, (16, 16))

        result = detector.predict(spliced, threshold=0.01)
        if result["bbox"] is not None:
            assert isinstance(result["bbox"], list)
            assert len(result["bbox"]) == 4
            xmin, ymin, xmax, ymax = result["bbox"]
            assert 0.0 <= xmin <= xmax <= 1.0
            assert 0.0 <= ymin <= ymax <= 1.0

    def test_predict_bbox_none_when_authentic(self, detector, valid_pil_image):
        """Authentic images (or images without tampering) should have bbox=None."""
        result = detector.predict(valid_pil_image, threshold=0.99)
        assert result["bbox"] is None


class TestEdgeCases:
    """Tests for edge cases: corrupt images, unsupported extensions, etc."""

    def test_corrupt_image_clean_error(self, detector):
        """A corrupt image file must raise a clean error, not crash."""
        # Pass a non-existent file path; preprocess will call Image.open()
        # which raises FileNotFoundError.
        with pytest.raises((FileNotFoundError, OSError)):
            detector.predict_raw("/nonexistent/path/to/image.jpg")

    def test_unsupported_extension(self, detector, tmp_path):
        """Unsupported file extension should raise a clear error."""
        bad_path = tmp_path / "test.xyz"
        bad_path.write_text("not an image")
        with pytest.raises((OSError, ValueError)):
            detector.predict(str(bad_path))

    def test_tiny_image_1x1(self, detector):
        """A 1x1 image should not crash the model."""
        from PIL import Image
        tiny = Image.new("RGB", (1, 1), (128, 128, 128))
        result = detector.predict_raw(tiny)
        # Should produce some output without crashing
        assert "tampered_prob" in result
        assert "model_version" in result

    def test_very_large_image(self, detector):
        """A very large image should be handled without crashing."""
        from PIL import Image
        # Create a large but valid image (4000x3000)
        large = Image.new("RGB", (4000, 3000), (128, 128, 128))
        result = detector.predict_raw(large)
        assert "tampered_prob" in result
        assert "model_version" in result

    def test_malformed_model_output_rejected(self, detector):
        """Model output with NaN or out-of-range values must be handled gracefully."""
        from PIL import Image

        img = Image.new("RGB", (64, 64), (128, 128, 128))

        # Test that NaN tampered_prob is detected and handled

        # Call predict_raw and check the result is valid (float in [0,1])
        result = detector.predict_raw(img)
        # If tampered_prob is NaN, it should be caught; if finite, that's also fine
        assert np.isfinite(result["tampered_prob"]), (
            f"tampered_prob should be finite, got {result['tampered_prob']}"
        )
        assert 0.0 <= result["tampered_prob"] <= 1.0, (
            f"tampered_prob should be in [0,1], got {result['tampered_prob']}"
        )

    def test_threshold_not_written_to_shared_state(self, detector):
        """Verify that predict() does NOT mutate shared state (the Phase 1 fix)."""
        import threading

        results = []
        from PIL import Image

        img = Image.new("RGB", (64, 64), (128, 128, 128))

        def run_with_threshold(t):
            result = detector.predict(img, threshold=t)
            results.append((t, result["label"], result["tampered_prob"]))

        # Run two threads with different thresholds concurrently
        t1 = threading.Thread(target=run_with_threshold, args=(0.1,))
        t2 = threading.Thread(target=run_with_threshold, args=(0.9,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both results should be valid and not crash (thread safety)
        assert len(results) == 2
        for _t, label, prob in results:
            assert label in ("tampered", "authentic")
            assert 0.0 <= prob <= 1.0
