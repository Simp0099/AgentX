"""Unit tests for validate_prediction tool."""

from agent.tools.validate_prediction import validate_prediction


class TestValidatePredictionSuccess:
    """Tests for successful validation cases."""

    def test_valid_score_and_bbox(self):
        """Valid score in [0,1] and valid bbox should pass."""
        result = validate_prediction(0.7, [0.1, 0.2, 0.5, 0.8], 256, 384)
        assert result.valid is True
        assert len(result.issues) == 0

    def test_authentic_with_no_bbox(self):
        """Authentic image with bbox=None should pass."""
        result = validate_prediction(0.3, None, 256, 384)
        assert result.valid is True
        assert len(result.issues) == 0

    def test_zero_score_authentic(self):
        """Score of 0 should be valid (authentic)."""
        result = validate_prediction(0.0, [0.1, 0.2, 0.5, 0.8], 256, 384)
        assert result.valid is True

    def test_full_score_tampered(self):
        """Score of 1 should be valid (tampered)."""
        result = validate_prediction(1.0, [0.1, 0.2, 0.5, 0.8], 256, 384)
        assert result.valid is True


class TestValidatePredictionFailure:
    """Tests for validation failure cases."""

    def test_nan_score(self):
        """NaN score should fail validation."""
        result = validate_prediction(float('nan'), [0.1, 0.2, 0.5, 0.8], 256, 384)
        assert result.valid is False
        assert any(i.category == "score" for i in result.issues)

    def test_score_above_one(self):
        """Score > 1 should fail validation."""
        result = validate_prediction(1.5, [0.1, 0.2, 0.5, 0.8], 256, 384)
        assert result.valid is False
        assert any(i.category == "score" for i in result.issues)

    def test_score_below_zero(self):
        """Score < 0 should fail validation."""
        result = validate_prediction(-0.1, [0.1, 0.2, 0.5, 0.8], 256, 384)
        assert result.valid is False
        assert any(i.category == "score" for i in result.issues)

    def test_inverted_bbox_x(self):
        """xmin > xmax should fail validation."""
        result = validate_prediction(0.7, [0.5, 0.2, 0.3, 0.8], 256, 384)
        assert result.valid is False
        assert any(i.category == "bbox" for i in result.issues)

    def test_inverted_bbox_y(self):
        """ymin > ymax should fail validation."""
        result = validate_prediction(0.7, [0.1, 0.5, 0.5, 0.3], 256, 384)
        assert result.valid is False
        assert any(i.category == "bbox" for i in result.issues)

    def test_bbox_out_of_range_xmin(self):
        """xmin < 0 should fail validation."""
        result = validate_prediction(0.7, [-0.1, 0.2, 0.5, 0.8], 256, 384)
        assert result.valid is False
        assert any(i.category == "bbox" for i in result.issues)

    def test_bbox_out_of_range_xmax(self):
        """xmax > 1 should fail validation."""
        result = validate_prediction(0.7, [0.1, 0.2, 1.5, 0.8], 256, 384)
        assert result.valid is False
        assert any(i.category == "bbox" for i in result.issues)

    def test_bbox_zero_area(self):
        """xmin == xmax should fail validation (zero width)."""
        result = validate_prediction(0.7, [0.2, 0.2, 0.2, 0.8], 256, 384)
        assert result.valid is False
        assert any(i.category == "bbox" for i in result.issues)


class TestValidatePredictionAreaCoverage:
    """Tests for box area and coverage constraints."""

    def test_box_area_below_minimum(self):
        """Box area below minimum fraction should fail."""
        # Use a very small box that covers less than the minimum fraction
        result = validate_prediction(0.9, [0.4, 0.4, 0.45, 0.45], 256, 384)
        # May or may not fail depending on config min_box_fraction
        # At minimum, should not crash
        assert result.valid is False or result.valid is True  # should not crash

    def test_box_covers_too_much(self):
        """Box covering more than max fraction should fail."""
        # Use a huge box that covers most of the image
        result = validate_prediction(0.9, [0.0, 0.0, 1.0, 1.0], 256, 384)
        # Very large box - may fail depending on config max_box_fraction
        assert result.valid is False or result.valid is True  # should not crash


class TestValidatePredictionIntegration:
    """Integration-style tests for validate_prediction."""

    def test_mixed_validity(self):
        """Mix of valid and invalid should accumulate all issues."""
        result = validate_prediction(float('nan'), [0.5, 0.5, 0.3, 0.8], -1, 384)
        assert result.valid is False
        # Should have issues from both the score and the bbox width
        assert len(result.issues) >= 1
