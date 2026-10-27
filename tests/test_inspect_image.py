"""Unit tests for inspect_image tool."""

import os
import tempfile
from pathlib import Path

from agent.tools.inspect_image import inspect_image

PNG_PATH = "data/CASIA2/CASIA 2 Groundtruth/Tp_D_NNN_S_N_arc00071_ind00001_00421_gt.png"
SHORT_PNG_PATH = PNG_PATH


class TestInspectImageSuccess:
    """Tests for successful inspection cases."""

    def test_valid_png_image(self):
        """Valid PNG image should return proper metadata."""
        result = inspect_image(PNG_PATH)
        assert result.valid is True
        assert result.format is not None
        assert result.width > 0
        assert result.height > 0
        assert result.sha256 != ""
        assert len(result.sha256) == 64

    def test_sha256_is_64_chars(self):
        """SHA-256 should be a 64-character hex string."""
        result = inspect_image(PNG_PATH)
        assert len(result.sha256) == 64


class TestInspectImageFailure:
    """Tests for failure cases."""

    def test_nonexistent_file(self):
        """Nonexistent file should return invalid result."""
        result = inspect_image("/nonexistent/path/to/image.png")
        assert result.valid is False
        assert len(result.issues) > 0

    def test_empty_string_path(self):
        """Empty string path should return invalid result."""
        result = inspect_image("")
        assert result.valid is False


class TestInspectImageWithPath:
    """Tests with pathlib.Path objects."""

    def test_with_path_object(self):
        """Test with pathlib.Path object."""
        p = Path(PNG_PATH)
        result = inspect_image(str(p))
        assert result.valid is True

    def test_path_object_direct(self):
        """Test passing Path object directly (convert to str)."""
        from pathlib import Path
        p = Path(PNG_PATH)
        result = inspect_image(p)
        assert result.valid is True


class TestInspectImageEdgeCases:
    """Tests for edge cases."""

    def test_small_image_no_crash(self):
        """Tiny 1x1 image should not crash the tool."""
        from PIL import Image
        tiny = Image.new("RGB", (1, 1), (128, 128, 128))
        # Save to a temp file and get the path
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            try:
                tiny.save(f.name, "PNG")
                result = inspect_image(f.name)
                # Should not crash; valid may be True or False depending on implementation
                assert result is not None
            finally:
                os.unlink(f.name)

    def test_digest_matches_file_bytes(self, tmp_path):
        """Inspection hashes the bytes actually read from disk."""
        import hashlib

        from PIL import Image

        image_path = tmp_path / "sample.png"
        Image.new("RGB", (3, 2), (12, 34, 56)).save(image_path)
        result = inspect_image(image_path)
        assert result.sha256 == hashlib.sha256(image_path.read_bytes()).hexdigest()
