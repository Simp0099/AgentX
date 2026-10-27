"""Phase 8 tests for batch/ZIP processing with security checks."""

import os
import tempfile

from agent.batch import (
    BatchStatus,
    _make_zip_with_member,
    process_batch,
    process_mixed_batch,
)


def test_zip_slip_detection():
    """ZIP-slip paths are detected and batch fails."""
    zip_bytes, member_filenames = _make_zip_with_member(
        [
            ("safe.jpg", b"\x89PNG\r\n\x1a\n"),
            ("..%2f..%2fescape.txt", b"malicious"),
        ]
    )

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
        tmp.write(zip_bytes)

    try:
        result = process_batch(zip_path, "/tmp/work")
        assert result["status"] == BatchStatus.FAILED, f"Expected FAILED, got {result['status']}"
        assert (
            "zip-slip" in result["error"].lower() or "path traversal" in result["error"].lower()
        ), f"Expected zip-slip error, got: {result['error']}"
        print("test_zip_slip_detection PASSED")
    finally:
        os.unlink(zip_path)


def test_zip_bomb_byte_counting():
    """ZIP bomb is detected by member count."""
    # 6000 members should trigger the bomb check
    zip_bytes, _ = _make_zip_with_member([(f"file_{i:04d}.txt", b"X" * 10) for i in range(6000)])

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
        tmp.write(zip_bytes)

    try:
        result = process_batch(zip_path, "/tmp/work")
        # Should fail due to too many members
        assert result["status"] == BatchStatus.FAILED, f"Expected FAILED, got {result['status']}"
        print("test_zip_bomb_byte_counting PASSED")
    finally:
        os.unlink(zip_path)


def test_symlink_member_rejection():
    """ZIP with symlink member is rejected."""
    import io
    import zipfile

    buffer = io.BytesIO()
    info = zipfile.ZipInfo("link.png")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(info, "target.png")
    zip_bytes = buffer.getvalue()

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
        tmp.write(zip_bytes)

    try:
        result = process_batch(zip_path, "/tmp/work")
        assert result["status"] == BatchStatus.FAILED
        assert "symlink" in result["error"].lower()
    finally:
        os.unlink(zip_path)


def test_mixed_valid_and_corrupt_batch_completes(tmp_path):
    """A valid image and corrupt member retain per-file results."""
    import io

    from PIL import Image

    valid = io.BytesIO()
    Image.new("RGB", (12, 10), "blue").save(valid, format="PNG")
    zip_bytes, _ = _make_zip_with_member(
        [
            ("image.png", valid.getvalue()),
            ("corrupt.png", b"This is just corrupt image data."),
        ]
    )

    zip_path = tmp_path / "mixed.zip"
    zip_path.write_bytes(zip_bytes)

    class Detector:
        def predict_raw(self, image):
            return {
                "tampered_prob": 0.2,
                "bbox_raw": None,
                "image_size": (12, 10),
                "model_version": "stub",
            }

    result = process_batch(str(zip_path), str(tmp_path), detector=Detector())
    assert result["status"] == BatchStatus.COMPLETED
    statuses = sorted(item["status"] for item in result["results"])
    assert statuses == [BatchStatus.COMPLETED, BatchStatus.FAILED]
    validation = process_mixed_batch(str(zip_path), str(tmp_path))
    assert sorted(item["status"] for item in validation["results"]) == ["corrupt", "valid"]


def test_cancellation_mid_batch():
    """Batch processing can be cancelled mid-stream."""

    class CancelAfterFirstMember:
        def __init__(self):
            self.checks = 0

        def get(self, _key, _default=None):
            self.checks += 1
            return self.checks > 1

    zip_bytes, _ = _make_zip_with_member(
        [(f"file_{i:03d}.jpg", os.urandom(1000)) for i in range(100)]
    )

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
        tmp.write(zip_bytes)

    try:
        result = process_mixed_batch(zip_path, "/tmp/work", CancelAfterFirstMember())
        assert result["status"] == BatchStatus.CANCELLED
        assert len(result["results"]) == 1
    finally:
        os.unlink(zip_path)


def test_nested_zip_detection():
    """Nested ZIP is detected."""
    # Create inner ZIP
    inner_bytes, _ = _make_zip_with_member([("inner.txt", b"inner content")])

    zip_bytes, _ = _make_zip_with_member(
        [
            ("outer.zip", inner_bytes),
        ]
    )

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
        tmp.write(zip_bytes)

    try:
        result = process_batch(zip_path, "/tmp/work")
        assert result["status"] == BatchStatus.FAILED
        assert "nested" in result["error"].lower()
    finally:
        os.unlink(zip_path)


def test_streamed_uncompressed_byte_limit(tmp_path, monkeypatch):
    """A small compressed archive cannot exceed the streamed-byte budget."""
    import io
    import zipfile

    from agent.config import get_config

    config = get_config()["forensics"]
    monkeypatch.setitem(config, "max_zip_compression_ratio", 100000)
    monkeypatch.setitem(config, "max_zip_uncompressed_bytes", 1_000_000)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("large.png", b"\0" * 1_000_001)
    zip_path = tmp_path / "bomb.zip"
    zip_path.write_bytes(buffer.getvalue())
    result = process_batch(str(zip_path), str(tmp_path))
    assert result["status"] == BatchStatus.FAILED
    assert "streamed uncompressed" in result["error"].lower()


def test_zip_bomb_compression_ratio_rejected(tmp_path):
    import zipfile

    from agent.config import get_config

    zip_path = tmp_path / "ratio-bomb.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("compressed.png", b"A" * 200_000)
    result = process_batch(str(zip_path), str(tmp_path))
    assert result["status"] == BatchStatus.FAILED
    assert "compression ratio" in result["error"].lower()
    assert get_config()["forensics"]["max_zip_compression_ratio"] > 0


def test_absolute_zip_path_rejected(tmp_path):
    zip_bytes, _ = _make_zip_with_member([("/tmp/escape.png", b"x")])
    zip_path = tmp_path / "absolute.zip"
    zip_path.write_bytes(zip_bytes)
    result = process_batch(str(zip_path), str(tmp_path))
    assert result["status"] == BatchStatus.FAILED
    assert "slip" in result["error"].lower()


def test_valid_image_batch_uses_safe_names_and_removes_job_temp(tmp_path, monkeypatch):
    import io

    from PIL import Image

    image_buffer = io.BytesIO()
    Image.new("RGB", (20, 12), "green").save(image_buffer, format="PNG")
    zip_bytes, _ = _make_zip_with_member([("../unsafe name.png", image_buffer.getvalue())])
    zip_path = tmp_path / "valid.zip"
    zip_path.write_bytes(zip_bytes)

    # A fake specialist keeps this integration test CPU-only and avoids weight loads.
    class Detector:
        def predict_raw(self, image):
            return {
                "tampered_prob": 0.2,
                "bbox_raw": None,
                "image_size": (20, 12),
                "model_version": "stub",
            }

    # This archive member intentionally traverses, so first prove it is rejected.
    rejected = process_batch(str(zip_path), str(tmp_path), detector=Detector())
    assert rejected["status"] == BatchStatus.FAILED
    assert "zip-slip" in rejected["error"].lower()

    zip_bytes, _ = _make_zip_with_member([("user supplied.png", image_buffer.getvalue())])
    zip_path.write_bytes(zip_bytes)
    result = process_batch(str(zip_path), str(tmp_path), detector=Detector())
    assert result["status"] == BatchStatus.COMPLETED
    assert result["results"][0]["filename"] == "user supplied.png"
    assert result["results"][0]["status"] == BatchStatus.COMPLETED
    assert result.get("batch_pdf")
    assert not list(tmp_path.glob("job_*"))
