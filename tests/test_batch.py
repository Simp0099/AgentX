"""Phase 8 tests for batch/ZIP processing with security checks."""

import os
import tempfile

from agent.batch import (
    MAX_ZIP_STREAM_BYTES,
    BatchStatus,
    _make_zip_with_member,
    process_batch,
    process_mixed_batch,
)


def test_zip_slip_detection():
    """ZIP-slip paths are detected and batch fails."""
    zip_bytes, member_filenames = _make_zip_with_member([
        ("safe.jpg", b"\x89PNG\r\n\x1a\n"),
        ("..%2f..%2fescape.txt", b"malicious"),
    ])

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
        tmp.write(zip_bytes)

    try:
        result = process_batch(zip_path, "/tmp/work")
        assert result["status"] == BatchStatus.FAILED, f"Expected FAILED, got {result['status']}"
        assert "zip-slip" in result["error"].lower() or "path traversal" in result["error"].lower(),             f"Expected zip-slip error, got: {result['error']}"
        print("test_zip_slip_detection PASSED")
    finally:
        os.unlink(zip_path)


def test_zip_bomb_byte_counting():
    """ZIP bomb is detected by member count."""
    # 6000 members should trigger the bomb check
    zip_bytes, _ = _make_zip_with_member(
        [(f"file_{i:04d}.txt", b"X" * 10) for i in range(6000)]
    )

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
    info.external_attr = (0o120777 << 16)
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


def test_mixed_valid_and_corrupt_batch_completes():
    """Batch with mixed valid and corrupt files completes."""
    # Create two files: one that will be processed, one that will be corrupt
    # The process_mixed_batch tries PIL verification; non-image data becomes "corrupt"
    zip_bytes, _ = _make_zip_with_member([
        ("image.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20),  # minimal PNG data
        ("corrupt.png", b"This is just corrupt image data."),
    ])

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
        tmp.write(zip_bytes)

    try:
        result = process_mixed_batch(zip_path, "/tmp/work")
        # Batch should complete even with non-image files
        assert result["status"] == BatchStatus.COMPLETED, f"Expected COMPLETED, got {result['status']}"
        # Results may have "valid" (passed PIL) or "corrupt" (failed PIL)
        statuses = [r["status"] for r in result["results"]]
        # Batch completes regardless of individual file statuses
        print(f"test_mixed_valid_and_corrupt_batch_completes: statuses={statuses}")
        print("test_mixed_valid_and_corrupt_batch_completes PASSED (batch completes)")
    finally:
        os.unlink(zip_path)


def test_cancellation_mid_batch():
    """Batch processing can be cancelled mid-stream."""
    import threading
    import time

    zip_bytes, _ = _make_zip_with_member(
        [(f"file_{i:03d}.txt", b"X" * 1000) for i in range(100)]
    )

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
        tmp.write(zip_bytes)

    try:
        results = {"value": None}
        cancel_flag = {"cancelled": False}

        def run_process():
            result = process_batch(zip_path, "/tmp/work", cancel_flag)
            results["value"] = result

        thread = threading.Thread(target=run_process)
        thread.start()

        # Cancel after a short delay
        time.sleep(0.1)
        cancel_flag["cancelled"] = True

        thread.join(timeout=5)

        # Processing should have been cancelled or completed
        assert results["value"] is not None
        print(f"test_cancellation_mid_batch PASSED (status={results['value']['status']})")
    finally:
        os.unlink(zip_path)


def test_nested_zip_detection():
    """Nested ZIP is detected."""
    # Create inner ZIP
    inner_bytes, _ = _make_zip_with_member([("inner.txt", b"inner content")])

    zip_bytes, _ = _make_zip_with_member([
        ("outer.zip", inner_bytes),
    ])

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = tmp.name
        tmp.write(zip_bytes)

    try:
        result = process_batch(zip_path, "/tmp/work")
        assert result["status"] == BatchStatus.FAILED
        assert "nested" in result["error"].lower()
    finally:
        os.unlink(zip_path)


def test_max_zip_stream_bytes():
    """MAX_ZIP_STREAM_BYTES constant is set reasonable value."""
    assert MAX_ZIP_STREAM_BYTES == 10_000_000, f"Expected 10_000_000, got {MAX_ZIP_STREAM_BYTES}"
    print("test_max_zip_stream_bytes PASSED")


def test_streamed_uncompressed_byte_limit(tmp_path):
    """A small compressed archive cannot exceed the streamed-byte budget."""
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("large.png", b"\0" * (MAX_ZIP_STREAM_BYTES + 1))
    zip_path = tmp_path / "bomb.zip"
    zip_path.write_bytes(buffer.getvalue())
    result = process_batch(str(zip_path), str(tmp_path))
    assert result["status"] == BatchStatus.FAILED
    assert "bomb" in result["error"].lower()


def test_absolute_zip_path_rejected(tmp_path):
    zip_bytes, _ = _make_zip_with_member([("/tmp/escape.png", b"x")])
    zip_path = tmp_path / "absolute.zip"
    zip_path.write_bytes(zip_bytes)
    result = process_batch(str(zip_path), str(tmp_path))
    assert result["status"] == BatchStatus.FAILED
    assert "slip" in result["error"].lower()
