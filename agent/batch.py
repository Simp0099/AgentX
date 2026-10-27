"""agent/batch.py: multiple files or ZIP processing with security checks.

Security: zip-slip prevention, zip bomb detection (byte counting),
nested zip, symlink member rejection, mixed valid/corrupt batch,
and cancellation mid-batch.

All operations are offline/CPU-bounded. No network calls.
"""

import os
import tempfile
import zipfile
from urllib.parse import unquote

# Maximum bytes to stream from a ZIP file before aborting.
# This prevents zip bomb attacks (e.g. 4GB zipped from 99 bytes).
MAX_ZIP_STREAM_BYTES = 10_000_000  # 10 MB


class BatchStatus:
    """Status values for batch processing."""
    QUEUED = "Queued"
    VALIDATING = "Validating"
    PROCESSING = "Processing"
    ADDITIONAL_CHECK = "Additional check"
    COMPLETED = "Completed"
    INCONCLUSIVE = "Inconclusive"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


def _make_zip_with_member(filenames_and_content):
    """Helper: create a ZIP in memory with given files.

    Returns (zip_bytes, member_filenames) for testing.
    """
    import io
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zz:
        for fname, content in filenames_and_content:
            zz.writestr(fname, content)
    zip_buf.seek(0)
    return zip_buf.read(), [fname for fname, _ in filenames_and_content]


def _check_zip_slip(member_filenames):
    """Check for zip-slip paths (traversal outside extraction directory).

    Returns True if any member is unsafe.
    """
    for filename in member_filenames:
        # Check for path traversal
        normalized = unquote(filename).replace("\\", "/")
        if ".." in normalized.split("/") or normalized.startswith("/"):
            return True
    return False


def _check_zip_bomb(member_count):
    """Check for zip bomb indicators.

    Returns True if the zip appears to be a bomb (too many members).
    """
    if member_count > 5000:
        return True
    return False


def _has_symlink_member(zip_bytes):
    """Check Unix mode bits of ZIP members for symlinks."""
    import io

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        return any((member.external_attr >> 16) & 0o170000 == 0o120000
                   for member in archive.infolist())


def process_batch(zip_path: str, work_dir: str, cancel_flag: dict = None) -> dict:
    """Process a ZIP file containing images for tamper analysis.

    Returns a dict with:
    - status: BatchStatus value
    - results: list of per-file results
    - error: error message if any
    """
    cancel_flag = cancel_flag or {"cancelled": False}
    results = []
    tmp_dir = tempfile.mkdtemp(prefix="batch_")

    try:
        with zipfile.ZipFile(zip_path, "r") as zz:
            members = zz.infolist()
            member_count = len(members)

            # Check for zip-slip paths
            member_filenames = [m.filename for m in members]
            if _check_zip_slip(member_filenames):
                return {
                    "status": BatchStatus.FAILED,
                    "results": [],
                    "error": "ZIP-slip detected: path traversal in ZIP members",
                }

            # Check for zip bomb indicators
            if _check_zip_bomb(member_count):
                return {
                    "status": BatchStatus.FAILED,
                    "results": [],
                    "error": "ZIP bomb detected: excessive members",
                }

            # Check for symlink members
            for member in members:
                if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                    return {
                        "status": BatchStatus.FAILED,
                        "results": [],
                        "error": "ZIP contains symlink members (rejected for security)",
                    }
                if member.is_dir():
                    continue
                if member.filename.lower().endswith(".zip"):
                    return {"status": BatchStatus.FAILED, "results": [],
                            "error": "Nested ZIP archives are not accepted"}
                if os.path.splitext(member.filename)[1].lower() not in {
                    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"
                }:
                    return {"status": BatchStatus.FAILED, "results": [],
                            "error": "Non-image ZIP member rejected"}

            # Extract members safely
            total_streamed = 0
            for member in members:
                if cancel_flag.get("cancelled"):
                    return {
                        "status": BatchStatus.CANCELLED,
                        "results": results,
                        "error": "Batch processing cancelled mid-extraction",
                    }

                # Skip directories
                if member.is_dir():
                    continue

                filename = os.path.basename(member.filename)
                out_path = os.path.join(tmp_dir, filename)

                # Create parent directories
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                # Extract the file, with bomb detection
                bytes_read = 0
                with zz.open(member) as source, open(out_path, "wb") as target:
                    while True:
                        data = source.read(8192)
                        if not data:
                            break
                        bytes_read += len(data)
                        total_streamed += len(data)
                        if total_streamed > MAX_ZIP_STREAM_BYTES:
                            return {
                                "status": BatchStatus.FAILED,
                                "results": results,
                                "error": (
                                    f"ZIP bomb: file exceeded "
                                    f"{MAX_ZIP_STREAM_BYTES/1e6}MB limit"
                                ),
                            }
                        target.write(data)

                results.append({
                    "filename": filename,
                    "path": out_path,
                    "status": "processed",
                    "size": bytes_read,
                })

    except zipfile.BadZipFile:
        return {
            "status": BatchStatus.FAILED,
            "results": [],
            "error": "Invalid ZIP file",
        }
    except Exception as e:
        return {
            "status": BatchStatus.FAILED,
            "results": results,
            "error": str(e),
        }
    finally:
        # Cleanup temp dir if not cancelled
        import shutil
        if not cancel_flag.get("cancelled"):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Batch completed
    return {
        "status": BatchStatus.COMPLETED,
        "results": results,
        "error": None,
    }


def process_mixed_batch(zip_path: str, work_dir: str, cancel_flag: dict = None) -> dict:
    """Process a ZIP with mixed valid and corrupt files.

    The batch completes even if some files are corrupt.
    """
    cancel_flag = cancel_flag or {"cancelled": False}
    results = []
    tmp_dir = tempfile.mkdtemp(prefix="mixed_batch_")

    try:
        with zipfile.ZipFile(zip_path, "r") as zz:
            for member in zz.infolist():
                if cancel_flag.get("cancelled"):
                    return {
                        "status": BatchStatus.CANCELLED,
                        "results": results,
                        "error": "Batch processing cancelled mid-stream",
                    }

                if member.is_dir():
                    continue

                filename = os.path.basename(member.filename)
                out_path = os.path.join(tmp_dir, filename)

                try:
                    with zz.open(member) as source, open(out_path, "wb") as target:
                        bytes_read = 0
                        while True:
                            data = source.read(8192)
                            if not data:
                                break
                            bytes_read += len(data)
                            target.write(data)

                    # Try to verify it's a valid image
                    from PIL import Image
                    try:
                        img = Image.open(out_path)
                        img.verify()
                        results.append({
                            "filename": filename,
                            "path": out_path,
                            "status": "valid",
                            "size": bytes_read,
                        })
                    except Exception:
                        # Corrupt image - still recorded
                        results.append({
                            "filename": filename,
                            "path": out_path,
                            "status": "corrupt",
                            "size": bytes_read,
                        })
                except Exception as e:
                    results.append({
                        "filename": filename,
                        "path": None,
                        "status": "error",
                        "error": str(e),
                        "size": 0,
                    })
    except Exception as e:
        return {
            "status": BatchStatus.FAILED,
            "results": results,
            "error": str(e),
        }
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Batch completes with mixed statuses
    return {
        "status": BatchStatus.COMPLETED,
        "results": results,
        "error": None,
    }
