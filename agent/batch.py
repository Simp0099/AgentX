"""Bounded, per-job ZIP batch validation and forensic investigation."""

from __future__ import annotations

import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from PIL import Image

from agent.config import get_config


class BatchStatus:
    """Lifecycle values displayed for batch processing."""

    QUEUED = "Queued"
    VALIDATING = "Validating"
    PROCESSING = "Processing"
    ADDITIONAL_CHECK = "Additional check"
    COMPLETED = "Completed"
    INCONCLUSIVE = "Inconclusive"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


def _make_zip_with_member(filenames_and_content):
    """Create a small in-memory ZIP for tests."""
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, content in filenames_and_content:
            archive.writestr(filename, content)
    return buffer.getvalue(), [filename for filename, _ in filenames_and_content]


def _member_name_is_unsafe(filename: str) -> bool:
    normalized = unquote(filename).replace("\\", "/")
    path = PurePosixPath(normalized)
    return (
        normalized.startswith("/")
        or bool(path.parts and ":" in path.parts[0])
        or ".." in path.parts
    )


def _check_zip_slip(member_filenames):
    return any(_member_name_is_unsafe(filename) for filename in member_filenames)


def _check_zip_bomb(member_count, compressed_bytes=0, uncompressed_bytes=0, max_ratio=100):
    if member_count > 5000:
        return True
    if uncompressed_bytes <= 0:
        return False
    return uncompressed_bytes / max(compressed_bytes, 1) > max_ratio


def _has_symlink_member(zip_bytes):
    import io

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        return any(
            ((member.external_attr >> 16) & 0o170000) == 0o120000 for member in archive.infolist()
        )


def _forensics_config():
    return get_config().get("forensics", {})


def _tool_error(name: str, error: str) -> dict:
    return {"filename": name, "status": BatchStatus.FAILED, "error": error}


def _is_junk_member(filename: str) -> bool:
    path = PurePosixPath(filename.replace("\\", "/"))
    return (
        "__MACOSX" in path.parts
        or path.name.startswith(".")
        or path.name.casefold()
        in {
            "thumbs.db",
            "desktop.ini",
        }
        if not _member_name_is_unsafe(filename)
        else False
    )


def _analyze_extracted(path: Path, original_name: str, detector, image_index: int) -> dict:
    from agent.orchestrator import run_investigation

    events = [BatchStatus.QUEUED, BatchStatus.VALIDATING, BatchStatus.PROCESSING]
    record = run_investigation(
        str(path),
        goal="quick_check",
        config={
            "max_tool_calls_per_image": int(_forensics_config().get("max_tool_calls_per_image", 10))
        },
        detector=detector,
    )
    if any(
        obs.tool in ("generate_ela", "inspect_metadata", "compare_evidence")
        for obs in record.observations
    ):
        events.append(BatchStatus.ADDITIONAL_CHECK)
    if record.verdict is None:
        status = BatchStatus.FAILED
    elif record.verdict.value == "Inconclusive":
        status = BatchStatus.INCONCLUSIVE
    else:
        status = BatchStatus.COMPLETED
    events.append(status)
    record.filename = original_name
    return {
        "filename": original_name,
        "status": status,
        "events": events,
        "verdict": record.verdict.value if record.verdict else "Inconclusive",
        "tampered_prob": next(
            (
                (obs.result or {}).get("tampered_prob")
                for obs in record.observations
                if obs.tool == "run_localizer"
            ),
            None,
        ),
        "record": record,
    }


def _prepare_members(archive: zipfile.ZipFile, config: dict) -> tuple[list, str | None]:
    members = [
        member
        for member in archive.infolist()
        if not member.is_dir() and not _is_junk_member(member.filename)
    ]
    if any(
        _member_name_is_unsafe(member.filename)
        for member in archive.infolist()
        if not member.is_dir()
    ):
        return [], "ZIP-slip detected: unsafe member path"
    if len(members) > int(config.get("max_zip_members", 100)):
        return [], "ZIP contains too many files"
    if any(_member_name_is_unsafe(member.filename) for member in members):
        return [], "ZIP-slip detected: unsafe member path"
    if any(((member.external_attr >> 16) & 0o170000) == 0o120000 for member in members):
        return [], "ZIP contains a symlink member"
    total_header_size = sum(member.file_size for member in members)
    total_compressed_size = sum(member.compress_size for member in members)
    if _check_zip_bomb(
        len(members),
        total_compressed_size,
        total_header_size,
        float(config.get("max_zip_compression_ratio", 100)),
    ):
        return [], "ZIP bomb detected: compression ratio exceeds configured limit"
    for member in members:
        if Path(member.filename).suffix.lower() == ".zip":
            return [], "Nested ZIP archives are not accepted"
    return members, None


def process_batch(
    zip_path: str,
    work_dir: str,
    cancel_flag: dict | None = None,
    detector=None,
) -> dict:
    """Validate and investigate every archive image using generated safe names."""
    cancel_flag = cancel_flag if cancel_flag is not None else {"cancelled": False}
    config = _forensics_config()
    limits = {
        "compressed": int(config.get("max_zip_compressed_bytes", 25_000_000)),
        "uncompressed": int(config.get("max_zip_uncompressed_bytes", 100_000_000)),
        "members": int(config.get("max_zip_members", 100)),
        "pixels": int(config.get("max_image_pixels", 40_000_000)),
        "file_bytes": int(config.get("max_file_bytes", 50_000_000)),
        "seconds": float(config.get("max_batch_processing_seconds", 120)),
        "workers": max(1, int(config.get("batch_workers", 2))),
    }
    started = time.monotonic()
    archive_path = Path(zip_path)
    if not archive_path.is_file():
        return {"status": BatchStatus.FAILED, "results": [], "error": "ZIP file does not exist"}
    if archive_path.stat().st_size > limits["compressed"]:
        return {
            "status": BatchStatus.FAILED,
            "results": [],
            "error": "ZIP exceeds compressed-byte limit",
        }
    destination = Path(work_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    records = []
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            members, validation_error = _prepare_members(archive, limits | config)
            if validation_error:
                return {"status": BatchStatus.FAILED, "results": [], "error": validation_error}
            if not members:
                return {"status": BatchStatus.COMPLETED, "results": [], "error": None}

            supported = set(
                config.get(
                    "supported_image_extensions",
                    [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"],
                )
            )
            images = []
            for member in members:
                if Path(member.filename).suffix.lower() in supported:
                    images.append(member)
                else:
                    results.append(
                        {
                            "filename": member.filename,
                            "status": "Skipped",
                            "reason": "not an image",
                        }
                    )
            members = images
            if not members:
                return {"status": BatchStatus.COMPLETED, "results": results, "error": None}

            with tempfile.TemporaryDirectory(prefix="job_", dir=destination) as temp_name:
                job_dir = Path(temp_name)
                prepared = []
                total_streamed = 0
                for index, member in enumerate(members, start=1):
                    if cancel_flag.get("cancelled"):
                        return {
                            "status": BatchStatus.CANCELLED,
                            "results": results,
                            "error": "Batch processing cancelled during extraction",
                        }
                    if time.monotonic() - started > limits["seconds"]:
                        return {
                            "status": BatchStatus.CANCELLED,
                            "results": results,
                            "error": "Batch processing time limit exceeded",
                        }
                    suffix = Path(member.filename).suffix.lower()
                    safe_path = job_dir / f"image_{index:06d}{suffix}"
                    first_bytes = b""
                    actual_size = 0
                    try:
                        with archive.open(member) as source, safe_path.open("wb") as target:
                            while chunk := source.read(64 * 1024):
                                if not first_bytes:
                                    first_bytes = chunk[:4]
                                actual_size += len(chunk)
                                total_streamed += len(chunk)
                                if actual_size > limits["file_bytes"]:
                                    raise ValueError("image exceeds configured file-size limit")
                                if total_streamed > limits["uncompressed"]:
                                    return {
                                        "status": BatchStatus.FAILED,
                                        "results": results,
                                        "error": "ZIP exceeds streamed uncompressed-byte limit",
                                    }
                                target.write(chunk)
                        if first_bytes.startswith(b"PK\x03\x04") or first_bytes.startswith(
                            b"PK\x05\x06"
                        ):
                            return {
                                "status": BatchStatus.FAILED,
                                "results": results,
                                "error": "Nested ZIP content is not accepted",
                            }
                        with Image.open(safe_path) as image:
                            width, height = image.size
                            if width <= 0 or height <= 0 or width * height > limits["pixels"]:
                                raise ValueError("image dimensions exceed configured limit")
                            image.verify()
                        prepared.append(
                            {
                                "path": safe_path,
                                "filename": member.filename,
                                "width": width,
                                "height": height,
                                "size": actual_size,
                            }
                        )
                    except Exception as exc:  # a corrupt image remains a per-file failure
                        results.append(_tool_error(member.filename, str(exc)))

                if not prepared:
                    return {"status": BatchStatus.COMPLETED, "results": results, "error": None}

                if detector is None:
                    try:
                        from src.inference import TamperingDetector

                        model_config = get_config()
                        checkpoint_dir = model_config.get("output", {}).get(
                            "checkpoint_dir", "checkpoints/"
                        )
                        checkpoint_name = model_config.get("output", {}).get(
                            "checkpoint_name", "best_localizer.keras"
                        )
                        detector = TamperingDetector(str(Path(checkpoint_dir) / checkpoint_name))
                    except Exception as exc:
                        detector_error = f"detector unavailable: {type(exc).__name__}"
                    else:
                        detector_error = ""
                else:
                    detector_error = ""

                if detector_error:
                    results.extend(
                        _tool_error(item["filename"], detector_error) for item in prepared
                    )
                else:
                    future_map = {}
                    with ThreadPoolExecutor(max_workers=limits["workers"]) as pool:
                        for index, item in enumerate(prepared, start=1):
                            if cancel_flag.get("cancelled"):
                                break
                            if time.monotonic() - started > limits["seconds"]:
                                break
                            future = pool.submit(
                                _analyze_extracted, item["path"], item["filename"], detector, index
                            )
                            future_map[future] = item
                        try:
                            for future in as_completed(
                                future_map,
                                timeout=max(0.01, limits["seconds"] - (time.monotonic() - started)),
                            ):
                                if cancel_flag.get("cancelled"):
                                    for pending in future_map:
                                        pending.cancel()
                                    break
                                item = future_map[future]
                                try:
                                    result = future.result()
                                    results.append(result)
                                    if result.get("record"):
                                        records.append(result["record"])
                                except Exception as exc:
                                    results.append(_tool_error(item["filename"], str(exc)))
                        except TimeoutError:
                            for future in future_map:
                                future.cancel()
                            results.extend(
                                _tool_error(
                                    item["filename"], "batch processing time limit exceeded"
                                )
                                for future, item in future_map.items()
                                if not future.done()
                            )

                if cancel_flag.get("cancelled"):
                    return {
                        "status": BatchStatus.CANCELLED,
                        "results": results,
                        "error": "Batch processing cancelled",
                    }
                if time.monotonic() - started > limits["seconds"]:
                    return {
                        "status": BatchStatus.CANCELLED,
                        "results": results,
                        "error": "Batch processing time limit exceeded",
                    }
                if records:
                    try:
                        from agent.reporting import generate_batch_pdf

                        pdf_path = destination / f"batch_{int(time.time())}.pdf"
                        generate_batch_pdf(records, pdf_path)
                        return {
                            "status": BatchStatus.COMPLETED,
                            "results": results,
                            "error": None,
                            "batch_pdf": str(pdf_path),
                        }
                    except Exception as exc:
                        return {
                            "status": BatchStatus.COMPLETED,
                            "results": results,
                            "error": f"batch PDF unavailable: {type(exc).__name__}",
                        }
    except zipfile.BadZipFile:
        return {"status": BatchStatus.FAILED, "results": [], "error": "Invalid ZIP file"}
    except Exception as exc:
        return {"status": BatchStatus.FAILED, "results": results, "error": str(exc)}
    return {"status": BatchStatus.COMPLETED, "results": results, "error": None}


def process_mixed_batch(zip_path: str, work_dir: str, cancel_flag: dict | None = None) -> dict:
    """Validation-only batch variant used to report valid and corrupt members."""
    cancel_flag = cancel_flag if cancel_flag is not None else {"cancelled": False}
    config = _forensics_config()
    destination = Path(work_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            members, error = _prepare_members(archive, config)
            if error:
                return {"status": BatchStatus.FAILED, "results": [], "error": error}
            supported = set(
                config.get(
                    "supported_image_extensions",
                    [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"],
                )
            )
            for member in members:
                if Path(member.filename).suffix.lower() not in supported:
                    results.append(
                        {
                            "filename": member.filename,
                            "status": "Skipped",
                            "reason": "not an image",
                        }
                    )
            members = [
                member for member in members if Path(member.filename).suffix.lower() in supported
            ]
            with tempfile.TemporaryDirectory(prefix="mixed_job_", dir=destination) as temp_name:
                job_dir = Path(temp_name)
                total = 0
                limit = int(config.get("max_zip_uncompressed_bytes", 100_000_000))
                for index, member in enumerate(members, start=1):
                    if cancel_flag.get("cancelled"):
                        return {
                            "status": BatchStatus.CANCELLED,
                            "results": results,
                            "error": "Batch processing cancelled mid-batch",
                        }
                    path = job_dir / f"image_{index:06d}{Path(member.filename).suffix.lower()}"
                    count = 0
                    with archive.open(member) as source, path.open("wb") as target:
                        while chunk := source.read(64 * 1024):
                            count += len(chunk)
                            total += len(chunk)
                            if total > limit:
                                return {
                                    "status": BatchStatus.FAILED,
                                    "results": results,
                                    "error": "ZIP exceeds streamed uncompressed-byte limit",
                                }
                            target.write(chunk)
                    try:
                        with Image.open(path) as image:
                            width, height = image.size
                            if width * height > int(config.get("max_image_pixels", 40_000_000)):
                                raise ValueError("image dimensions exceed configured limit")
                            image.verify()
                        status = "valid"
                    except Exception:
                        status = "corrupt"
                    results.append(
                        {
                            "filename": member.filename,
                            "status": status,
                            "size": count,
                            "width": width if status == "valid" else None,
                            "height": height if status == "valid" else None,
                        }
                    )
    except zipfile.BadZipFile:
        return {"status": BatchStatus.FAILED, "results": [], "error": "Invalid ZIP file"}
    except Exception as exc:
        return {"status": BatchStatus.FAILED, "results": results, "error": str(exc)}
    return {"status": BatchStatus.COMPLETED, "results": results, "error": None}
