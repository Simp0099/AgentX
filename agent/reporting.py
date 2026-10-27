"""Generate deterministic PDF reports from stored investigation records."""

from __future__ import annotations

import html
import json
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as ReportImage,
)
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from agent.config import get_config
from agent.schemas import InvestigationRecord
from src.visualize import draw_boxes

_base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle(
        "report-title",
        parent=_base["Title"],
        fontSize=20,
        leading=25,
        textColor=colors.HexColor("#1a4d7a"),
        spaceAfter=8,
    ),
    "subtitle": ParagraphStyle(
        "report-subtitle",
        parent=_base["Normal"],
        fontSize=10,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
    ),
    "h1": ParagraphStyle(
        "report-h1",
        parent=_base["Heading1"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1a4d7a"),
        spaceBefore=8,
        spaceAfter=6,
    ),
    "h2": ParagraphStyle(
        "report-h2",
        parent=_base["Heading2"],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#23272b"),
        spaceBefore=6,
        spaceAfter=4,
    ),
    "body": ParagraphStyle(
        "report-body", parent=_base["Normal"], fontSize=9, leading=12, spaceAfter=4, wordWrap="CJK"
    ),
    "small": ParagraphStyle(
        "report-small",
        parent=_base["Normal"],
        fontSize=7.5,
        leading=9,
        spaceAfter=3,
        wordWrap="CJK",
    ),
    "cell": ParagraphStyle(
        "report-cell", parent=_base["Normal"], fontSize=8, leading=10, wordWrap="CJK"
    ),
}

_ADDITIONAL_TOOLS = ("generate_ela", "inspect_metadata", "compare_evidence")
_ALL_TOOLS = (
    "inspect_image",
    "run_localizer",
    "validate_prediction",
    *_ADDITIONAL_TOOLS,
)


def _escape_text(value: object) -> str:
    """Escape untrusted values before passing them to ReportLab Paragraph."""
    if value is None:
        return ""
    return html.escape(str(value), quote=False).replace("\n", "<br/>")


def _paragraph(label: str, value: object, style="body") -> Paragraph:
    # Paragraph resolves one HTML entity layer, so escape that layer again to
    # retain literal entities in the rendered and extracted report text.
    safe_label = html.escape(_escape_text(label), quote=False)
    safe_value = html.escape(_escape_text(value), quote=False)
    return Paragraph(f"<b>{safe_label}</b> {safe_value}", S[style])


def _observations(record: InvestigationRecord) -> list:
    return [item for item in record.observations if not isinstance(item, list)]


def _result(record: InvestigationRecord, tool: str) -> dict:
    for observation in _observations(record):
        if observation.tool == tool and observation.result:
            return observation.result
    return {}


def _asset_roots(record: InvestigationRecord) -> list[Path]:
    config = record.config_snapshot or get_config()
    forensics = config.get("forensics", {}) if isinstance(config, dict) else {}
    roots = forensics.get(
        "report_asset_roots", ["artifacts/uploads", "artifacts/batch", "artifacts/ela"]
    )
    ela_root = forensics.get("ela_temp_dir", "artifacts/ela")
    return [Path(root).resolve() for root in [*roots, ela_root]]


def _trusted_asset_path(record: InvestigationRecord, candidate: object) -> Path | None:
    if not isinstance(candidate, str) or not candidate:
        return None
    path = Path(candidate).resolve()
    for root in _asset_roots(record):
        try:
            path.relative_to(root)
            if path.is_file():
                return path
        except ValueError:
            continue
    return None


def _asset_flowable(path: Path | None, caption: str, max_width=8.5 * cm, max_height=6.5 * cm):
    if path is None:
        return [Paragraph(_escape_text(f"{caption}: unavailable"), S["small"])]
    try:
        with PILImage.open(path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((1200, 900))
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            flowable = ReportImage(buffer)
            scale = min(max_width / flowable.imageWidth, max_height / flowable.imageHeight, 1.0)
            flowable.drawWidth = flowable.imageWidth * scale
            flowable.drawHeight = flowable.imageHeight * scale
        return [Paragraph(_escape_text(caption), S["small"]), flowable, Spacer(1, 5)]
    except (OSError, ValueError):
        return [Paragraph(_escape_text(f"{caption}: unavailable"), S["small"])]


def _source_image_path(record: InvestigationRecord) -> Path | None:
    if record.image_path:
        trusted = _trusted_asset_path(record, record.image_path)
        if trusted:
            return trusted
    inspected = _result(record, "inspect_image")
    for candidate in (
        inspected.get("path"),
        *[obs.args.get("path") for obs in _observations(record) if obs.tool == "inspect_image"],
    ):
        trusted = _trusted_asset_path(record, candidate)
        if trusted:
            return trusted
    return None


def _build_story(record: InvestigationRecord) -> list:
    observations = _observations(record)
    inspect = _result(record, "inspect_image")
    localizer = _result(record, "run_localizer")
    validation = _result(record, "validate_prediction")
    metadata = _result(record, "inspect_metadata")
    ela = _result(record, "generate_ela")
    verdict = record.verdict.value if record.verdict else "Inconclusive"
    filename = record.filename or "(not recorded)"
    score = localizer.get("tampered_prob")
    bbox = localizer.get("bbox", localizer.get("bbox_raw"))
    source_path = _source_image_path(record)
    story = [
        Paragraph("Deepfake Forensic Investigation", S["title"]),
        Paragraph("Structured report generated from stored tool results", S["subtitle"]),
        Spacer(1, 10),
        Paragraph("Cover", S["h1"]),
        _paragraph("Report ID:", record.investigation_id),
        _paragraph("UTC timestamp:", record.created_at_utc),
        _paragraph("Scope:", f"{record.goal}: {record.goal_text}".strip()),
        _paragraph("Outcome:", verdict),
        PageBreak(),
        Paragraph("Image identity", S["h1"]),
        _paragraph("Image:", filename),
        _paragraph("SHA-256:", record.sha256 or inspect.get("sha256", "not recorded")),
        _paragraph(
            "Dimensions:",
            f"{inspect.get('width', 'not reported')} x {inspect.get('height', 'not reported')}",
        ),
        _paragraph("Format:", inspect.get("format", "not reported")),
        _paragraph("Byte size:", inspect.get("byte_size", "not reported")),
        Spacer(1, 6),
        Paragraph("Evidence", S["h1"]),
    ]
    story.extend(_asset_flowable(source_path, "Image thumbnail"))
    if verdict == "Flagged for review" and validation.get("valid") and bbox:
        if source_path:
            try:
                with PILImage.open(source_path) as source:
                    overlay = draw_boxes(source, [bbox])
                overlay_buffer = BytesIO()
                overlay.save(overlay_buffer, format="PNG")
                overlay_buffer.seek(0)
                flowable = ReportImage(overlay_buffer)
                scale = min(8.5 * cm / flowable.imageWidth, 6.5 * cm / flowable.imageHeight, 1.0)
                flowable.drawWidth, flowable.drawHeight = (
                    flowable.imageWidth * scale,
                    flowable.imageHeight * scale,
                )
                story.extend(
                    [Paragraph("Model-predicted tampered region (ELA-based)", S["small"]), flowable]
                )
            except (OSError, ValueError, TypeError):
                story.append(Paragraph("Overlay: unavailable", S["small"]))
    ela_path = _trusted_asset_path(record, ela.get("artifact"))
    if ela:
        story.extend(_asset_flowable(ela_path, "ELA image"))
    if metadata.get("exif_status") == "absent":
        story.append(_paragraph("EXIF:", "absent"))
    elif metadata.get("exif_status") == "parse_error":
        story.append(_paragraph("Metadata:", f"parse error: {metadata.get('parse_errors', [])}"))
    elif metadata.get("present_fields"):
        story.append(_paragraph("Metadata fields present:", ", ".join(metadata["present_fields"])))

    story.extend(
        [
            PageBreak(),
            Paragraph("Model output", S["h1"]),
            _paragraph(
                "tampered score (uncalibrated):", score if score is not None else "not reported"
            ),
            _paragraph("Threshold used:", record.threshold),
            _paragraph("Bounding box:", ", ".join(map(str, bbox)) if bbox else "not reported"),
            _paragraph(
                "Model version:",
                record.model_version or localizer.get("model_version", "not reported"),
            ),
            Paragraph("Additional checks", S["h1"]),
        ]
    )
    conditional = [
        obs for obs in observations if obs.tool in _ADDITIONAL_TOOLS and obs.status != "rejected"
    ]
    if conditional:
        for obs in conditional:
            story.append(
                _paragraph(
                    f"{obs.tool} ({obs.status}):",
                    json.dumps(obs.result or {}, sort_keys=True),
                    "small",
                )
            )
    else:
        story.append(Paragraph("No additional checks were executed.", S["body"]))

    story.extend([PageBreak(), Paragraph("Observed facts", S["h1"])])
    facts = []
    if inspect:
        facts.extend(
            [
                f"Image inspection valid: {inspect.get('valid')}",
                f"Image dimensions: {inspect.get('width')} x {inspect.get('height')}",
            ]
        )
    if score is not None:
        facts.append(f"Localizer score: {score}")
    if validation:
        facts.append(
            f"Prediction validation: {validation.get('valid')}; "
            f"issues: {validation.get('issues', [])}"
        )
    for item in facts:
        story.append(Paragraph(_escape_text(item), S["body"]))
    story.append(Paragraph("Interpretation", S["h1"]))
    for reason in record.verdict_reasons:
        story.append(Paragraph(_escape_text(reason), S["body"]))
    if not record.verdict_reasons:
        story.append(Paragraph("No interpretation rationale was recorded.", S["body"]))

    config = record.config_snapshot or get_config()
    limitations = config.get("forensics", {}).get("report_limitations", [])
    story.extend([PageBreak(), Paragraph("Limitations", S["h1"])])
    for limitation in limitations:
        story.append(Paragraph(f"- {_escape_text(limitation)}", S["body"]))
    record_limitations = list(record.errors)
    record_limitations.extend(
        f"{obs.tool}: {obs.rationale}" for obs in observations if obs.status in ("error", "timeout")
    )
    if record_limitations:
        story.append(Paragraph("Record-specific limitations", S["h2"]))
        for limitation in record_limitations:
            story.append(Paragraph(f"- {_escape_text(limitation)}", S["body"]))

    story.extend([PageBreak(), Paragraph("Audit trail", S["h1"])])
    for obs in observations:
        story.extend(
            [
                Paragraph(_escape_text(f"{obs.tool} - {obs.status}"), S["h2"]),
                _paragraph("Timestamp:", obs.timestamp_utc, "small"),
                _paragraph(
                    "Duration (ms):",
                    obs.duration_ms if obs.duration_ms is not None else "not reported",
                    "small",
                ),
                _paragraph(
                    "Arguments:", json.dumps(obs.args, sort_keys=True, default=str), "small"
                ),
                _paragraph("Result:", json.dumps(obs.result, sort_keys=True, default=str), "small"),
                _paragraph("Error / rationale:", obs.rationale or "none", "small"),
            ]
        )
    story.append(_paragraph("Config snapshot hash:", record.config_snapshot_hash, "small"))
    story.append(
        _paragraph("Config snapshot:", json.dumps(config, sort_keys=True, default=str), "small")
    )
    return story


def _page_number(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {document.page}")
    canvas.restoreState()


def generate_pdf(record: InvestigationRecord, output_path: Path):
    """Generate a PDF using only the structured record and configured text."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Deepfake Forensic Investigation",
        author="Local Forensic Pipeline",
    )
    document.build(_build_story(record), onFirstPage=_page_number, onLaterPages=_page_number)
    return path


def generate_batch_pdf(records: list[InvestigationRecord], output_path: Path):
    """Generate a batch summary table plus a compact per-record outcome list."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [["Filename", "Status", "Raw score", "Verdict category"]]
    for record in records:
        localizer = _result(record, "run_localizer")
        status = "completed" if record.verdict else "incomplete"
        data.append(
            [
                Paragraph(_escape_text(record.filename), S["cell"]),
                Paragraph(_escape_text(status), S["cell"]),
                Paragraph(_escape_text(localizer.get("tampered_prob", "not reported")), S["cell"]),
                Paragraph(
                    _escape_text(record.verdict.value if record.verdict else "Inconclusive"),
                    S["cell"],
                ),
            ]
        )
    table = Table(data, repeatRows=1, colWidths=[6 * cm, 3 * cm, 3 * cm, 5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a4d7a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa7b1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f7")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    document = SimpleDocTemplate(str(path), pagesize=A4, title="Batch Forensic Summary")
    document.build(
        [Paragraph("Batch summary", S["title"]), Spacer(1, 8), table],
        onFirstPage=_page_number,
        onLaterPages=_page_number,
    )
    return path


def generate_from_records(records, output_dir: Path):
    """Generate one record-based PDF for every input record."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for record in records:
        output_path = output_dir / f"{record.investigation_id}.pdf"
        generate_pdf(record, output_path)
        results.append((record.investigation_id, output_path))
    return results
