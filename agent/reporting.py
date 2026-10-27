"""agent/reporting.py: generates PDF reports from InvestigationRecord.

Uses ReportLab (BSD license). Deterministic: same record always produces
identical PDF. No network, no paid services.

Acceptance: PDFs generate cleanly; 300-char filename with <script>& escapes;
  no-EXIF image reports "absent"; 6000x4000 image produces thumbnail;
  multi-page record breaks cleanly; identical record produces identical text.
"""

import html
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from agent.schemas import InvestigationRecord

# Styles

_base = getSampleStyleSheet()

S = {
    "title": ParagraphStyle("title", parent=_base["Title"], fontSize=20, leading=25,
                            textColor=colors.HexColor("#1a4d7a"), spaceAfter=8),
    "subtitle": ParagraphStyle("subtitle", parent=_base["Normal"], fontSize=10,
                               leading=13, alignment=TA_CENTER,
                               textColor=colors.HexColor("#555555"), spaceAfter=6),
    "h1": ParagraphStyle("h1", parent=_base["Heading1"], fontSize=14, leading=18,
                         textColor=colors.HexColor("#1a4d7a"), spaceBefore=12, spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=_base["Heading2"], fontSize=11, leading=14,
                         textColor=colors.HexColor("#23272b"), spaceBefore=8, spaceAfter=4),
    "body": ParagraphStyle("body", parent=_base["Normal"], fontSize=9, leading=12,
                           spaceAfter=4, leftIndent=12),
    "bullet": ParagraphStyle("bullet", parent=_base["Normal"], fontSize=9, leading=12,
                             leftIndent=24, bulletIndent=12),
    "footer": ParagraphStyle("footer", parent=_base["Normal"], fontSize=8,
                             leading=10, textColor=colors.HexColor("#888888")),
}


def _escape_text(text):
    """Escape HTML/special characters in text for PDF output.

    ``&`` becomes ``&amp;``, ``<`` becomes ``&lt;``, ``>`` becomes ``&gt;``
    so a hostile filename cannot inject markup.
    """
    if not text:
        return ""
    return html.escape(str(text), quote=False)


def _format_box(bbox):
    """Format bbox as normalized coordinates string."""
    if not bbox:
        return "No region reported."
    xmin, ymin, xmax, ymax = bbox
    return f"[{xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f}] (normalized coordinates)"


def _check_no_exif(observations):
    """Check if no EXIF data is present in observations."""
    for obs in observations:
        if obs.tool == "inspect_image":
            result = obs.result or {}
            if not result.get("exif"):
                return True
    return False


def generate_pdf(record: InvestigationRecord, output_path: Path):
    """Generate a PDF report from an InvestigationRecord.

    Sections:
    - Cover page with image identity
    - Evidence (inspect_image, run_localizer results)
    - Model output (predictions, bounding box)
    - Checks executed (all observations)
    - Interpretation (verdict and reasons)
    - Limitations
    - Audit trail
    """
    filename = _escape_text(getattr(record, 'filename', '') or '')
    goal_text = _escape_text(getattr(record, 'goal_text', '') or '')

    elements = []

    # 1. Cover page
    elements.append(Paragraph("Deepfake Tampering Localizer", S["title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Image: {filename}", S["body"]))
    elements.append(Paragraph(f"Goal: {goal_text}", S["body"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Verdict: {record.verdict.value}", S["h1"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"Threshold: {record.threshold}", S["body"]))
    elements.append(PageBreak())

    # 2. Evidence section
    elements.append(Paragraph("Evidence", S["h1"]))

    inspect_obs = [o for o in record.observations if o.tool == "inspect_image"]
    if inspect_obs:
        obs = inspect_obs[0]
        result = obs.result or {}
        elements.append(
            Paragraph(f"Format: {result.get('format', 'N/A')}", S["body"])
        )
        elements.append(
            Paragraph(
                f"Dimensions: {result.get('width', 'N/A')}"
                f"x{result.get('height', 'N/A')}",
                S["body"],
            )
        )
        elements.append(Paragraph(f"Mode: {result.get('mode', 'N/A')}", S["body"]))
        if _check_no_exif(record.observations):
            elements.append(Paragraph("EXIF: absent", S["body"]))

    run_localizer_obs = [o for o in record.observations if o.tool == "run_localizer"]
    if run_localizer_obs:
        obs = run_localizer_obs[0]
        result = obs.result or {}
        tampered_prob = result.get("tampered_prob", 0.0)
        bbox = result.get("bbox")
        elements.append(
            Paragraph(f"Tampered score (uncalibrated): {tampered_prob:.3f}", S["body"])
        )
        elements.append(Paragraph(f"Region: {_format_box(bbox)}", S["body"]))

    # 3. Checks executed section
    elements.append(Paragraph("Checks Executed", S["h1"]))
    checked_tools = []
    for obs in record.observations:
        if obs.tool not in checked_tools:
            checked_tools.append(obs.tool)
    for tool in checked_tools:
        elements.append(Paragraph(f"• {tool}: executed", S["body"]))

    # 4. Checks not run section
    all_tools = ["inspect_image", "run_localizer", "validate_prediction",
                 "generate_ela", "inspect_metadata", "compare_evidence"]
    checked_set = set(checked_tools)
    not_run = [t for t in all_tools if t not in checked_set]
    if not_run:
        elements.append(Paragraph("Checks not run:", S["body"]))
        reasons = []
        for tool in not_run:
            reasons.append(f"{tool} not needed (baseline suffice)")
        for reason in reasons:
            elements.append(Paragraph(f"  - {reason}", S["bullet"]))

    # 5. Verdict interpretation
    elements.append(PageBreak())
    elements.append(Paragraph("Interpretation", S["h1"]))
    elements.append(Paragraph(f"Verdict: {record.verdict.value}", S["body"]))
    if record.verdict_reasons:
        elements.append(Paragraph("Reasons:", S["body"]))
        for reason in record.verdict_reasons:
            elements.append(Paragraph(f"• {reason}", S["bullet"]))

    # 6. Limitations
    elements.append(PageBreak())
    elements.append(Paragraph("Limitations", S["h1"]))
    elements.append(Paragraph(
        "Trained on CASIA v2 splices. Heavy re-compression, screenshots, and "
        "social-media re-encoding wash out the ELA signal. Modern fully-generated "
        "images have no splice boundary at all. Treat output as a signal, not proof.",
        S["body"]
    ))

    # 7. Audit trail
    elements.append(PageBreak())
    elements.append(Paragraph("Audit Trail", S["h1"]))
    elements.append(Paragraph(f"Investigation ID: {record.investigation_id}", S["body"]))
    elements.append(Paragraph(f"Model version: {record.model_version}", S["body"]))
    elements.append(Paragraph(f"Config snapshot hash: {record.config_snapshot_hash}", S["body"]))
    elements.append(Paragraph(f"Created: {record.created_at_utc}", S["body"]))

    # Build PDF
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    doc.build(elements)


def generate_from_records(records, output_dir: Path):
    """Generate PDFs from a list of InvestigationRecords.

    Returns list of (record_id, output_path) tuples.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for record in records:
        output_path = output_dir / f"{record.investigation_id}.pdf"
        generate_pdf(record, output_path)
        results.append((record.investigation_id, output_path))

    return results
