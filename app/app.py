"""Gradio entry point for orchestrated image investigation."""

import os
import sys
from html import escape
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402

from agent.batch import process_batch  # noqa: E402
from app.chat_handler import agent_chat as _agent_chat  # noqa: E402
from app.handler import analyze as _analyze  # noqa: E402

CHECKPOINT = os.environ.get("CHECKPOINT", "checkpoints/best_localizer.keras")
DETECTOR = None
LOAD_ERROR = None
try:
    from src.inference import TamperingDetector

    DETECTOR = TamperingDetector(CHECKPOINT)
except Exception as exc:  # noqa: BLE001 - missing checkpoint is shown in UI
    LOAD_ERROR = str(exc)


def analyze(image, threshold):
    """Run investigation through the orchestrator and render its record."""
    return _analyze(image, threshold, detector=DETECTOR)


def chat(message, image, state):
    """Route a chat turn through the testable chat handler."""
    result = _agent_chat(message, image, state, detector=DETECTOR)
    return (
        render_chat_card(result),
        result["tool_log"],
        result["ela_image_path"],
        result.get("_state", state),
    )


def render_chat_card(data):
    """Render the structured chat reply as an evidence card."""
    verdict = data.get("verdict") or "Inconclusive"
    icon = {
        "Flagged for review": "🚩",
        "Flagged, region unavailable": "⚠️",
        "Not flagged by this detector": "✅",
        "Inconclusive": "❔",
    }.get(verdict, "❔")
    score = data.get("tampered_score_uncalibrated")
    score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "Unavailable"
    image = data.get("image") or {}
    width, height = image.get("width"), image.get("height")
    size_text = (
        f"{width} × {height} px"
        if width is not None and height is not None
        else "Unavailable"
    )
    duration = data.get("processing_seconds")
    duration_text = f"{duration:.3f} s" if isinstance(duration, (int, float)) else "Unavailable"
    region = data.get("region")
    if region:
        horizontal = region["horizontal"]
        vertical = region["vertical"]
        region_html = (
            '<h3>Suspected tampered region</h3><table><thead><tr>'
            '<th>Axis</th><th>From (px)</th><th>To (px)</th></tr></thead><tbody>'
            f'<tr><td>Horizontal</td><td>{escape(str(horizontal[0]))}</td>'
            f'<td>{escape(str(horizontal[1]))}</td></tr>'
            f'<tr><td>Vertical</td><td>{escape(str(vertical[0]))}</td>'
            f'<td>{escape(str(vertical[1]))}</td></tr></tbody></table>'
        )
        coverage = data.get("coverage_percent")
        if isinstance(coverage, (int, float)):
            region_html += f'<p>Covers about {coverage:g}%</p>'
    else:
        region_html = "<h3>Suspected tampered region</h3><p>No region localized</p>"
    answer = escape(str(data.get("answer_text") or "")).replace("\n", "<br>")
    caveat = escape(str(data.get("caveat") or "")).replace("\n", "<br>")
    return (
        '<article class="evidence-card">'
        f'<h2>{icon} {escape(verdict)}</h2>'
        '<table><tbody>'
        f'<tr><th>Verdict</th><td>{escape(verdict)}</td></tr>'
        f'<tr><th>Tampered score (uncalibrated)</th><td>{score_text}</td></tr>'
        f'<tr><th>Image size</th><td>{size_text}</td></tr>'
        f'<tr><th>Processing time</th><td>{duration_text}</td></tr>'
        f'</tbody></table>{region_html}'
        f'<blockquote>{caveat}</blockquote>'
        f'<p>{answer}</p></article>'
    )


def analyze_uploads(files, threshold, goal, custom_text):
    """Run each uploaded image through the same investigation handler."""
    if not files:
        return [], [], [], "Upload one or more images to begin."
    if not isinstance(files, list):
        files = [files]
    originals, ela_maps, overlays, details = [], [], [], []
    for uploaded in files:
        path = getattr(uploaded, "name", uploaded)
        try:
            with Image.open(path) as source:
                original = source.convert("RGB")
            original.filename = Path(path).name
            ela, _scores, overlay, detail = _analyze(
                original,
                threshold,
                detector=DETECTOR,
                goal=goal,
                custom_text=custom_text,
            )
            caption = escape_filename(Path(path).name)
            originals.append((original, caption))
            if ela is not None:
                ela_maps.append((ela, caption))
            if overlay is not None:
                overlay_caption = f"{caption} — Model-predicted tampered region (ELA-based)"
                overlays.append((overlay, overlay_caption))
            details.append(f"## {caption}\n\n{detail}")
        except Exception as exc:  # one malformed upload must not hide other results
            details.append(f"## Upload failed\n\n{type(exc).__name__}: {escape_filename(str(exc))}")
    return originals, ela_maps, overlays, "\n\n---\n\n".join(details)


def escape_filename(value):
    """Escape filenames before rendering them in Markdown captions."""
    from agent.rendering import escape_untrusted

    return escape_untrusted(value)


def analyze_batch(zip_file):
    """Validate and unpack an uploaded image archive through the batch API."""
    if zip_file is None:
        return "Upload a ZIP file to begin."
    zip_path = getattr(zip_file, "name", zip_file)
    result = process_batch(str(zip_path), "artifacts/batch", detector=DETECTOR)
    lines = [f"Batch status: {result['status']}"]
    if result.get("error"):
        lines.append(f"Error: {result['error']}")
    for item in result.get("results", []):
        verdict = f" ({item['verdict']})" if item.get("verdict") else ""
        lines.append(f"- {item['filename']}: {item['status']}{verdict}")
    if result.get("batch_pdf"):
        lines.append(f"Batch PDF: {result['batch_pdf']}")
    if not result.get("results") and not result.get("error"):
        lines.append("- No image files found.")
    return "\n".join(lines)


DESCRIPTION = """
# Deepfake / Tampering Region Localizer

Upload one or more images to run the deterministic forensic checks. The investigation
record supplies the verdict, uncalibrated score, and executed-check log.

**Limitations.** Trained on CASIA v2 splices. Heavy re-compression,
screenshots, social-media re-encoding, and fully-generated images may not
produce reliable splice signals. Treat output as a signal, not proof.
"""

with gr.Blocks(title="Deepfake Tampering Localizer") as demo:
    with gr.Tabs():
        with gr.Tab("Analysis"):
            gr.Markdown(DESCRIPTION)
            if DETECTOR is None:
                gr.Markdown(f"> No trained checkpoint loaded: `{LOAD_ERROR}`")
            with gr.Row():
                with gr.Column():
                    image_input = gr.File(
                        type="filepath",
                        file_count="multiple",
                        file_types=["image"],
                        label="Upload one or more images",
                    )
                    goal = gr.Dropdown(
                        choices=["Quick check", "Detailed", "Custom text"],
                        value="Quick check",
                        label="Investigation goal",
                    )
                    custom_goal = gr.Textbox(label="Custom goal", lines=2, max_lines=4)
                    threshold = gr.Slider(
                        0.1,
                        0.9,
                        value=0.5,
                        step=0.05,
                        label="Decision threshold",
                        info="Lower catches more tampering but raises false alarms.",
                    )
                    run = gr.Button("Analyze", variant="primary")
                with gr.Column():
                    original_output = gr.Gallery(label="Original images", columns=2, height="auto")
                    overlay_output = gr.Gallery(
                        label="Model-predicted tampered region (ELA-based)",
                        columns=2,
                        height="auto",
                    )
            with gr.Row():
                ela_output = gr.Gallery(
                    label="ELA maps (checks actually run)", columns=2, height="auto"
                )
                detail_output = gr.Markdown()

            gr.Markdown("## Batch ZIP validation")
            with gr.Row():
                batch_input = gr.File(label="Upload an image ZIP", file_types=[".zip"])
                batch_button = gr.Button("Process batch")
            batch_output = gr.Textbox(label="Batch result", lines=6, interactive=False)
            batch_button.click(analyze_batch, inputs=[batch_input], outputs=[batch_output])

            run.click(
                analyze_uploads,
                inputs=[image_input, threshold, goal, custom_goal],
                outputs=[original_output, ela_output, overlay_output, detail_output],
            )

        with gr.Tab("Chat"):
            gr.Markdown(
                "Ask a question about an image. Follow-up questions can reuse the image above."
            )
            chat_state = gr.State(
                {"history": [], "image_path": None, "image_hash": None, "filename": "image"}
            )
            chat_image = gr.Image(type="filepath", label="Image (optional after first turn)")
            chat_message = gr.Textbox(
                label="Goal or question", placeholder="Is this edited?", lines=2
            )
            chat_submit = gr.Button("Ask", variant="primary")
            with gr.Row():
                with gr.Column():
                    chat_answer = gr.HTML(label="Answer")
                    with gr.Accordion("Tool calls and evidence", open=False):
                        chat_tool_log = gr.JSON(
                            label="Calls, arguments, status, and result preview"
                        )
                with gr.Column():
                    chat_ela = gr.Image(label="ELA (Error Level Analysis)")
            chat_submit.click(
                chat,
                inputs=[chat_message, chat_image, chat_state],
                outputs=[chat_answer, chat_tool_log, chat_ela, chat_state],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
