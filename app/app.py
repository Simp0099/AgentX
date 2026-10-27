"""Gradio entry point for orchestrated image investigation."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402

from agent.batch import process_batch  # noqa: E402
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


def analyze_batch(zip_file):
    """Validate and unpack an uploaded image archive through the batch API."""
    if zip_file is None:
        return "Upload a ZIP file to begin."
    zip_path = getattr(zip_file, "name", zip_file)
    result = process_batch(str(zip_path), "artifacts/batch")
    lines = [f"Batch status: {result['status']}"]
    if result.get("error"):
        lines.append(f"Error: {result['error']}")
    for item in result.get("results", []):
        lines.append(f"- {item['filename']}: {item['status']}")
    if not result.get("results") and not result.get("error"):
        lines.append("- No image files found.")
    return "\n".join(lines)


DESCRIPTION = """
# Deepfake / Tampering Region Localizer

Upload an image to run the deterministic forensic checks. The investigation
record supplies the verdict, uncalibrated score, and executed-check log.

**Limitations.** Trained on CASIA v2 splices. Heavy re-compression,
screenshots, social-media re-encoding, and fully-generated images may not
produce reliable splice signals. Treat output as a signal, not proof.
"""

with gr.Blocks(title="Deepfake Tampering Localizer") as demo:
    gr.Markdown(DESCRIPTION)
    if DETECTOR is None:
        gr.Markdown(f"> No trained checkpoint loaded: `{LOAD_ERROR}`")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload an image")
            threshold = gr.Slider(
                0.1, 0.9, value=0.5, step=0.05,
                label="Decision threshold",
                info="Lower catches more tampering but raises false alarms.",
            )
            run = gr.Button("Analyze", variant="primary")
        with gr.Column():
            overlay_output = gr.Image(label="Predicted tampered region")
            label_output = gr.Label(label="Prediction", num_top_classes=2)
    with gr.Row():
        ela_output = gr.Image(label="ELA map")
        detail_output = gr.Markdown()

    gr.Markdown("## Batch ZIP validation")
    with gr.Row():
        batch_input = gr.File(label="Upload an image ZIP", file_types=[".zip"])
        batch_button = gr.Button("Process batch")
    batch_output = gr.Textbox(label="Batch result", lines=6, interactive=False)
    batch_button.click(analyze_batch, inputs=[batch_input], outputs=[batch_output])

    run.click(analyze, inputs=[image_input, threshold],
              outputs=[ela_output, label_output, overlay_output, detail_output])
    image_input.change(analyze, inputs=[image_input, threshold],
                       outputs=[ela_output, label_output, overlay_output, detail_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
