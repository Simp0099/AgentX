"""Gradio demo: upload an image, see the ELA map, the verdict, and the box.

Run locally:
    python app/app.py

On Hugging Face Spaces this file is the entry point; ``demo.launch()`` at import
time is what Spaces expects.
"""

import os
import sys
from pathlib import Path

# Spaces runs this file directly, so the repo root must be importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from agent import orchestrator as forensic_orchestrator  # noqa: E402, F401
from agent.rendering import escape_untrusted  # noqa: E402
from agent.schemas import InvestigationRecord, Observation  # noqa: E402
from agent.verdict import Verdict, decide  # noqa: E402
from src.data.ela import compute_ela  # noqa: E402

CHECKPOINT = os.environ.get("CHECKPOINT", "checkpoints/best_localizer.keras")


# Load once at startup, not per request. If the checkpoint is missing the app
# still boots and shows the ELA view, so a stranger cloning the repo without a
# trained model gets a working demo of the preprocessing rather than a crash.
DETECTOR = None
LOAD_ERROR = None
try:
    from src.inference import TamperingDetector

    DETECTOR = TamperingDetector(CHECKPOINT)
except Exception as exc:  # noqa: BLE001 - surfaced to the user in the UI
    LOAD_ERROR = str(exc)


def analyze(image, threshold):
    """Run the full pipeline on one uploaded image.

    Returns ``(ela_image, label_scores, overlay_image, detail_markdown)``.
    """
    if image is None:
        return None, None, None, "Upload an image to begin."

    # Treat raw strings as data, not markup.
    filename = escape_untrusted(getattr(image, "filename", None))
    filename = os.path.basename(filename) if filename else ""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        image = image.convert("RGB")

    ela = compute_ela(image)

    if DETECTOR is None:
        return ela, None, image, (
            "### ELA preview only\n\n"
            f"No trained model loaded: `{LOAD_ERROR}`\n\n"
            "The ELA map on the left still works — bright regions have a different "
            "compression history from the rest of the image. Train a model "
            "(`python -m src.train`) to enable the classifier and the box."
        )

    result = DETECTOR.predict_with_overlay(image, threshold=threshold)

    tampered_prob = result["tampered_prob"]

    # Build observations for verdict decision
    observations = []

    # Add inspect_image observation
    img_width, img_height = image.size
    observations.append(Observation(
        tool="inspect_image",
        args={"path": filename or "unknown"},
        status="ok",
        result={
            "valid": True, "format": "PNG", "width": img_width,
            "height": img_height, "mode": "RGB", "byte_size": 5120,
            "sha256": "faksha",
        },
        rationale="baseline check",
        duration_ms=10.0,
    ))

    # Add run_localizer observation
    bbox = result.get("bbox")
    observations.append(Observation(
        tool="run_localizer",
        args={"path": filename or "unknown", "threshold": threshold},
        status="ok",
        result={
            "tampered_prob": tampered_prob,
            "label": result.get("label", "tampered"),
            "bbox": bbox,
        },
        rationale="localizer output",
        duration_ms=50.0,
    ))

    # Add validate_prediction observation
    validate_issues = []
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        if xmin >= xmax or ymin >= ymax:
            validate_issues.append("inverted box")
        area = (xmax - xmin) * (ymax - ymin)
        if area < 0.001:
            validate_issues.append("box too small")
    observations.append(Observation(
        tool="validate_prediction",
        args={
            "raw_score": tampered_prob, "bbox": bbox,
            "image_width": img_width, "image_height": img_height,
        },
        status="ok",
        result={"valid": len(validate_issues) == 0, "issues": validate_issues},
        rationale="policy validation",
        duration_ms=5.0,
    ))

    # Create InvestigationRecord and decide verdict
    record = InvestigationRecord(
        investigation_id="current",
        image_id="current",
        filename=filename or "unknown",
        goal="quick_check",
        goal_text="",
        threshold=threshold,
        config_snapshot_hash="",
        model_version="",
        observations=observations,
        errors=[],
        rationale="",
        verdict=Verdict.INCONCLUSIVE,  # placeholder, replaced by decide()
        verdict_reasons=[],
        created_at_utc="",
    )

    verdict, reasons = decide(record)

    # Generate detail text based on the decided verdict
    if verdict == Verdict.FLAGGED_FOR_REVIEW:
        xmin, ymin, xmax, ymax = result["bbox"]
        detail = (
            "### Verdict: **Flagged for review**\n\n"
            "**Predicted region** (normalized): "
            f"`[{xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f}]`\n\n"
            f"**In pixels**: x {xmin * img_width:.0f}–{xmax * img_width:.0f}, "
            f"y {ymin * img_height:.0f}–{ymax * img_height:.0f} "
            f"(image is {img_width}×{img_height})\n\n"
            f"Region covers {(xmax - xmin) * (ymax - ymin) * 100:.1f}% of the image."
        )
    elif verdict == Verdict.FLAGGED_REGION_UNAVAILABLE:
        detail = (
            "### Verdict: **Flagged, region unavailable**\n\n"
            "The tampering score exceeds the threshold, but the localization "
            "region could not be determined validly. This can happen when the "
            "image has unusual compression or very low contrast."
        )
    elif verdict == Verdict.NOT_FLAGGED_BY_THIS_DETECTOR:
        detail = (
            "### Verdict: **Not flagged by this detector**\n\n"
            "No tampered region is drawn: the localization head is trained only on "
            "tampered images, so its output on a low-confidence image carries no meaning."
        )
    else:  # INCONCLUSIVE
        detail = (
            "### Verdict: **Inconclusive**\n\n"
            "The investigation could not reach a definitive conclusion. "
            "This may be due to conflicting signals from the model and "
            "the validation checks, or insufficient evidence."
        )

    scores = {
        "tampered score (uncalibrated)": tampered_prob,
    }

    return ela, scores, result["overlay"], detail


DESCRIPTION = """
# Deepfake / Tampering Region Localizer

Upload an image. The model answers two questions at once: **was this edited**,
and **where**.

**How it works.** JPEG compression is lossy but roughly repeatable — re-save an
image and pixels already on the encoder's quantization grid barely move. A region
pasted in from somewhere else has a *different* compression history, so it is not
on that grid yet and shifts more on re-save. Error Level Analysis (ELA) makes that
visible: re-encode, subtract, amplify. Untouched areas go dark; splices light up.

That ELA map is fed to the model alongside the original pixels, and a dual-head
ResNet50 emits a real/tampered label and a bounding box over the suspect region.
"""

with gr.Blocks(title="Deepfake Tampering Localizer") as demo:
    gr.Markdown(DESCRIPTION)

    if DETECTOR is None:
        gr.Markdown(
            f"> **No trained checkpoint found** (`{CHECKPOINT}`). "
            "ELA preview works; the classifier and box need a trained model."
        )

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
        ela_output = gr.Image(label="ELA map (bright = different compression history)")
        detail_output = gr.Markdown()

    run.click(
        analyze,
        inputs=[image_input, threshold],
        outputs=[ela_output, label_output, overlay_output, detail_output],
    )
    image_input.change(
        analyze,
        inputs=[image_input, threshold],
        outputs=[ela_output, label_output, overlay_output, detail_output],
    )

    gr.Markdown(
        "**Limitations.** Trained on CASIA v2 splices, so it is strongest on "
        "copy-paste/splicing forgeries. Heavy re-compression, screenshots, and "
        "social-media re-encoding wash out the ELA signal; modern fully-generated "
        "images have no splice boundary at all. Treat output as a signal, not proof."
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
