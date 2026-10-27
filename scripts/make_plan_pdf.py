"""Generate DEEPFAKE_UPGRADE_PLAN.pdf -- the full upgrade plan for turning the
CASIA2 splice localizer into an all-purpose fake-image detector + localizer.

    python scripts/make_plan_pdf.py

Pure reportlab, no pandoc/LaTeX. Content lives in PLAN below as (kind, text)
pairs so editing the doc never means touching layout code.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parent.parent / "DEEPFAKE_UPGRADE_PLAN.pdf"

ACCENT = colors.HexColor("#1a4d7a")
CODE_BG = colors.HexColor("#f2f2f0")
CODE_BORDER = colors.HexColor("#d8d8d4")
WARN_BG = colors.HexColor("#fdf3e3")

# ── styles ────────────────────────────────────────────────────────────────────

_base = getSampleStyleSheet()

S = {
    "title": ParagraphStyle("title", parent=_base["Title"], fontSize=24, leading=29,
                            textColor=ACCENT, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", parent=_base["Normal"], fontSize=12,
                               leading=16, alignment=TA_CENTER,
                               textColor=colors.HexColor("#555555"), spaceAfter=18),
    "h1": ParagraphStyle("h1", parent=_base["Heading1"], fontSize=16, leading=20,
                         textColor=ACCENT, spaceBefore=16, spaceAfter=8),
    "h2": ParagraphStyle("h2", parent=_base["Heading2"], fontSize=12.5, leading=16,
                         textColor=colors.HexColor("#23272b"), spaceBefore=12,
                         spaceAfter=5),
    "body": ParagraphStyle("body", parent=_base["Normal"], fontSize=9.7, leading=14.2,
                           spaceAfter=6),
    "bullet": ParagraphStyle("bullet", parent=_base["Normal"], fontSize=9.7,
                             leading=14.2, leftIndent=14, bulletIndent=4,
                             spaceAfter=3),
    "code": ParagraphStyle("code", parent=_base["Code"], fontName="Courier",
                           fontSize=7.4, leading=9.4, backColor=CODE_BG,
                           borderColor=CODE_BORDER, borderWidth=0.5,
                           borderPadding=6, leftIndent=2, spaceBefore=4,
                           spaceAfter=8),
    "caption": ParagraphStyle("caption", parent=_base["Normal"], fontSize=8.4,
                              leading=11.5, textColor=colors.HexColor("#666666"),
                              spaceAfter=10),
    "callout": ParagraphStyle("callout", parent=_base["Normal"], fontSize=9.4,
                              leading=13.5, backColor=WARN_BG,
                              borderColor=colors.HexColor("#e6cfa1"),
                              borderWidth=0.5, borderPadding=7, spaceBefore=4,
                              spaceAfter=10),
}

TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.2),
    ("LEADING", (0, 0), (-1, -1), 11),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
])


def _cells(rows, widths):
    """Rows of strings -> Table with wrapped Paragraph cells."""
    head = ParagraphStyle("th", parent=S["body"], fontSize=8.2, leading=11,
                          textColor=colors.white, fontName="Helvetica-Bold",
                          spaceAfter=0)
    cell = ParagraphStyle("td", parent=S["body"], fontSize=8.2, leading=11,
                          spaceAfter=0)
    data = [[Paragraph(c, head if r == 0 else cell) for c in row]
            for r, row in enumerate(rows)]
    table = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    table.setStyle(TABLE_STYLE)
    return table


# ── content ───────────────────────────────────────────────────────────────────
# ("h1"|"h2"|"body"|"bullet"|"code"|"callout"|"caption", text)
# ("table", (rows, col_widths_cm)) | ("break", None) | ("space", points)

PLAN = [
    ("title", "From Splice Localizer to All-Purpose Fake Image Detector"),
    ("subtitle", "Upgrade plan for the ELA + ResNet50 tampering localizer<br/>"
                 "Target: detect and localize splicing, face-swap deepfakes, and "
                 "fully AI-generated images"),

    ("h1", "0. Read this first"),
    ("body", "Your model is not broken. It is trained on the wrong thing for the goal "
             "you now have. Everything below is additive: the existing CASIA2 "
             "checkpoint keeps its job and becomes one expert inside a larger system."),
    ("body", "The plan has three phases. <b>Phase 1 is the one that matters</b> - it "
             "takes a few hours, needs no GPU, and gets you a detector that actually "
             "flags real internet deepfakes. Phases 2 and 3 are optional depth."),

    ("table", ([
        ["Phase", "What you get", "Effort", "Needs GPU"],
        ["<b>1. Ensemble</b>",
         "Pretrained SigLIP2 deepfake/AI classifier fused with your ELA localizer. "
         "Handles all three forgery families. Face-box and Grad-CAM localization.",
         "~4-6 hours,<br/>~600 MB download", "No"],
        ["<b>2. Retrain</b>",
         "Fine-tune your own dual-head model on deepfake data with real forgery masks "
         "(OpenForensics), so your model localizes deepfakes natively.",
         "1-2 days + training time", "Yes (Colab ok)"],
        ["<b>3. Distill</b>",
         "Collapse the ensemble into a single deployable model. Portfolio-grade "
         "story: one model, three forgery types, one box.",
         "2-3 days", "Yes"],
    ], [2.2, 8.6, 3.3, 2.4])),
    ("caption", "Stop after Phase 1 if the goal is a working tool. Do Phase 2 if the "
                "goal is a research/portfolio artifact that is yours end to end."),

    ("h1", "1. Diagnosis: why the current model scores deepfakes as authentic"),
    ("body", "The screenshots got this right. Restating it precisely, because every "
             "design choice below follows from it:"),
    ("body", "<b>ELA measures compression history discontinuity.</b> Re-save a JPEG at "
             "quality 90 and pixels already on the encoder quantization grid barely "
             "move. A region pasted in from another image has a different compression "
             "history, is not on that grid, and shifts more. Subtract, amplify, and the "
             "splice lights up. That is the entire premise of your preprocessing, and "
             "it is sound."),
    ("body", "<b>A deepfake has no splice boundary.</b> A face-swap is blended and then "
             "the whole frame is re-encoded once, so the manipulated region shares its "
             "compression history with everything around it. A fully generated image "
             "(SDXL, Flux, Midjourney) was never composited at all - it is coherent "
             "from the first pixel. ELA sees a uniformly dark map and your model, "
             "correctly given its training set, says authentic."),

    ("table", ([
        ["Forgery family", "Physical trace", "Does ELA see it?", "Your model today"],
        ["Splicing / copy-paste<br/>(CASIA2, IMD2020)",
         "Mismatched JPEG quantization at the paste boundary",
         "<b>Yes</b> - this is exactly what it measures", "Works, plus a box"],
        ["Face-swap deepfake<br/>(FF++, Celeb-DF, DFDC)",
         "Blending seams, warped facial geometry, GAN upsampling checkerboard, "
         "eye/teeth artifacts",
         "Weakly, and only before re-encoding", "Misses - reads as authentic"],
        ["Fully AI-generated<br/>(SDXL, Flux, MJ, SD3)",
         "Generator frequency fingerprint across the whole image, no local anomaly",
         "<b>No</b> - nothing is locally anomalous", "Misses - reads as authentic"],
        ["Re-encoded anything<br/>(screenshots, WhatsApp)",
         "Original traces overwritten by the new encode",
         "Signal destroyed", "Unreliable in both directions"],
    ], [3.6, 5.2, 3.9, 3.8])),

    ("callout", "<b>The consequence that decides the architecture:</b> no single "
                "feature space covers all three families. Compression forensics finds "
                "splices and is blind to generation. Learned semantic features find "
                "generation and are weak on a clean local paste. So the system needs "
                "both, run in parallel, with an explicit fusion rule. Not one bigger "
                "model - two specialists and a referee."),

    ("break", None),

    ("h1", "2. Target architecture"),
    ("code", """                          ┌──────────────────┐
                          │  uploaded image  │
                          └────────┬─────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
     ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
     │ EXPERT A        │  │ EXPERT B        │  │ EXPERT C        │
     │ generation      │  │ splice          │  │ provenance      │
     │                 │  │                 │  │                 │
     │ SigLIP2 ViT     │  │ YOUR MODEL      │  │ C2PA manifest   │
     │ (pretrained)    │  │ ResNet50 + ELA  │  │ EXIF / software │
     │                 │  │ (existing ckpt) │  │ tag             │
     │ p(artificial)   │  │ p(tampered)     │  │ hard evidence   │
     │ p(deepfake)     │  │ bbox            │  │ (rare, but      │
     │ p(real)         │  │                 │  │  decisive)      │
     └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                       ┌───────────▼───────────┐
                       │   FUSION (noisy-OR)   │
                       │   + verdict routing   │
                       └───────────┬───────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
      verdict = spliced    verdict = deepfake    verdict = ai_generated
              │                    │                    │
      ┌───────▼──────┐    ┌────────▼───────┐    ┌───────▼────────┐
      │ ELA bbox     │    │ face boxes     │    │ whole image +  │
      │ (your head)  │    │ + Grad-CAM     │    │ Grad-CAM heat  │
      └──────────────┘    └────────────────┘    └────────────────┘"""),
    ("caption", "Each expert answers the question it can actually answer. The fusion "
                "layer is ten lines of arithmetic, not a learned model - which means "
                "you can explain and tune every number in it."),

    ("h1", "3. Phase 1, Step 0: environment"),
    ("body", "Three new dependencies. PyTorch is CPU/MPS only here - the SigLIP2 model "
             "is 93M parameters and runs in roughly 0.3-1 s per image on Apple Silicon "
             "without CUDA."),
    ("code", """# from the Model/ directory, in your existing venv
pip install torch transformers opencv-python

# verify Apple Silicon acceleration is available (should print True)
python -c "import torch; print(torch.backends.mps.is_available())"

# warm the model cache once (~370 MB, downloads to ~/.cache/huggingface)
python -c "from transformers import SiglipForImageClassification as M; \\
           M.from_pretrained('prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0')" """),
    ("body", "Append to <font face='Courier'>requirements.txt</font>:"),
    ("code", """# Pretrained generation/deepfake expert (Phase 1)
torch>=2.4
transformers>=4.45
opencv-python>=4.10"""),
    ("callout", "<b>TensorFlow and PyTorch in one process.</b> This works, but both "
                "libraries grab memory eagerly. If you hit allocation errors, set "
                "<font face='Courier'>TF_FORCE_GPU_ALLOW_GROWTH=true</font> and load "
                "the Torch model before the Keras one. On a 16 GB Mac the combined "
                "footprint is around 3.5 GB, which is fine."),

    ("h1", "4. Phase 1, Step 1: the generation expert"),
    ("body", "New file: <font face='Courier'>src/detectors/__init__.py</font> (empty) "
             "and <font face='Courier'>src/detectors/generated.py</font>."),
    ("body", "Model choice, with reasoning. "
             "<font face='Courier'>prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0</font> is "
             "a SigLIP2 base fine-tune with a <b>three-way</b> head - Artificial, "
             "Deepfake, Real - reporting 99.15 percent eval accuracy. The three-way "
             "split is the reason to pick it over a stronger binary model: it "
             "distinguishes 'this whole image was generated' from 'a real photo had its "
             "face swapped', and those two need different localization strategies. "
             "Treat the 99 percent as in-distribution only; expect materially lower on "
             "images from generators it never saw."),
    ("code", '''"""Pretrained generation/deepfake expert.

Answers the question ELA cannot: was this image synthesized, rather than
locally edited. Wraps a SigLIP2 classifier so the rest of the codebase never
imports transformers directly.
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

MODEL_ID = "prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0"


def _pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class GeneratedImageDetector:
    """Three-way classifier: artificial / deepfake / real."""

    def __init__(self, model_id=MODEL_ID, device=None):
        self.device = device or _pick_device()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = (SiglipForImageClassification
                      .from_pretrained(model_id).to(self.device).eval())
        # Read labels off the config, never hardcode them: this card uses
        # "Artificial" in label2id but calls it "AI" in the prose, and a
        # silently mismatched key would make the fusion rule read zeros.
        self.id2label = {i: v.lower()
                         for i, v in self.model.config.id2label.items()}

    def _to_pil(self, image):
        if isinstance(image, np.ndarray):
            return Image.fromarray(image.astype(np.uint8)).convert("RGB")
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        return image.convert("RGB")

    @torch.no_grad()
    def predict(self, image):
        """-> {"artificial": p, "deepfake": p, "real": p}, summing to 1."""
        inputs = self.processor(images=self._to_pil(image),
                                return_tensors="pt").to(self.device)
        probs = self.model(**inputs).logits.softmax(-1)[0].cpu().numpy()
        return {self.id2label[i]: float(p) for i, p in enumerate(probs)}


if __name__ == "__main__":
    # Self-check: probabilities are a valid distribution over the 3 expected
    # classes. Catches a model swap that silently changes the label set.
    d = GeneratedImageDetector()
    out = d.predict(Image.new("RGB", (400, 400), (128, 130, 127)))
    assert set(out) == {"artificial", "deepfake", "real"}, out
    assert abs(sum(out.values()) - 1.0) < 1e-4, out
    print("ok ", {k: round(v, 3) for k, v in out.items()}, d.device)'''),

    ("h1", "5. Phase 1, Step 2: face localization"),
    ("body", "A face-swap deepfake is, by definition, localized to a face. You do not "
             "need a trained box head for that case - you need a face detector, and "
             "OpenCV ships one in its wheel with no extra download."),
    ("body", "New file: <font face='Courier'>src/detectors/faces.py</font>."),
    ("code", '''"""Face boxes, in the project's normalized [xmin, ymin, xmax, ymax] form.

A face-swap manipulation IS the face region, so for deepfake verdicts the face
box is the localization -- no trained box head required. Haar cascades are old
and miss profiles, but they ship inside opencv-python with no extra weights,
which is the right trade for a fallback localizer.
"""

import cv2
import numpy as np
from PIL import Image

_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(image, min_size=48, pad=0.12):
    """-> list of normalized (xmin, ymin, xmax, ymax), largest first.

    ``pad`` expands each box, because swap artifacts live at the blend seam
    along the jaw and hairline -- just outside a tight face crop.
    """
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)[..., :3]
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

    found = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                      minSize=(min_size, min_size))
    height, width = gray.shape[:2]
    boxes = []
    for x, y, box_w, box_h in found:
        dx, dy = box_w * pad, box_h * pad
        boxes.append((
            max(0.0, (x - dx) / width),
            max(0.0, (y - dy) / height),
            min(1.0, (x + box_w + dx) / width),
            min(1.0, (y + box_h + dy) / height),
        ))
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes


if __name__ == "__main__":
    # Self-check: every returned box is inside the frame and has real area.
    # A negative-area box would silently poison IoU downstream.
    import urllib.request

    url = ("https://upload.wikimedia.org/wikipedia/commons/thumb/"
           "9/99/Lena_Forsen.png/256px-Lena_Forsen.png")
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            image = Image.open(response).convert("RGB")
    except Exception:
        image = Image.new("RGB", (256, 256), (200, 180, 160))

    for box in detect_faces(image):
        xmin, ymin, xmax, ymax = box
        assert 0.0 <= xmin < xmax <= 1.0 and 0.0 <= ymin < ymax <= 1.0, box
    print("ok  faces:", len(detect_faces(image)))'''),
    ("callout", "<b>Upgrade path, when Haar is not enough.</b> It only finds roughly "
                "frontal faces. If your test images are profiles, groups, or low light, "
                "swap the implementation for RetinaFace "
                "(<font face='Courier'>pip install retina-face</font>) or MediaPipe "
                "Face Detection. Keep the function signature identical and nothing "
                "downstream changes."),

    ("break", None),

    ("h1", "6. Phase 1, Step 3: Grad-CAM heatmap for generated images"),
    ("body", "For a fully generated image there is no region to box - the whole frame "
             "is fake. But 'the whole image' is a useless answer for a tool called a "
             "localizer, and users want to see <i>why</i>. Grad-CAM over the ViT patch "
             "grid gives a heatmap of which patches drove the fake logit. It is "
             "attribution rather than ground-truth localization; label it that way in "
             "the UI and it is genuinely informative."),
    ("body", "New file: <font face='Courier'>src/explain.py</font>."),
    ("code", '''"""Grad-CAM over the SigLIP2 patch grid -> where the fake evidence is.

This is attribution, not localization: it shows which image patches pushed the
logit, not a ground-truth forged region. Presented honestly in the UI it is
useful; presented as a box it would be a lie.
"""

import numpy as np
import torch
from PIL import Image


def gradcam(detector, image, target_label=None, out_size=None):
    """-> (heatmap HxW float32 in [0,1], label the map explains).

    ``detector`` is a GeneratedImageDetector. Hooks the last vision encoder
    layer, backprops the target logit, and weights patch activations by their
    mean gradient -- standard Grad-CAM, adapted from CNN maps to the ViT
    patch sequence.
    """
    model, processor = detector.model, detector.processor
    pil = detector._to_pil(image)
    inputs = processor(images=pil, return_tensors="pt").to(detector.device)

    # Layer path is version-dependent. Verify once with:
    #   print(detector.model)
    layer = model.vision_model.encoder.layers[-1]

    cache = {}

    def forward_hook(_module, _inputs, output):
        activation = output[0] if isinstance(output, tuple) else output
        activation.retain_grad()
        cache["activation"] = activation

    handle = layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(**inputs).logits
            if target_label is None:
                index = int(logits[0].argmax())
            else:
                inverse = {v: k for k, v in detector.id2label.items()}
                index = inverse[target_label]
            logits[0, index].backward()
        activation = cache["activation"]          # (1, tokens, channels)
        gradient = activation.grad
    finally:
        handle.remove()

    if gradient is None:
        raise RuntimeError("no gradient captured -- check the encoder layer path")

    weights = gradient.mean(dim=1, keepdim=True)              # (1, 1, C)
    cam = torch.relu((activation * weights).sum(-1))[0]       # (tokens,)
    cam = cam.detach().cpu().numpy()

    # SigLIP2 has no CLS token, so tokens form a perfect square patch grid.
    side = int(round(len(cam) ** 0.5))
    if side * side != len(cam):          # a CLS token snuck in -- drop it
        cam, side = cam[1:], int(round((len(cam) - 1) ** 0.5))
    cam = cam[: side * side].reshape(side, side)

    span = cam.max() - cam.min()
    cam = (cam - cam.min()) / span if span > 1e-8 else np.zeros_like(cam)

    target = out_size or pil.size
    heat = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        target, Image.BICUBIC)
    return np.asarray(heat, dtype=np.float32) / 255.0, detector.id2label[index]


def overlay_heatmap(image, heatmap, alpha=0.45):
    """Red-hot heatmap composited over the image, for display."""
    base = image.convert("RGB")
    heat = np.asarray(Image.fromarray((heatmap * 255).astype(np.uint8))
                      .resize(base.size, Image.BICUBIC), dtype=np.float32) / 255.0
    array = np.asarray(base, dtype=np.float32) / 255.0
    tint = np.stack([heat, heat * 0.25, heat * 0.1], axis=-1)
    blended = array * (1 - alpha * heat[..., None]) + tint * (alpha * heat[..., None] * 2)
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))


def heatmap_to_bbox(heatmap, percentile=90):
    """Tightest normalized box around the hottest region. Coarse by nature."""
    threshold = np.percentile(heatmap, percentile)
    rows = np.any(heatmap >= threshold, axis=1)
    cols = np.any(heatmap >= threshold, axis=0)
    if not rows.any() or not cols.any():
        return None
    ymin, ymax = np.flatnonzero(rows)[[0, -1]]
    xmin, xmax = np.flatnonzero(cols)[[0, -1]]
    height, width = heatmap.shape[:2]
    return (xmin / width, ymin / height, (xmax + 1) / width, (ymax + 1) / height)'''),

    ("break", None),

    ("h1", "7. Phase 1, Step 4: the fusion layer"),
    ("body", "This is the heart of the upgrade and the part worth understanding rather "
             "than pasting. New file: <font face='Courier'>src/fusion.py</font>."),
    ("h2", "The combination rule"),
    ("body", "Two independent experts each report a probability that the image is fake. "
             "Averaging them is wrong: an average lets a confident REAL from the expert "
             "that cannot see this forgery type cancel a confident FAKE from the expert "
             "that can. A clean SDXL image gets p(splice) near 0.02 from ELA, and "
             "averaging drags a correct 0.97 down to 0.50."),
    ("body", "Use <b>noisy-OR</b> instead. Read it as: the image is authentic only if "
             "<i>every</i> expert independently fails to find evidence."),
    ("code", """p_fake = 1 - (1 - p_generation) * (1 - alpha * p_splice)"""),
    ("body", "The <font face='Courier'>alpha</font> term (default 0.85) discounts the "
             "splice expert slightly, because ELA has a well-known false-positive mode: "
             "any strong texture or sharp edge produces a bright ELA response that looks "
             "like a splice. Noisy-OR is aggressive by design, so an undiscounted "
             "false-positive-prone expert would raise the floor on every busy photo. "
             "Tune alpha down if you see clean photographs flagged, up if splices slip "
             "through."),
    ("h2", "Verdict routing"),
    ("body", "The fused probability answers 'is it fake'. A separate argmax over the "
             "individual evidence answers 'what kind', which is what selects the "
             "localization strategy. Keep the two decisions separate - collapsing them "
             "into one softmax is how you end up unable to explain a verdict."),

    ("code", '''"""Fuse the generation expert and the splice expert into one verdict.

The fusion rule is deliberately arithmetic, not learned. Every number here is
inspectable and tunable; a learned fusion head would need a labelled
multi-forgery validation set you do not have yet, and would turn an
explainable system into a second black box.
"""

from dataclasses import dataclass, field

from PIL import Image

from src.detectors.faces import detect_faces
from src.detectors.generated import GeneratedImageDetector
from src.inference import TamperingDetector


@dataclass
class FusionConfig:
    # Splice-expert discount in the noisy-OR. ELA fires on strong texture and
    # sharp edges, so an undiscounted vote raises the floor on busy photos.
    splice_weight: float = 0.85
    # Verdict threshold on the fused probability.
    fake_threshold: float = 0.55
    # Minimum evidence for a family to be named as the verdict type. Below
    # this the system says "manipulated, type unclear" rather than guessing.
    type_threshold: float = 0.35
    ela_checkpoint: str = "checkpoints/best_localizer.keras"
    weights: dict = field(default_factory=dict)


class UniversalDetector:
    """Splicing + face-swap + fully-generated, with a localization for each."""

    def __init__(self, config=None):
        self.config = config or FusionConfig()
        self.generated = GeneratedImageDetector()
        # The ELA expert is optional: a missing checkpoint degrades the system
        # to generation-only rather than refusing to start.
        try:
            self.splice = TamperingDetector(self.config.ela_checkpoint)
        except Exception as exc:            # noqa: BLE001 - surfaced in output
            self.splice, self.splice_error = None, str(exc)
        else:
            self.splice_error = None

    def predict(self, image, want_heatmap=True):
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        cfg = self.config

        gen = self.generated.predict(image)
        p_artificial = gen.get("artificial", 0.0)
        p_deepfake = gen.get("deepfake", 0.0)
        p_generation = max(p_artificial, p_deepfake)

        if self.splice is not None:
            splice_out = self.splice.predict(image)
            p_splice = splice_out["tampered_prob"]
            splice_bbox = splice_out["bbox"]
        else:
            p_splice, splice_bbox = 0.0, None

        # Noisy-OR: authentic only if EVERY expert finds nothing. Averaging
        # here would let the blind expert veto the one that can see.
        p_fake = 1.0 - (1.0 - p_generation) * (1.0 - cfg.splice_weight * p_splice)

        evidence = {
            "ai_generated": p_artificial,
            "deepfake": p_deepfake,
            "spliced": cfg.splice_weight * p_splice,
        }
        kind, kind_score = max(evidence.items(), key=lambda kv: kv[1])
        is_fake = p_fake >= cfg.fake_threshold
        if not is_fake:
            verdict = "authentic"
        elif kind_score < cfg.type_threshold:
            verdict = "manipulated (type unclear)"
        else:
            verdict = kind

        result = {
            "verdict": verdict,
            "fake_probability": p_fake,
            "confidence": p_fake if is_fake else 1.0 - p_fake,
            "evidence": {**evidence, "real_score": gen.get("real", 0.0)},
            "experts": {"generation": gen, "splice_prob": p_splice},
            "boxes": [],
            "box_meaning": None,
            "heatmap": None,
            "image": image,
            "notes": [],
        }
        if self.splice_error:
            result["notes"].append(f"ELA expert unavailable: {self.splice_error}")

        self._localize(result, image, verdict, splice_bbox, want_heatmap)
        return result

    def _localize(self, result, image, verdict, splice_bbox, want_heatmap):
        """Pick the localization strategy the verdict actually supports."""
        if verdict == "authentic":
            return

        if verdict == "spliced" and splice_bbox is not None:
            result["boxes"] = [splice_bbox]
            result["box_meaning"] = "ELA-localized spliced region"
            return

        if verdict == "deepfake":
            faces = detect_faces(image)
            if faces:
                result["boxes"] = faces
                result["box_meaning"] = "face region (swap target)"
            else:
                result["notes"].append(
                    "Deepfake signal without a detected face -- Haar cascades miss "
                    "profiles and low light. Falling back to the heatmap.")

        if want_heatmap and verdict in ("ai_generated", "deepfake",
                                        "manipulated (type unclear)"):
            try:
                from src.explain import gradcam
                heat, explains = gradcam(self.generated, image)
                result["heatmap"] = heat
                result["heatmap_explains"] = explains
                if not result["boxes"]:
                    result["box_meaning"] = (
                        "attribution heatmap -- which patches drove the score, "
                        "NOT a ground-truth forged region")
            except Exception as exc:        # noqa: BLE001
                result["notes"].append(f"heatmap unavailable: {exc}")


if __name__ == "__main__":
    # Self-check on the fusion arithmetic, which is the part that can be
    # silently wrong: a blind expert must never veto a confident one.
    cfg = FusionConfig()

    def fuse(p_gen, p_spl):
        return 1.0 - (1.0 - p_gen) * (1.0 - cfg.splice_weight * p_spl)

    # Confident generation + blind ELA must stay confident (averaging gives .50)
    assert fuse(0.97, 0.02) > 0.95, fuse(0.97, 0.02)
    # Confident splice + blind generation must stay confident
    assert fuse(0.03, 0.96) > 0.80, fuse(0.03, 0.96)
    # Both silent must stay low
    assert fuse(0.02, 0.03) < 0.10
    # Monotonic in both arguments
    assert fuse(0.5, 0.5) > fuse(0.4, 0.5) and fuse(0.5, 0.5) > fuse(0.5, 0.4)
    print("ok  fusion arithmetic")'''),

    ("break", None),

    ("h1", "8. Phase 1, Step 5: wire it into the app"),
    ("body", "Your existing <font face='Courier'>app/app.py</font> calls "
             "<font face='Courier'>TamperingDetector</font> directly. Swap in "
             "<font face='Courier'>UniversalDetector</font> and widen the output panel. "
             "The important UI change is honesty about what each localization means - "
             "an ELA box and an attribution heatmap are not the same claim, and showing "
             "them identically is the fastest way to make the tool untrustworthy."),
    ("code", '''# app/app.py -- replace the detector construction
from src.fusion import UniversalDetector, FusionConfig

DETECTOR = UniversalDetector(FusionConfig(ela_checkpoint=CHECKPOINT))


def analyze(image, threshold):
    if image is None:
        return None, None, None, None, "Upload an image to begin."

    image = (Image.fromarray(image.astype(np.uint8)).convert("RGB")
             if isinstance(image, np.ndarray) else image.convert("RGB"))

    DETECTOR.config.fake_threshold = threshold
    result = DETECTOR.predict(image)

    scores = {
        "AI-generated": result["evidence"]["ai_generated"],
        "Deepfake (face swap)": result["evidence"]["deepfake"],
        "Spliced / edited": result["evidence"]["spliced"],
        "Authentic": 1.0 - result["fake_probability"],
    }

    overlay = result["image"]
    if result["boxes"]:
        from src.visualize import draw_boxes
        overlay = draw_boxes(result["image"], result["boxes"],
                             labels=[result["box_meaning"]] * len(result["boxes"]))
    elif result["heatmap"] is not None:
        from src.explain import overlay_heatmap
        overlay = overlay_heatmap(result["image"], result["heatmap"])

    lines = [
        f"### Verdict: **{result['verdict']}** "
        f"({result['confidence'] * 100:.1f}% confident)",
        "",
        f"- Fake probability (fused): `{result['fake_probability']:.3f}`",
        f"- AI-generated evidence: `{result['evidence']['ai_generated']:.3f}`",
        f"- Face-swap evidence: `{result['evidence']['deepfake']:.3f}`",
        f"- Splice evidence (ELA): `{result['evidence']['spliced']:.3f}`",
    ]
    if result["box_meaning"]:
        lines += ["", f"**Overlay shows:** {result['box_meaning']}"]
    lines += ["", *(f"> {note}" for note in result["notes"])]

    return compute_ela(image), scores, overlay, "\\n".join(lines)'''),
    ("callout", "<b>Do not delete the ELA panel.</b> It is the most explainable thing "
                "in the whole system - a viewer can look at the map and see the splice "
                "themselves. Nothing the neural experts produce is that legible."),

    ("h1", "9. Phase 1, Step 6: tests"),
    ("body", "Add <font face='Courier'>tests/test_fusion.py</font>. The fusion "
             "arithmetic is the only genuinely new logic, so it is the only thing that "
             "needs real test coverage - the experts are pretrained and your ELA model "
             "already has tests."),
    ("code", '''import pytest

from src.fusion import FusionConfig


def fuse(p_gen, p_spl, cfg=FusionConfig()):
    return 1.0 - (1.0 - p_gen) * (1.0 - cfg.splice_weight * p_spl)


def test_blind_expert_never_vetoes_confident_one():
    """The defect this system exists to avoid: an expert that cannot see a
    given forgery type must not drag down one that can."""
    assert fuse(0.97, 0.02) > 0.95      # SDXL image: ELA is blind
    assert fuse(0.03, 0.96) > 0.80      # clean splice: generation is blind


def test_both_silent_stays_low():
    assert fuse(0.02, 0.03) < 0.10


@pytest.mark.parametrize("p_gen,p_spl", [(0.1, 0.1), (0.5, 0.5), (0.8, 0.2)])
def test_monotonic(p_gen, p_spl):
    assert fuse(p_gen + 0.05, p_spl) > fuse(p_gen, p_spl)
    assert fuse(p_gen, p_spl + 0.05) > fuse(p_gen, p_spl)


def test_bounded():
    assert 0.0 <= fuse(0.0, 0.0) and fuse(1.0, 1.0) <= 1.0


@pytest.mark.slow
def test_end_to_end_smoke():
    """Downloads ~370 MB on first run. Run with: pytest -m slow"""
    from PIL import Image
    from src.fusion import UniversalDetector

    result = UniversalDetector().predict(Image.new("RGB", (512, 512), (120, 120, 118)))
    assert 0.0 <= result["fake_probability"] <= 1.0
    assert result["verdict"] in {
        "authentic", "ai_generated", "deepfake", "spliced",
        "manipulated (type unclear)",
    }'''),

    ("break", None),

    ("h1", "10. Phase 1, Step 7: calibrate before you trust it"),
    ("body", "This step is skippable and skipping it is the single most common way "
             "these projects end up confidently wrong. The default thresholds are "
             "guesses. Twenty minutes of collecting images turns them into measurements."),
    ("body", "Build a small benchmark - about 20 images per row below is enough to see "
             "gross miscalibration - and store it as "
             "<font face='Courier'>benchmark/&lt;category&gt;/*.jpg</font>."),

    ("table", ([
        ["Category", "Where to get it", "Expected verdict"],
        ["Real photos, your own camera", "Your phone, unedited, straight off the device",
         "authentic"],
        ["Real photos, re-encoded", "Same photos sent through WhatsApp or screenshotted",
         "authentic (this is the hard false-positive case)"],
        ["AI-generated", "Midjourney/SDXL/Flux showcase galleries, "
                         "thisxdoesnotexist.com", "ai_generated"],
        ["Face-swap deepfakes", "Known deepfake clips, frame-grabbed; "
                                "Celeb-DF or FF++ samples", "deepfake"],
        ["Spliced", "Your CASIA2 test split, held out from training",
         "spliced"],
    ], [3.6, 8.0, 4.9])),

    ("body", "Then sweep the two knobs and read the confusion matrix:"),
    ("code", '''# scripts/calibrate.py
import json
from pathlib import Path

from src.fusion import FusionConfig, UniversalDetector

BENCH = Path("benchmark")
detector = UniversalDetector()

# Cache expert outputs once, then sweep thresholds for free -- re-running the
# models per threshold would turn a 2-minute sweep into an hour.
cache = []
for folder in sorted(p for p in BENCH.iterdir() if p.is_dir()):
    for image_path in sorted(folder.glob("*.[jp][pn]g")):
        result = detector.predict(image_path, want_heatmap=False)
        cache.append({
            "truth": folder.name,
            "p_fake": result["fake_probability"],
            "evidence": result["evidence"],
        })
        print(f"{folder.name:22s} {image_path.name:34s} "
              f"p_fake={result['fake_probability']:.3f} -> {result['verdict']}")

Path("benchmark_raw.json").write_text(json.dumps(cache, indent=2))

for threshold in [0.35, 0.45, 0.55, 0.65, 0.75]:
    true_pos = sum(r["p_fake"] >= threshold for r in cache
                   if r["truth"] != "real")
    false_pos = sum(r["p_fake"] >= threshold for r in cache
                    if r["truth"] == "real")
    n_fake = sum(r["truth"] != "real" for r in cache)
    n_real = sum(r["truth"] == "real" for r in cache) or 1
    print(f"thr={threshold:.2f}  recall={true_pos / max(n_fake, 1):.2%}  "
          f"false-alarm={false_pos / n_real:.2%}")'''),
    ("callout", "<b>Pick the threshold from the false-alarm rate, not the accuracy.</b> "
                "A detector that flags 10 percent of genuine family photos as fake is "
                "worse than useless - people stop believing it, including when it is "
                "right. Published open-source detectors run 6-11 percent false positives "
                "on real photos; if yours is much better than that on a 20-image set, "
                "your benchmark is too easy, not your model too good."),

    ("break", None),

    ("h1", "11. Phase 2 (optional): retrain for native deepfake localization"),
    ("body", "After Phase 1 the system detects all three families but only "
             "<i>localizes</i> splices natively - deepfakes get a face box, generated "
             "images get a heatmap. Phase 2 fixes that by training your own dual-head "
             "model on deepfake data that ships forged-region masks."),
    ("h2", "Dataset selection"),
    ("body", "The screenshots pointed at FaceForensics++. It is the standard citation, "
             "but for a <i>localizer</i> it is awkward: it ships videos and per-frame "
             "masks you must extract yourself, the access form takes days, and c40 is "
             "tens of gigabytes against your 63 GB free. OpenForensics is the better "
             "fit - it is images, not video, and it ships per-face forgery segmentation "
             "masks that convert to boxes with the "
             "<font face='Courier'>mask_to_bbox</font> function you already wrote."),

    ("table", ([
        ["Dataset", "Size", "Masks?", "Access", "Verdict for this project"],
        ["<b>OpenForensics</b>", "~30 GB", "<b>Yes</b>, per-face segmentation",
         "Zenodo, direct download",
         "<b>Start here.</b> Only large deepfake set with localization ground truth. "
         "Feeds both your heads unchanged."],
        ["140k Real and Fake Faces", "~4 GB", "No (labels only)", "Kaggle, direct",
         "Cheap classification-head boost. StyleGAN-era, so weak on diffusion. "
         "Good smoke-test dataset."],
        ["FaceForensics++ c40", "~40 GB", "Masks per frame, video",
         "Signed form, days of wait",
         "The citation everyone expects. Add for credibility once the pipeline works."],
        ["Celeb-DF v2", "~15 GB", "No", "Signed form",
         "Higher visual quality than FF++. Classification only."],
        ["DFDC", "~470 GB full", "No", "Kaggle", "Full set is out of the question on "
         "63 GB. Use the ~5 GB preview only."],
        ["IMD2020 / DEFACTO", "~10 GB", "<b>Yes</b>", "Direct download",
         "Real-world splices with masks. Best complement to CASIA2 for the "
         "tampering head."],
    ], [3.3, 1.9, 2.6, 2.4, 6.3])),
    ("caption", "Sizes are approximate and change as mirrors are updated. Check free "
                "disk before every download: you have roughly 63 GB, and CASIA2 already "
                "occupies 3.3 GB."),

    ("h2", "Code changes for Phase 2"),
    ("body", "Your dataset module is already close. The changes are additive."),

    ("table", ([
        ["File", "Change"],
        ["<font face='Courier'>src/data/dataset.py</font>",
         "Add <font face='Courier'>build_openforensics_index()</font> returning the same "
         "<font face='Courier'>(path, label, bbox)</font> tuples as "
         "<font face='Courier'>build_index()</font>. Reuse "
         "<font face='Courier'>mask_to_bbox</font> unchanged. Add "
         "<font face='Courier'>build_combined_index()</font> that concatenates sources "
         "and returns a per-record source tag."],
        ["<font face='Courier'>src/data/dataset.py</font>",
         "Add a <font face='Courier'>forgery_type</font> field (0 authentic, 1 splice, "
         "2 deepfake, 3 generated) so the model can learn a 4-way head instead of "
         "binary. This is what makes it a general detector rather than a binary one."],
        ["<font face='Courier'>src/models/localizer.py</font>",
         "Change the label head from <font face='Courier'>Dense(1, sigmoid)</font> to "
         "<font face='Courier'>Dense(4, softmax)</font>; loss becomes "
         "<font face='Courier'>sparse_categorical_crossentropy</font>. The box head, "
         "CornerDecode, and masked_bbox_loss all stay exactly as they are - that work "
         "was done right and carries over."],
        ["<font face='Courier'>src/models/localizer.py</font>",
         "Consider adding a high-frequency branch (SRM filters or a fixed Laplacian "
         "stack) alongside the ELA branch. GAN and diffusion fingerprints live in high "
         "frequency, which is precisely where ELA is weakest."],
        ["<font face='Courier'>src/train.py</font>",
         "Class weights over 4 classes rather than 2. Keep the two-phase frozen then "
         "fine-tune schedule - it is correct and worth keeping in the writeup."],
        ["<font face='Courier'>configs/</font>",
         "New <font face='Courier'>universal.yaml</font> with the dataset mix and per-"
         "source sampling weights. Keep <font face='Courier'>default.yaml</font> so the "
         "CASIA2-only result stays reproducible."],
    ], [4.8, 11.7])),

    ("callout", "<b>Do this on Colab, not your Mac.</b> ResNet50 over roughly 100k "
                "images at 224x224 on Apple Silicon CPU is measured in days per epoch. "
                "A free Colab T4 does it in under an hour per epoch. Mount Drive, "
                "download the dataset there, train, and bring back only the "
                "<font face='Courier'>.keras</font> file."),

    ("h2", "Critical trap: mixed-dataset training"),
    ("body", "When you combine CASIA2 with a deepfake set, the model can hit 99 percent "
             "validation accuracy by learning <i>which dataset an image came from</i> - "
             "resolution, JPEG quality, and camera pipeline differ per source - rather "
             "than learning forgery. This is the most common silent failure in "
             "multi-dataset forensics work, and validation accuracy will not reveal it."),
    ("body", "Three defenses, all cheap:"),
    ("bullet", "Re-encode every image to a common JPEG quality and resolution before "
               "training, so the source signature is destroyed."),
    ("bullet", "Hold out one entire dataset from training and evaluate on it. "
               "Cross-dataset accuracy is the only number that means anything; expect "
               "it to be far below your validation number, and treat that gap as the "
               "real result."),
    ("bullet", "Keep the authentic-to-fake ratio matched <i>within</i> each source, so "
               "'this looks like CASIA' cannot be a shortcut to 'this is fake'."),

    ("break", None),

    ("h1", "12. Free accuracy: provenance checks"),
    ("body", "Before any pixel analysis, two nearly-free checks that are far more "
             "reliable than any classifier when they fire. Neither is a model. Both "
             "belong first in the pipeline."),
    ("body", "<b>C2PA Content Credentials.</b> Adobe, OpenAI, Microsoft, Google, and "
             "recent Leica, Sony, and Nikon cameras embed a signed manifest recording "
             "the capture device or generating model and the edit chain. When present, "
             "this is cryptographic evidence rather than a guess. Check it with "
             "<font face='Courier'>pip install c2pa-python</font>, or manually at "
             "verify.contentauthenticity.org."),
    ("body", "<b>EXIF sanity.</b> A file claiming to be a camera photo with no EXIF at "
             "all, or with a Software tag naming an image generator, is a strong signal. "
             "Weak on its own since social platforms strip EXIF wholesale, which is "
             "exactly why it must be a soft prior rather than a verdict."),
    ("code", '''# src/detectors/provenance.py -- runs before the neural experts
from PIL import Image

GENERATOR_HINTS = ("stable diffusion", "midjourney", "dall-e", "dalle",
                   "flux", "firefly", "imagen", "novelai", "comfyui",
                   "automatic1111", "invokeai")


def provenance_signals(path):
    """-> dict of hard/soft provenance evidence. Never a verdict on its own."""
    signals = {"c2pa": None, "exif_software": None, "has_exif": False,
               "generator_hint": False}
    try:
        exif = Image.open(path).getexif()
        signals["has_exif"] = bool(exif)
        software = str(exif.get(305, "")).strip()   # tag 305 = Software
        signals["exif_software"] = software or None
        signals["generator_hint"] = any(h in software.lower()
                                        for h in GENERATOR_HINTS)
    except Exception:                                # noqa: BLE001
        pass

    try:
        import c2pa
        signals["c2pa"] = c2pa.read_file(str(path), None)
    except Exception:                                # noqa: BLE001
        pass                                         # absent is the normal case
    return signals'''),
    ("callout", "<b>Asymmetry to respect.</b> Present credentials prove provenance. "
                "Absent credentials prove nothing at all - almost no image on the "
                "internet carries them. Never let a missing manifest raise your fake "
                "score, only let a present one settle the question."),

    ("h1", "13. Limitations to state plainly in your README"),
    ("body", "Every one of these is real, and writing them down is what separates a "
             "credible project from an overclaiming one. A reviewer who finds a "
             "limitation you did not disclose discounts everything else you claimed."),
    ("bullet", "<b>Generator generalization.</b> Detectors degrade sharply on "
               "generators absent from their training data. A model trained through "
               "SDXL will read weaker on whatever ships next year. Retraining is "
               "periodic maintenance, not a one-time step."),
    ("bullet", "<b>Re-encoding destroys forensic traces.</b> Screenshots, social media "
               "recompression, and messaging apps wash out ELA almost entirely and "
               "degrade the learned experts. Most images in the wild have been through "
               "at least one of these."),
    ("bullet", "<b>Heatmaps are attribution, not ground truth.</b> Grad-CAM shows which "
               "patches drove the score. Presenting it as a forged-region mask is a "
               "claim the method does not support."),
    ("bullet", "<b>False positives on real photos.</b> Heavily edited, upscaled, "
               "denoised, or low-light photographs share surface statistics with "
               "generated images. Published detectors run 6-11 percent false positives; "
               "assume you are in that range until your benchmark says otherwise."),
    ("bullet", "<b>Faces only, for the deepfake path.</b> The face expert and the Haar "
               "localizer both assume a visible, roughly frontal face. Body swaps, "
               "object insertion, and background manipulation fall through to the "
               "splice expert alone."),
    ("bullet", "<b>Not evidence.</b> Output is a signal for triage, not proof of "
               "manipulation. Say so in the UI, not only in the README."),

    ("break", None),

    ("h1", "14. Execution checklist"),
    ("body", "In order. Each line is independently verifiable, so you always know "
             "whether the previous step actually worked."),

    ("table", ([
        ["#", "Task", "Done when"],
        ["1", "<font face='Courier'>pip install torch transformers opencv-python</font>",
         "<font face='Courier'>torch.backends.mps.is_available()</font> prints True"],
        ["2", "Create <font face='Courier'>src/detectors/generated.py</font>",
         "<font face='Courier'>python -m src.detectors.generated</font> prints three "
         "probabilities summing to 1"],
        ["3", "Create <font face='Courier'>src/detectors/faces.py</font>",
         "<font face='Courier'>python -m src.detectors.faces</font> prints a face count "
         "with no assertion error"],
        ["4", "Create <font face='Courier'>src/explain.py</font>",
         "Grad-CAM returns a heatmap matching the image size; verify the encoder layer "
         "path with <font face='Courier'>print(model)</font> first"],
        ["5", "Create <font face='Courier'>src/fusion.py</font>",
         "<font face='Courier'>python -m src.fusion</font> prints "
         "<font face='Courier'>ok fusion arithmetic</font>"],
        ["6", "Add <font face='Courier'>tests/test_fusion.py</font>",
         "<font face='Courier'>pytest tests/ -m 'not slow'</font> is green"],
        ["7", "Rewire <font face='Courier'>app/app.py</font>",
         "Gradio shows four scores, and the overlay caption states which kind of "
         "localization is being displayed"],
        ["8", "Build the <font face='Courier'>benchmark/</font> folder",
         "At least 20 images per category, including re-encoded real photos"],
        ["9", "Run <font face='Courier'>scripts/calibrate.py</font>",
         "You have chosen <font face='Courier'>fake_threshold</font> and "
         "<font face='Courier'>splice_weight</font> from measured numbers"],
        ["10", "Update README with results and limitations",
         "Per-category accuracy table, and every limitation from section 13 stated"],
        ["11", "<i>Optional:</i> add provenance checks",
         "A C2PA-signed test image is identified without any model running"],
        ["12", "<i>Optional:</i> Phase 2 retrain on Colab",
         "Cross-dataset accuracy reported on a fully held-out dataset"],
    ], [1.0, 6.5, 9.0])),

    ("h1", "15. What to claim when you present this"),
    ("body", "Accurate framing, which is also the more impressive framing:"),
    ("body", "<i>\"A forensic image analysis system combining compression-artifact "
             "analysis with learned generation detection. The ELA branch is a "
             "ResNet50 dual-head model I trained on CASIA v2, which classifies splicing "
             "and regresses the tampered region. It is fused by noisy-OR with a "
             "pretrained SigLIP2 classifier covering face-swap and fully-generated "
             "imagery, since compression forensics is provably blind to synthesis with "
             "no splice boundary. Verdict routing selects a localization strategy per "
             "forgery type: a regressed box for splices, face regions for swaps, and "
             "Grad-CAM attribution for generated images. Thresholds were calibrated "
             "against a benchmark including re-encoded real photographs, which is the "
             "dominant false-positive mode.\"</i>"),
    ("body", "That sentence contains the diagnosis, the architecture, the reason for "
             "the architecture, and an acknowledged failure mode. It is worth "
             "considerably more than a claimed accuracy number."),
]


# ── build ─────────────────────────────────────────────────────────────────────

def _decorate(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#999999"))
    canvas.drawString(2 * cm, 1.15 * cm, "Deepfake Localizer - Upgrade Plan")
    canvas.drawRightString(A4[0] - 2 * cm, 1.15 * cm, f"page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#dddddd"))
    canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
    canvas.restoreState()


def build(path=OUT):
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=2 * cm,
        title="Deepfake Localizer Upgrade Plan",
        author="upgrade plan",
    )
    story = []
    for kind, text in PLAN:
        if kind == "break":
            story.append(PageBreak())
        elif kind == "space":
            story.append(Spacer(1, text))
        elif kind == "table":
            rows, widths = text
            story.append(_cells(rows, widths))
            story.append(Spacer(1, 8))
        elif kind == "bullet":
            story.append(Paragraph(text, S["bullet"], bulletText="•"))
        elif kind == "code":
            # Keep a code block with the line above it where it fits on one page.
            story.append(Preformatted(text, S["code"]))
        elif kind in ("h1", "h2"):
            # Never strand a heading at the bottom of a page.
            story.append(KeepTogether([Paragraph(text, S[kind]), Spacer(1, 1)]))
        else:
            story.append(Paragraph(text, S[kind]))

    doc.build(story, onFirstPage=_decorate, onLaterPages=_decorate)
    return path


if __name__ == "__main__":
    written = build()
    size_kb = written.stat().st_size / 1024
    assert written.exists() and size_kb > 20, f"suspiciously small PDF: {size_kb:.0f} KB"
    print(f"ok  wrote {written}  ({size_kb:.0f} KB)")
