# Inference

## Single image (CLI)

```bash
python -m src.inference --image path/to/photo.jpg
```

Save the visual outputs:

```bash
python -m src.inference \
  --image path/to/photo.jpg \
  --save-overlay out_overlay.png \
  --save-ela out_ela.png
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--image` | *(required)* | Path to the input image |
| `--checkpoint` | `checkpoints/best_localizer.keras` | Path to a trained model |
| `--threshold` | `0.5` | Classification threshold for the tampered verdict |
| `--save-overlay` | none | Write the image with the predicted box drawn on it |
| `--save-ela` | none | Write the raw ELA map used as model input |

## Python API

```python
from src.inference import TamperingDetector

detector = TamperingDetector(checkpoint="checkpoints/best_localizer.keras")
result = detector.predict("path/to/photo.jpg")

print(result["label"])       # "tampered" or "authentic"
print(result["confidence"])  # float in [0, 1]
print(result["bbox"])        # [xmin, ymin, xmax, ymax], normalized 0-1
```

`TamperingDetector.predict_with_overlay()` additionally returns the input
image with the predicted box rendered on it, useful for building your own
UI on top of the model.

## Batch inference

There is no dedicated batch CLI yet. For scoring many images, iterate over a
single `TamperingDetector` instance rather than reconstructing it per image
— model loading is the expensive part:

```python
from pathlib import Path
from src.inference import TamperingDetector

detector = TamperingDetector()
results = {p: detector.predict(p) for p in Path("images/").glob("*.jpg")}
```

A first-class `--batch` / directory-input CLI mode is a reasonable follow-up
enhancement; see [docs/limitations.md](docs/limitations.md).

## Gradio interface

```bash
python app/app.py
```

Open <http://localhost:7860>. Upload an image to get a verdict, confidence
score, and box overlay in the browser.

The app degrades gracefully: if no checkpoint is found at
`checkpoints/best_localizer.keras`, it falls back to an **ELA-preview mode**
that shows the ELA map without a trained prediction, rather than crashing —
so a fresh clone of the repository is still usable before training.

## Expected outputs

Every prediction path (CLI, Python API, Gradio) returns the same shape:

| Field | Type | Description |
|---|---|---|
| `label` | `str` | `"authentic"` or `"tampered"` |
| `confidence` | `float` | Sigmoid output of the classification head, in [0, 1] |
| `bbox` | `list[float]` | `[xmin, ymin, xmax, ymax]`, normalized to the image size |

Boxes are only meaningful when `label == "tampered"`; a box is still
produced for authentic images (since the model always predicts one) but
should be ignored — see the box-loss masking discussion in
[docs/architecture.md](docs/architecture.md).

## Docker

```bash
docker build -t deepfake-localizer .
docker run -p 7860:7860 deepfake-localizer
```

To bake a trained model into the image, make sure
`checkpoints/best_localizer.keras` exists before `docker build` — the
Dockerfile copies the `checkpoints/` directory in as-is.
