# Examples

## Score a single image from the command line

```bash
python -m src.inference --image path/to/photo.jpg --save-overlay out.png --save-ela out_ela.png
```

`out.png` shows the predicted box drawn on the original image; `out_ela.png`
shows the raw Error Level Analysis map the model saw as input, which is
often the fastest way to sanity-check a prediction visually.

<!-- Add before/after example images to docs/images/ once available, e.g.:
![authentic example](images/example_authentic.png)
![tampered example](images/example_tampered.png) -->

## Score an image programmatically

```python
from src.inference import TamperingDetector

detector = TamperingDetector(checkpoint="checkpoints/best_localizer.keras")
result = detector.predict("path/to/photo.jpg")

if result["label"] == "tampered":
    print(f"Tampered, confidence={result['confidence']:.2f}, box={result['bbox']}")
else:
    print(f"Authentic, confidence={result['confidence']:.2f}")
```

## Score a folder of images

```python
from pathlib import Path
from src.inference import TamperingDetector

detector = TamperingDetector()

for path in Path("images/").glob("*.jpg"):
    result = detector.predict(str(path))
    print(f"{path.name}: {result['label']} ({result['confidence']:.2f})")
```

## Draw a custom overlay

```python
from PIL import Image
from src.inference import TamperingDetector
from src.visualize import draw_bounding_box_on_image

detector = TamperingDetector()
result = detector.predict("path/to/photo.jpg")

image = Image.open("path/to/photo.jpg").convert("RGB")
if result["label"] == "tampered":
    xmin, ymin, xmax, ymax = result["bbox"]
    draw_bounding_box_on_image(image, xmin, ymin, xmax, ymax, color="red")
image.save("custom_overlay.png")
```

## Run the interactive demo

```bash
python app/app.py
```

Open <http://localhost:7860>, upload an image, and adjust the classification
threshold slider to see how the verdict changes. Works even without a
trained checkpoint (ELA-preview mode).

## Generate synthetic training data

Useful for testing changes to the pipeline without downloading CASIA v2:

```python
from src.data.dataset import synthetic_records

synthetic_records("data/synthetic", n=50)
```

Produces JPEG-settled background images with a random patch pasted in at a
known location, so the ground-truth box is exact by construction — ideal
for a quick pipeline correctness check.
