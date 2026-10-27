# AI Deepfake & Tampering Localizer

**Is this image edited — and where?** A dual-head ResNet50 that answers both
at once: a real/tampered verdict with a confidence score, and a bounding box
over the region that was spliced in. Localization is driven by **Error Level
Analysis (ELA)** — instead of hoping a CNN notices a forgery from raw pixels,
the image is re-compressed and differenced first, turning an invisible
compression-history mismatch into a visible bright patch the model can
actually learn from.

<!-- Add a demo GIF here after deploying: ![demo](docs/images/demo.gif) -->

[![CI](https://github.com/Simp0099/Ai-Deepfake-Tampering-Localizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Simp0099/Ai-Deepfake-Tampering-Localizer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

## Contents

- [Features](#features)
- [Why ELA works](#why-ela-works)
- [Architecture](#architecture)
- [Results](#results)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Agent modules and wiring](#agent-modules-and-wiring)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Dual-head model** — a single ResNet50 trunk drives both a real/tampered
  classifier and a tampered-region bounding-box regressor.
- **ELA preprocessing** — re-compression differencing surfaces splice
  boundaries before the network ever sees them, so a modest model can find
  forgeries that are invisible in raw RGB.
- **Masked box loss** — authentic images contribute zero localization loss,
  so the box head is never trained to shrink toward the origin.
- **Config-driven** — every hyperparameter (image size, ELA quality, learning
  rates, loss weights, split fractions) lives in `configs/default.yaml`, not
  in code.
- **CPU-friendly CI** — the test suite builds the model with `weights=None`,
  so linting and unit tests run on a free GitHub Actions runner with no GPU
  and no ImageNet download.
- **Gradio demo + Docker** — a one-command local demo, and a container image
  that runs in ELA-preview mode even without a trained checkpoint.

## Why ELA works

JPEG compression is lossy but roughly *repeatable*: when a JPEG is saved,
every 8×8 block of pixels snaps onto a coarse grid of allowed values. Save
the same image again at the same quality and most pixels barely move —
they're already sitting where the encoder would put them.

Paste in a region from a different photo and it arrives with a *different*
compression history. Re-saving the whole image moves that region noticeably
more than its surroundings. ELA measures exactly that:

```text
1. Re-save the image as JPEG at a known quality (90)
2. Subtract the re-saved version from the original
3. Amplify the difference
```

Untouched areas go near-black; regions with a foreign compression history
light up — a localization signal available before any learning happens.

> **Caveat:** ELA is a heuristic, not proof. Uniform re-compression of the
> whole forgery partially erases the discrepancy, and strong texture can
> produce bright ELA regardless of tampering. It narrows the search; the
> model still has to judge.

## Architecture

```text
                    INPUT IMAGE (any size)
                            │
              ┌─────────────┴──────────────┐
              │                            │
         resize 224²                  ELA at full res,
         (RGB pixels)                 THEN resize 224²
              │                            │
              └─────────────┬──────────────┘
                            │
                  stack → 224 × 224 × 6
                            │
              ┌─────────────┴──────────────┐
       channels 0:3                  channels 3:6
              │                            │
   ┌──────────▼──────────┐      ┌──────────▼──────────┐
   │  ResNet50 backbone  │      │   ELA branch         │
   │  ImageNet weights   │      │   3× Conv2D + pool   │
   │  frozen → unfreeze  │      │   (small on purpose)  │
   │  last 30 layers     │      │                       │
   └──────────┬──────────┘      └──────────┬──────────┘
              │ GlobalAvgPool              │ GlobalAvgPool
              │ (2048)                     │ (64)
              └─────────────┬──────────────┘
                            │
                     Concatenate (2112)
                            │
                    Dense(256) + Dropout(0.3)
                            │
              ┌─────────────┴──────────────┐
              │                            │
      ┌───────▼────────┐          ┌────────▼─────────┐
      │  Dense(1)       │          │  Dense(5)        │
      │  sigmoid        │          │  sigmoid/softplus│
      │  classification │          │  box regression  │
      └───────┬────────┘          └────────┬─────────┘
              │                            │
        real / tampered           [xmin, ymin, xmax, ymax,
        + confidence                has_box], normalized 0-1

  Loss = BinaryCrossentropy(label) + λ · masked_MSE(bbox)
                                        └── zeroed for authentic images
```

Full design rationale, including the box parameterization and why geometric
augmentation is deliberately omitted, is in **[docs/architecture.md](docs/architecture.md)**.

## Results

| Metric | Head | Synthetic (n=24) | CASIA v2 |
|---|---|---|---|
| Accuracy | classification | 1.00 | 0.95 |
| Macro F1 | classification | 1.00 | — |
| ROC AUC | classification | — | 0.99 |
| Mean IoU | localization | 0.38 | — |
| Median IoU | localization | 0.43 | — |
| % boxes with IoU > 0.5 | localization | 33% | — |

The synthetic column is a pipeline smoke test, not a performance claim — see
**[docs/results.md](docs/results.md)** for the full breakdown, evaluation
methodology, and how to reproduce these numbers.

## Installation

```bash
git clone https://github.com/Simp0099/Ai-Deepfake-Tampering-Localizer.git
cd Ai-Deepfake-Tampering-Localizer

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Full requirements, GPU notes, and troubleshooting: **[INSTALL.md](INSTALL.md)**.

## Quick start

```bash
# 1. Verify the install with synthetic data (no download, ~2 min)
python -c "from src.data.dataset import synthetic_records; synthetic_records('data/synthetic', n=200)"
python -m src.train --data-root data/synthetic --epochs-frozen 2 --epochs-finetune 0
python -m src.evaluate

# 2. Real training on CASIA v2 (see DATASET.md for the download)
python scripts/download_data.py --dest data
python -m src.train --config configs/default.yaml
python -m src.evaluate

# 3. Single-image inference
python -m src.inference --image path/to/photo.jpg --save-overlay out.png

# 4. Demo app → http://localhost:7860
python3 app/app.py
```

### Docker

```bash
docker build -t deepfake-localizer .
docker run -p 7860:7860 deepfake-localizer
```

## Training

Two-phase transfer learning: heads trained with the ResNet50 backbone frozen,
then the last 30 backbone layers unfrozen for fine-tuning at a lower learning
rate. Full walkthrough, tuning knobs, and expected training behavior:
**[TRAINING.md](TRAINING.md)**.

## Inference

CLI, Python API, and Gradio interface for scoring single images. Details:
**[INFERENCE.md](INFERENCE.md)**.

## Evaluation

Accuracy, F1, ROC AUC, and IoU-based localization metrics, plus how to run
cross-dataset evaluation. Details: **[EVALUATION.md](EVALUATION.md)**.

## Forensic agent

The Gradio app routes image investigations through `agent.orchestrator`.
Deterministic code always runs image inspection, the localizer, and prediction
validation. The configured policy may then run ELA generation, metadata
inspection, or evidence comparison based on the score, validation result, and
investigation goal. Calls are bounded by configured timeouts and a per-image
tool budget; unsupported planner proposals are rejected and recorded.

An optional Anthropic planner is used only when both `ANTHROPIC_API_KEY` and
`ANTHROPIC_MODEL` are set. It can request tools from the approved conditional
registry, but it cannot run commands, choose arbitrary paths, or set the
verdict. Without those environment values, the deterministic policy still
produces an investigation record and verdict. Verdicts come from explicit
rules and use the categories **Flagged for review**, **Flagged, region
unavailable**, **Not flagged by this detector**, and **Inconclusive**.

The UI labels the model output `tampered score (uncalibrated)`. It displays
only checks recorded as executed, and shows ELA or a localization overlay only
when the corresponding evidence was produced. Filenames, goal text, and
metadata are treated as untrusted content when rendered. The app supports
multiple image uploads and ZIP batch validation; PDF reports are generated
from structured investigation records. Email delivery is not implemented.

This pipeline does not prove whether an image was manipulated, identify its
source, or provide a calibrated probability. The optional planner does not
change model weights or preprocessing, and no improvement over the baseline
has been established on a real labeled benchmark. Email is not implemented.
See [docs/wiring.md](docs/wiring.md) for module details and
[docs/evaluation.md](docs/evaluation.md) for the pre-registered evaluation
criterion.

## Synthetic data disclaimer

The synthetic-data evaluation in `scripts/evaluate_agent.py` /
`results/eval_synthetic.json` is a **plumbing check, not evidence about real
forgeries**. Both "baseline" and "policy" arms call the same `decide()`
function on hand-built records; no model inference runs. See
`docs/evaluation.md` and `FINAL_REPORT.md`.

## Repository layout

```text
├── src/
│   ├── data/
│   │   ├── ela.py            # Error Level Analysis
│   │   └── dataset.py        # CASIA loader, mask→bbox, tf.data pipeline
│   ├── models/
│   │   └── localizer.py      # dual-head model + masked box loss
│   ├── train.py               # two-phase training
│   ├── evaluate.py            # accuracy / F1 / IoU / plots
│   ├── inference.py           # single-image prediction
│   └── visualize.py           # box drawing
├── app/app.py                 # Gradio demo
├── configs/default.yaml       # every hyperparameter, no magic numbers in code
├── scripts/download_data.py   # CASIA v2 fetch (Kaggle)
├── tests/                     # unit tests for ELA, model I/O, and the pipeline
├── docs/                      # architecture, limitations, results, examples
├── .github/workflows/ci.yml   # ruff + pytest on every push
└── Dockerfile
```

Everything configurable — paths, image size, ELA quality, batch size,
learning rates, epochs, split fractions, and the box-loss weight — lives in
`configs/default.yaml`, overridable by CLI flags.

## Limitations

The model localizes with axis-aligned boxes rather than segmentation masks,
handles a single tampered region per image, and detects compression-boundary
splicing rather than fully generative (GAN/diffusion) forgeries. Full,
unvarnished discussion: **[docs/limitations.md](docs/limitations.md)**.

## Future work

- Segmentation head for pixel-level localization instead of bounding boxes
- Multi-region detection (anchors + NMS)
- Box-aware geometric augmentation
- Cross-dataset evaluation (FaceForensics++, CelebDF, DFDC, GenImage,
  Synthbuster, UniversalFakeDetect)
- Additional forensic branches (noise residual, FFT, SRM filters) and
  modern backbones (EfficientNetV2, ConvNeXt, ViT)

## Citation

If this project is useful in your work, please cite it as:

```bibtex
@software{deepfake_tampering_localizer,
  title  = {AI Deepfake \& Tampering Localizer},
  url    = {https://github.com/Simp0099/Ai-Deepfake-Tampering-Localizer},
  year   = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

Datasets carry their own terms and are not redistributed here; see
[DATASET.md](DATASET.md).

## Acknowledgements

- [CASIA v2](DATASET.md#casia-v2) image tampering dataset
- ResNet50 ImageNet weights via `tf.keras.applications`
- Built with TensorFlow/Keras and Gradio
