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

The Gradio app has two paths. **Analysis** uses the deterministic
`agent.orchestrator.run_investigation` pipeline. **Chat** uses
`agent.loop.run_agent_full`, which lets a configured language model choose
among the approved tools below, reads each result, and then answers. The model
does not replace or change the trained localizer.

| Tool | Purpose |
|---|---|
| `inspect_image` | Check image readability, dimensions, format, and SHA-256. |
| `run_localizer` | Run the existing ELA-based classifier/localizer. |
| `validate_prediction` | Check the returned score and box. |
| `generate_ela` | Write an ELA map for visual review. |
| `inspect_metadata` | Read selected image metadata. |
| `compare_evidence` | Compare evidence gathered earlier in the run. |

### Run the agent

```bash
git clone https://github.com/Simp0099/Ai-Deepfake-Tampering-Localizer.git
cd Ai-Deepfake-Tampering-Localizer
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For model-driven Chat, copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`
and/or `GEMINI_API_KEY`. The app loads `.env` before reading provider settings;
it never displays or logs key values. `configs/agent.yaml` lists the free models
in routing order: OpenRouter Nemotron, then the verified Gemini free-tier
models. A quota or rate-limit response skips that model immediately and routes
to the next configured free model; transient failures retry twice first. A
model without a configured key is skipped. OpenCode Zen community models are
not routed because direct API probes were rejected upstream; their free tier is
limited to the OpenCode client. `GET /api/health` reports provider availability
without returning a key. With no usable free-model key, Chat explains the
missing-key reason and uses the deterministic fixed pipeline.

```bash
python app/app.py
```

Open <http://localhost:7860>, choose **Chat**, upload an image, enter a question,
and click **Ask**. The expandable evidence panel shows tool arguments, status,
and result previews; ELA appears beside the answer. Follow-ups reuse the
Gradio conversation state and saved image. Analysis and batch ZIP validation
remain available in the **Analysis** tab.

For Docker, export only the provider variables you intend to use, then pass
those variable names at runtime. Do not pass a shared `.env` file or add keys
to image build arguments:

```bash
docker build -t deepfake-localizer .
docker history --no-trunc deepfake-localizer
mkdir -p artifacts
docker run --rm -p 7860:7860 -v "$PWD/artifacts:/app/artifacts" \
  -e LLM_TIMEOUT_SECONDS -e OPENROUTER_API_KEY -e GEMINI_API_KEY deepfake-localizer
```

The agent keeps SQLite long-term investigation memory at
`artifacts/memory/investigations.db`, keyed by image hash and run configuration.
Conversation history and the active image path/hash are held in Gradio state.
Uploaded chat images are saved under `artifacts/uploads`; agent tools and the
local API reject paths outside that directory. The API should be bound to
loopback unless protected by its optional `AGENT_API_TOKEN`.

Failures remain visible in the evidence panel. If a provider is unavailable,
the UI uses the labeled fixed pipeline. Missing/corrupt images fail inspection;
a missing checkpoint makes localizer calls fail but does not invent a score.
Rate limits, malformed model tool calls, and exhausted model fallbacks can
prevent an agent answer. The localizer score is uncalibrated, and neither a
flagged nor unflagged result proves authenticity or tampering. ELA and the
localizer are weak on uniformly re-encoded images and do not detect fully
AI-generated images.

See [INFERENCE.md](INFERENCE.md#agent-chat), [docs/limitations.md](docs/limitations.md#agent-chat-and-deployment-failure-modes),
[docs/wiring.md](docs/wiring.md), and [docs/deploy_checklist.md](docs/deploy_checklist.md).
Evaluation details are in [docs/evaluation.md](docs/evaluation.md).
Email is not implemented; specifically, email delivery is not implemented.

### Agent model comparison

The table below records the Phase 1 one-shot text/tool-calling screen. It is a
model API compatibility check, not an image accuracy result. The separate
three-image tool-use smoke is listed after this table.

| Model | Provider | Tool-call probe | Median latency | Configured role |
|---|---|---:|---:|---|
| `nvidia/nemotron-3.5-lightning:free` | OpenRouter | 5/5 | 856 ms | Primary |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenRouter | 5/5 | 1,522 ms | Fallback |
| `gemini-3.6-flash` | Gemini | 5/5 | 2,026 ms | Fallback |
| `gemini-3.5-flash-lite` | Gemini | 5/5 | 6,792 ms | Fallback |

The underlying measurements, retired candidates, method, and caveats are in
[docs/model_selection.md](docs/model_selection.md). The three-image smoke was
not run in this environment because provider variables were unavailable; see
the results table below.

### Phase 9 live model smoke results

The real-image agent smoke could not be run in this checkout because none of
the configured provider credential variables was available. These entries are
explicitly **not run**, rather than inferred from the earlier one-shot tool
probe; run `python scripts/smoke_agent.py` when credentials are configured. The
runner checks authentic, tampered, and corrupt inputs, records tool arguments
and result sizes, and requires a written plan before its first tool result.

| Candidate model | Tool-calling result | Date |
|---|---|---|
| `gemini-2.5-flash` | Not run: `GEMINI_API_KEY` unavailable | 2026-10-27 |
| `gemini-3.6-flash` | Not run: `GEMINI_API_KEY` unavailable | 2026-10-27 |
| `gemini-3.5-flash-lite` | Not run: `GEMINI_API_KEY` unavailable | 2026-10-27 |
| `nvidia/nemotron-3.5-lightning:free` | Not run: `OPENROUTER_API_KEY` unavailable | 2026-10-27 |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | Not run: `OPENROUTER_API_KEY` unavailable | 2026-10-27 |
| `nvidia/nemotron-3-super-120b-a12b:free` | Not run: `OPENROUTER_API_KEY` unavailable | 2026-10-27 |
| `inclusionai/ling-3.0-flash-sante:free` | Not run: `OPENROUTER_API_KEY` unavailable | 2026-10-27 |
| `space-bunny-free` | Not run: `OPENCODE_ZEN_API_KEY` unavailable | 2026-10-27 |
| `big-pickle` | Not run: `OPENCODE_ZEN_API_KEY` unavailable | 2026-10-27 |

The live behavior checks (different tool sequences for different images and a
later call reacting to an earlier tool result) are also **not run**. The
20-image agent benchmark is invoked with `python scripts/benchmark_agent.py`;
the local CASIA2 copy has no ground-truth masks or declared test split, so the
runner uses generated images and labels the output `synthetic` in
`reports/agent_eval.json`.

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
