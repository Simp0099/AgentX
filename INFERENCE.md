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

Open <http://localhost:7860>. The **Analysis** tab runs the fixed forensic
pipeline and displays the verdict, uncalibrated score, checks, and any box
overlay. The **Chat** tab lets the configured model choose among the existing
forensic tools; its expandable panel lists each tool's arguments, status, and
result preview, with an ELA image beside the answer. Conversation history and
the active image are kept in Gradio state for follow-up questions.

The app degrades gracefully: if no checkpoint is found at
`checkpoints/best_localizer.keras`, it falls back to an **ELA-preview mode**
that shows the ELA map without a trained prediction, rather than crashing —
so a fresh clone of the repository is still usable before training.

## Agent Chat

Chat calls `agent.loop.run_agent_full` with a bounded tool loop. Its approved
tools are `inspect_image`, `run_localizer`, `validate_prediction`,
`generate_ela`, `inspect_metadata`, and `compare_evidence`. The language model
chooses the tools and reads their results; the verdict remains grounded in the
existing localizer and verdict rules.

The configured primary and fallback model variables are `OPENROUTER_API_KEY`
and `GEMINI_API_KEY`. Provide them through the shell or a secret manager. If
all configured providers are missing or unavailable, the handler calls the
deterministic `run_investigation` path and labels its answer **fixed pipeline
(no LLM)**. Missing checkpoints, corrupt uploads, provider errors, rate limits,
and malformed tool calls can limit the evidence or stop the agent. A failed
localizer does not produce an invented score; ELA remains available when the
image can be read.

Long-term results are stored in SQLite at
`artifacts/memory/investigations.db`, keyed by image hash and configuration.
Chat uploads live under `artifacts/uploads`; a per-turn temporary copy is
removed after processing. Tools and the local HTTP API reject paths outside
the uploads root. The API defaults to loopback and should not be exposed on a
public interface without `AGENT_API_TOKEN` protection.

See the [model comparison table in the README](README.md#agent-model-comparison)
and the detailed [candidate measurements](docs/model_selection.md). The
one-shot tool-call probe measures API compatibility, not forensic accuracy.

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
docker run --rm -p 7860:7860 -e OPENROUTER_API_KEY -e GEMINI_API_KEY deepfake-localizer
```

The Dockerfile has no API-key build arguments. Provider keys are passed only at
container run time. `.dockerignore` excludes `.env`, artifacts, logs, and the
Git directory from the build context. The Dockerfile copies the checkpoint
directory if a checkpoint is present in the build context.
