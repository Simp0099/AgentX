# Limitations

Honest, direct list of what this project does not (yet) do well.

## Agent chat and deployment failure modes

The Chat tab uses a language model to select among six existing tools:
`inspect_image`, `run_localizer`, `validate_prediction`, `generate_ela`,
`inspect_metadata`, and `compare_evidence`. The model does not provide a
separate detector and does not change the trained weights. Its tool choices
can be incomplete or malformed; provider timeouts, rate limits, unavailable
models, or a missing API credential can stop agent mode. In those cases the
handler may use `run_investigation`, and the answer is labeled **fixed pipeline
(no LLM)**. A missing checkpoint causes localizer errors; a corrupt or
unsupported upload can prevent all image analysis.

The localizer's score is an uncalibrated review signal, not a probability or
proof. A clean result does not establish authenticity, and the Chat model can
misread evidence or explain it incorrectly. Review the tool results and image
yourself. ELA and the localizer are limited to splice-style evidence and can
miss fully AI-generated images or uniformly re-encoded edits.

Chat conversation history is kept in Gradio state. Long-term memory stores
investigation records in `artifacts/memory/investigations.db`, keyed by image
hash and run configuration; it may return a prior matching result. Uploaded
chat images are stored under `artifacts/uploads`, while per-turn work copies
are removed. Tool and API paths are resolved and checked against that upload
root, including symlinks. The API defaults to loopback; exposing it publicly
without `AGENT_API_TOKEN` is unsafe.

The [README model comparison table](../README.md#agent-model-comparison) shows
tool-call compatibility measurements. Those one-shot probes do not measure
forensic accuracy or guarantee that a provider will be available at runtime.

## Boxes are a crude shape for splices

Real tampered regions are irregular. A rectangle around a curved cutout
necessarily includes untampered pixels, which caps achievable IoU regardless
of model quality. CASIA v2's ground truth is already pixel-level and is
discarded down to a box by `mask_to_bbox()` — a segmentation head predicting
the mask directly, rather than a box, would likely be the single largest
improvement available without changing the backbone.

## Single tampered region per image

The box head regresses one rectangle. An image with two separate splices
produces one box covering both regions, or a box that favors one over the
other. Multi-region output would require an anchor-based or query-based
detection head with non-maximum suppression, which is a materially larger
architecture change.

## No geometric data augmentation

Standard Keras augmentation layers (flip, rotate, crop) transform the image
but not an accompanying box label, so naively applying them would train the
model against incorrect box targets. Geometric augmentation also resamples
pixels, which smooths the exact compression artifacts ELA is built to
detect. The correct fix is a custom augmentation step that transforms image
and box together, applied before ELA is computed — not yet implemented.
Without augmentation, the model is more exposed to overfitting on a
training set of this size.

## ELA degrades under re-compression

Any pipeline that re-encodes the whole image uniformly — social media
uploads, screenshots, messaging apps — partially erases the
compression-history mismatch ELA relies on. The model is weakest on exactly
the kind of images most likely to be encountered outside a controlled
dataset.

## Splicing detection, not general deepfake detection

This model detects *composite* forgeries that have a compression boundary
between an original and a pasted region. A fully GAN- or diffusion-generated
image has no splice at all — it is uniformly synthetic, so ELA is close to
uniform across the whole frame and this approach has little to grip. Modern
generative-image detection typically requires generator-artifact or
frequency-spectrum methods instead. The project name reflects the familiar
umbrella term ("deepfake"); the current, honest scope is image splicing and
tampering detection specifically.

## Single-dataset evaluation

Reported metrics come from CASIA v2's own test split. Forensics models are
known to generalize poorly across datasets — an artifact learned from one
dataset's specific compression pipeline and camera sources may not transfer.
Cross-dataset evaluation (Columbia, CoMoFoD, IMD2020, or the datasets listed
in [DATASET.md](../DATASET.md#supported-datasets)) has not yet been run and
would be the meaningful test of generalization.

## Domain shift and dataset bias

CASIA v2 images share a specific era of cameras, compression settings, and
splicing styles. Performance on modern smartphone photos, different JPEG
quality regimes, or non-photographic content (screenshots, scanned
documents, AI-upscaled images) is untested.

## No batch or streaming inference tooling

Inference is single-image by default; batch scoring currently requires
writing a small wrapper script around `TamperingDetector` (see
[INFERENCE.md](../INFERENCE.md#batch-inference)) rather than a first-class
CLI flag.

## Confidence scores are a signal, not a verdict

A confidence score from a model trained on one splicing dataset is
appropriately used as a signal for a human reviewer to investigate further —
it is not a forensic conclusion and should not be treated as one in any
legal, journalistic, or moderation context.
