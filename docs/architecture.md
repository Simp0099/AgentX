# Architecture

## Overview

The model is a dual-head convolutional network built on a ResNet50 backbone.
A single forward pass produces both a real/tampered classification and a
bounding box localizing the tampered region, if any.

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
   │  ResNet50 backbone  │      │   ELA branch          │
   │  ImageNet weights   │      │   3× Conv2D + pool    │
   └──────────┬──────────┘      └──────────┬──────────┘
              │ GlobalAvgPool (2048)        │ GlobalAvgPool (64)
              └─────────────┬──────────────┘
                            │
                     Concatenate (2112)
                            │
                    Dense(256) + Dropout(0.3)
                            │
              ┌─────────────┴──────────────┐
      ┌───────▼────────┐          ┌────────▼─────────┐
      │  Dense(1)       │          │  Dense(5)         │
      │  sigmoid        │          │  see below        │
      │  classification │          │  box regression   │
      └────────────────┘          └───────────────────┘
```

## Error Level Analysis (ELA)

JPEG compression is lossy but approximately idempotent: re-saving an image
at the same quality moves most pixels only slightly, because they already
sit on the quantization grid the encoder produces. A region pasted in from
elsewhere carries a different compression history and is not on that grid,
so it shifts more under re-compression. ELA computes this directly:

1. Re-encode the image as JPEG at a fixed quality (default 90).
2. Take the per-pixel absolute difference against the original.
3. Rescale so the brightest difference maps to 255.

The result is near-black over untouched regions and bright over regions
with a mismatched compression history — a localization signal computed
before any learned features are involved. Implementation: `src/data/ela.py`.

ELA is computed **before** resizing to 224×224, not after. Resizing first
would resample away the fine compression artifacts ELA depends on.

## Backbone: ResNet50

`ResNet50` (`tf.keras.applications.ResNet50`, ImageNet weights) processes
the RGB channels. Training proceeds in two phases:

1. **Frozen phase** — the backbone is frozen; only the new heads train, at
   a higher learning rate (`lr_frozen`, default `1e-3`).
2. **Fine-tune phase** — the last `fine_tune_from` layers (default 30) are
   unfrozen and trained jointly with the heads at a lower learning rate
   (`lr_finetune`, default `1e-5`), to adapt high-level features to the
   splicing-detection task without destroying pretrained low-level filters.

## ELA branch

A small dedicated convolutional stack (3× `Conv2D` + pooling) processes the
ELA channels separately from the RGB backbone. It is deliberately shallow:
ELA is already an artifact map rather than natural image content, so it
does not benefit from a deep, ImageNet-scale feature extractor.

## Fusion and heads

Global-average-pooled features from both branches (2048 + 64 = 2112 dims)
are concatenated, passed through `Dense(256)` with dropout, and split into
two heads:

- **Classification head** — `Dense(1)`, sigmoid → tampered probability.
- **Box head** — `Dense(5)`, decoded as described below.

A shared trunk (rather than two independent models) is used because the
features that reveal *that* an image was edited substantially overlap with
the features that reveal *where* — the box head benefits from gradient
signal contributed by the classification objective as well as its own.

## Box parameterization

The box head predicts `(cx, cy, w, h, has_box)` — center coordinates and
size, plus a flag for whether a box applies at all — rather than corner
coordinates `(xmin, ymin, xmax, ymax)` directly.

Predicting corners directly through independent sigmoids allows the model
to output `xmin > xmax`, producing zero- or negative-area boxes with no
useful gradient once the sigmoids saturate. Centre/size parameterization
guarantees positive width and height by construction — `w` and `h` use
`softplus` (positive everywhere, non-saturating) rather than `sigmoid`
(bounded, but with vanishing gradient near its limits), so a coordinate
that drifts to an extreme value during training can still recover.

Corner clipping to `[0, 1]` is applied only at inference, not inside the
training graph, since `clip_by_value` has zero gradient outside its range
and would otherwise strand any prediction that drifted past the boundary.

Implementation: `src/models/localizer.py` (`ChannelSlice`, `CornerDecode`,
`center_size_to_corners`).

## Loss function

```text
total_loss = BinaryCrossentropy(label) + λ · masked_MSE(bbox)
```

`λ` is `bbox_loss_weight` in `configs/default.yaml` (default `1.0`).

The box loss is **masked** by the `has_box` flag: authentic images have no
tampered region, so a naive box loss would train the head to shrink every
prediction toward `[0, 0, 0, 0]`. Instead, authentic images contribute
exactly zero box loss, and only tampered images (where `has_box == 1`)
drive the localization objective.

## Normalization choice

Both the RGB and ELA feature paths use `LayerNormalization` rather than
`BatchNormalization` ahead of the fully connected layers. `BatchNormalization`
uses batch statistics during training but accumulated moving averages at
inference; with the default momentum, those averages need a large number of
steps to converge, and on short or small-batch training runs the two modes
can diverge substantially, silently corrupting inference-time predictions
even though training-time metrics look fine. `LayerNormalization` computes
per-sample statistics, so training and inference behavior match by
construction, with no warm-up period.

This is validated directly: `python -m src.models.localizer` runs one batch
through both training mode and inference mode and asserts the box outputs
match (`train/infer drift ≈ 0`). This check is a regression test for exactly
this class of bug and should stay green.

## Training monitoring

`EarlyStopping` and `ModelCheckpoint` both monitor `val_loss` (the combined
objective), not `val_label_accuracy`. Classification accuracy on this task
saturates within the first few epochs and stops moving well before
localization has converged; monitoring accuracy alone would cause both
callbacks to stop tracking real progress once the easier half of the task
plateaus, discarding gains from the harder localization objective in the
epochs that follow. `val_loss` (BCE + λ·masked box loss) continues to
reflect improvement as long as either head is still learning.

## Design decisions not taken

- **No geometric augmentation.** Flips and rotations transform the image
  but not the box label under standard Keras preprocessing layers, which
  would teach the model an incorrect, mirrored location. Augmentation also
  resamples pixels, smoothing the exact compression artifacts ELA depends
  on. A correct fix requires custom augmentation that transforms image and
  box together, applied before ELA is computed — see
  [limitations.md](limitations.md).
- **Bounding boxes, not segmentation masks.** CASIA v2 provides pixel-level
  masks, but the current pipeline reduces them to a single bounding box per
  image, discarding shape information. See [limitations.md](limitations.md).

## Inference pipeline

`src/inference.py`'s `TamperingDetector` wraps preprocessing (resize + ELA),
the forward pass, and box decoding into a single call, so the CLI, Gradio
app, and any external caller all read predictions through the same code
path and convention. See [INFERENCE.md](../INFERENCE.md).
