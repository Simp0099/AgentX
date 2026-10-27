# Evaluation

```bash
python -m src.evaluate
```

Evaluates the trained checkpoint against the held-out test split saved
during training (`checkpoints/test_split.json`), so metrics are always
computed on images the model never trained or validated on.

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/best_localizer.keras` | Model to evaluate |
| `--split` | `checkpoints/test_split.json` | Held-out records to evaluate against |
| `--config` | `configs/default.yaml` | Config for image size / ELA quality |
| `--out-dir` | `reports` | Where to write plots and metrics |

## Metrics

**Classification head**

- **Accuracy** — fraction of images correctly labeled real/tampered
- **Macro F1** — precision/recall balance across both classes, unaffected by
  class imbalance
- **ROC AUC** — ranking quality of the confidence score independent of
  threshold
- **Confusion matrix** — written to `reports/confusion_matrix.png`

**Localization head**

- **IoU (Intersection over Union)** — overlap between predicted and
  ground-truth box, computed only on tampered images (authentic images have
  no ground-truth box)
- **Mean / median IoU** — aggregate localization quality
- **% boxes with IoU > 0.5** — a common threshold for "roughly located the
  right region"
- **IoU distribution** — written to `reports/iou_distribution.png`, with
  mean and the 0.5 threshold marked

## Output artifacts

| File | Contents |
|---|---|
| `reports/eval_results.json` | All metrics above, machine-readable |
| `reports/confusion_matrix.png` | Classification confusion matrix |
| `reports/iou_distribution.png` | Histogram of per-image IoU |
| `reports/qualitative.png` | Grid of predicted (red) vs. ground-truth (green) boxes on real images |

**Always inspect `reports/qualitative.png` before trusting the aggregate
numbers.** A model that always predicts one centered, half-image-sized box
can post a deceptively reasonable mean IoU on an imbalanced test set; the
qualitative grid catches that pattern immediately where the summary
statistics do not.

## Cross-dataset evaluation

Not currently implemented. `src/evaluate.py` scores against the CASIA v2
test split only. Evaluating cross-dataset generalization (e.g. training on
CASIA v2, testing on Columbia, CoMoFoD, or IMD2020) would require adding a
loader for each target dataset and pointing `--split` at an externally
built record list. This is flagged as a priority next step in
[docs/limitations.md](docs/limitations.md) — forensics models are known to
generalize poorly across datasets, and CASIA-only metrics should be read
with that caveat.

## Benchmark procedure

To reproduce the numbers in [docs/results.md](docs/results.md):

1. Train with `configs/default.yaml` unmodified (fixed seed, fixed split
   fractions).
2. Run `python -m src.evaluate` immediately after training completes, using
   the `test_split.json` written by that run.
3. Report all fields from `reports/eval_results.json` without cherry-picking
   a subset.
