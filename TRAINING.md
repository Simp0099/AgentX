# Training

This document covers training in detail: configuration, the two-phase
schedule, checkpointing, and what a healthy run looks like. For a condensed
end-to-end command list, see the [Quick start](README.md#quick-start) in the
README.

> Always run modules with `python -m src.train`, not `python src/train.py`.
> The `-m` form puts the repository root on `sys.path`; the direct form
> fails with `ModuleNotFoundError: No module named 'src'`.

## 1. Confirm the environment is healthy

```bash
python -m src.models.localizer
```

This asserts that a training-mode forward pass and an inference-mode forward
pass agree on box predictions for the same input (`train/infer drift ≈ 0`).
A large drift indicates the model would train normally and then produce
incorrect boxes at inference time.

Then the full suite:

```bash
pytest tests/ -q
ruff check src/ app/ tests/ scripts/
```

## 2. Dry run on synthetic data

Proves the pipeline end-to-end before committing to a multi-gigabyte
download:

```bash
python -c "from src.data.dataset import synthetic_records; synthetic_records('data/synthetic', n=200)"

python -m src.train --data-root data/synthetic --epochs-frozen 8 --epochs-finetune 0 --batch-size 16

python -m src.evaluate
```

Accuracy hitting 1.0 on synthetic data is expected and not informative —
these splices are trivially separable. **Mean IoU** is the number that
signals whether the box head is actually learning; anything meaningfully
above 0 confirms the plumbing works. Clean up before real training:

```bash
rm -rf data/synthetic checkpoints reports
```

## 3. Get CASIA v2

See [DATASET.md](DATASET.md) for the full download and preparation guide.

```bash
python scripts/download_data.py --dest data
python -c "from src.data.dataset import build_index; build_index('data/CASIA2')"
```

## 4. Train

A short rehearsal surfaces path and memory issues in minutes rather than
hours:

```bash
python -m src.train --config configs/default.yaml --epochs-frozen 1 --epochs-finetune 0
```

If that completes cleanly, start the full run:

```bash
rm -rf checkpoints

nohup python -m src.train --config configs/default.yaml > train.log 2>&1 &
tail -f train.log
```

`nohup ... &` detaches the process so closing the terminal will not kill it.
On CPU, training 10 frozen + 10 fine-tune epochs on the full CASIA v2 set
(~12k images) takes on the order of hours, since ELA re-encodes every image
in Python and is the dominant cost. A CUDA GPU or a hosted GPU runtime
reduces this substantially.

If training is too slow, reduce epochs rather than batch size:

```bash
python -m src.train --config configs/default.yaml --epochs-frozen 5 --epochs-finetune 3
```

Under memory pressure:

```bash
python -m src.train --config configs/default.yaml --batch-size 16
```

### What a healthy run looks like

Watch `val_bbox_loss`, not accuracy:

- `val_label_accuracy` reaches close to 1.0 within a few epochs and stays
  there. This is expected — classification is the easier half of the task
  and is not informative on its own.
- `val_bbox_loss` should fall steadily across training. This is the signal
  that localization is actually improving.
- `val_loss` (the combined objective) is what `EarlyStopping` and
  `ModelCheckpoint` monitor, specifically so that checkpointing keeps
  tracking improvement in the box head after classification has plateaued.

### Tuning knobs (`configs/default.yaml`)

| Symptom | Change |
|---|---|
| Boxes are vague, classification is fine | raise `bbox_loss_weight` to 2.0–5.0 |
| Box loss dominates, accuracy stalls | lower `bbox_loss_weight` to 0.5 |
| Val accuracy collapses in phase 2 | lower `lr_finetune` to 1e-6 |
| Overfitting (train ≫ val) | lower `epochs_finetune`, or set `fine_tune_from: -10` |

### Outputs

- `checkpoints/best_localizer.keras` — best model by `val_loss`
- `checkpoints/test_split.json` — held-out records, fixed at training time so
  evaluation always scores the same test set
- `checkpoints/history.json` — per-epoch metrics for both phases

## 5. Evaluate

```bash
python -m src.evaluate
```

See [EVALUATION.md](EVALUATION.md) for metric definitions and how to read
the generated report.

## Checkpoints and resuming

Training does not currently support resuming from a partial run — each
invocation of `python -m src.train` starts from ImageNet weights (or
`weights=None` with `--no-pretrained`) and trains both phases fresh. To
continue experimenting from a saved checkpoint, load
`checkpoints/best_localizer.keras` directly with
`tf.keras.models.load_model()` and call `model.fit()` in a custom script.
Native resume support is tracked as a future enhancement.

## Mixed precision

Not currently enabled. `configs/default.yaml` has no mixed-precision option;
enabling `tf.keras.mixed_precision` would primarily benefit GPU training and
is a reasonable addition for anyone training at scale (see
[docs/limitations.md](docs/limitations.md)).

## TensorBoard

Not currently wired up. `output.log_dir` in `configs/default.yaml` is
reserved for this; adding a `tf.keras.callbacks.TensorBoard` callback in
`src/train.py` is a small, low-risk enhancement.

## Hardware recommendations

| Setup | Expectation |
|---|---|
| CPU only (8 GB RAM) | Works; training the full CASIA v2 set takes several hours. Reduce `batch_size` if memory-constrained. |
| CUDA GPU | Training time drops substantially; no code changes needed. |
| Hosted GPU (e.g. Colab T4) | Recommended for full training runs without local hardware. Clone the repo, install requirements, and run the same commands as above. |

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Use `python -m src.train`, not `python src/train.py` |
| `Could not find Au/ and Tp/ under ...` | Check `ls data/CASIA2/`; rename folders or extend the candidate lists in `src/data/dataset.py::build_index` |
| `No checkpoint at checkpoints/...` | Train before running inference or evaluation |
| `No test split at checkpoints/test_split.json` | Run training before evaluation |
| Process killed / out of memory | Lower `--batch-size` |
| Training is very slow | Expected on CPU; reduce epochs or use a GPU runtime |
| Accuracy near 1.0 but mean IoU near 0 | See [docs/architecture.md](docs/architecture.md#design-notes) — this indicates the box head has stopped learning, addressed in this codebase via `LayerNormalization` and monitoring `val_loss` instead of `val_label_accuracy` |
