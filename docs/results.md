# Results

Numbers here are produced by `python -m src.evaluate` and written to
`reports/`. See [EVALUATION.md](../EVALUATION.md) for metric definitions and
the exact reproduction procedure.

## Headline metrics (CASIA v2 test split)

| Metric | Value |
|---|---|
| Label accuracy | ≈ 95% |
| Validation accuracy | ≈ 88% |
| ROC AUC | ≈ 0.99 |

These reflect the classification head on the held-out CASIA v2 test split
described in [DATASET.md](../DATASET.md). Localization metrics (mean IoU,
median IoU, % IoU > 0.5) for the CASIA v2 test split are pending a full
report run — see the `_TBD_` markers in the results table until an official
run is recorded and this file is updated.

## Pipeline smoke test (synthetic data)

| Metric | Head | Value |
|---|---|---|
| Accuracy | classification | 1.00 |
| Macro F1 | classification | 1.00 |
| Mean IoU | localization | 0.38 |
| Median IoU | localization | 0.43 |
| % boxes with IoU > 0.5 | localization | 33% |

This run uses 24 synthetically generated splices with exact ground-truth
boxes, trained for 10 frozen epochs with no fine-tuning. It is a **plumbing
check, not a performance claim** — perfect classification accuracy is
expected on data this separable and says nothing about performance on real
forgeries. The mean IoU of 0.38 is the more informative number: even with
boxes that are trivially easy to classify, regressing tight box coordinates
back out is a harder task, and this figure is a useful lower bound for
sanity-checking the localization head before spending compute on a full
CASIA v2 run.

## Reading the qualitative report

`reports/qualitative.png` shows predicted (red) vs. ground-truth (green)
boxes on a sample of real images. Always check this before trusting the
aggregate IoU numbers — a model that always predicts one centered,
half-image box can post a misleadingly reasonable mean IoU on an imbalanced
test set, and the qualitative grid is what makes that visible.

## Cross-dataset results

Not yet available. See [docs/limitations.md](limitations.md#single-dataset-evaluation).

## Updating this file

After running `python -m src.evaluate` against a new checkpoint, copy every
field from `reports/eval_results.json` into the tables above — report
results as-is, including a weak or unexpected metric. A stated limitation is
more credible than a suspiciously perfect number.
