# Pre-registered evaluation criterion

This criterion is registered before the evaluation run. It applies to a labeled
hard subset of **tampered images containing re-encoded and low-contrast
splices**. The subset definition and decision rule are fixed before collecting
or comparing evaluation outcomes.

## Metric

Measure the percentage of tampered images whose verdict is exactly
**“Not flagged by this detector”**. Lower is better.

## Comparison

Compare the full conditional investigation policy against a baseline run using
baseline checks only. Both arms must use the same images, labels, detector,
threshold, and verdict definition.

## Success threshold

The full policy meets the criterion only if its metric is at least **5
percentage points lower** than the baseline metric.

## Statistical method

Estimate the difference in metric between arms with a **95% bootstrap
confidence interval**, using fixed random seed **42**. The interval must
**exclude zero** in the direction favoring the full policy. Report the subset
size, both rates, their percentage-point difference, and the confidence
interval.

Synthetic or hand-authored records do not satisfy the real labeled hard-subset
requirement. If no real labeled hard-subset data is available, any synthetic
run is a plumbing check only and the criterion status must be **NOT EVALUATED**.
