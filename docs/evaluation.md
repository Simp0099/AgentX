# Pre-registered Evaluation Criterion

## Criterion (Phase 10, registered before data collection)

**Synthetic data is a plumbing check, not evidence about real forgeries.**

## Audit criterion amendment (not pre-registered)

The requested criterion for any future evaluation is: on a labeled hard subset
of re-encoded and low-contrast splices, the full policy must reduce wrong
"Not flagged by this detector" outcomes by at least 5 percentage points versus
baseline checks alone, with a 95% bootstrap confidence interval excluding zero
and a fixed seed. This amendment is **not claimed as pre-registered**: this
workspace has no `.git`, and the sandbox denied `git init`, so commit order
cannot be established. `results/eval_synthetic.json` is plumbing-only and the
two arms call the same verdict function. Phase 10 remains NOT EVALUATED.

On a hard subset of synthetic records (re-encoded JPEG splices and low-contrast splices),
the policy must reduce wrong "Not flagged by this detector" outcomes by at least 5
percentage points versus the baseline checks alone.

### Baseline (checks alone)
- The baseline consists of: inspect_image → run_localizer → validate_prediction
- Without any policy/agent intervention, the "Not flagged by this detector" rate on
  hard subsets is expected to be high (approximately 55-65% depending on splice quality).
- This baseline is measured by running the deterministic pipeline on synthetic records
  with re-encoding and low-contrast manipulations.

### Target (with policy/agent)
- The policy/agent should correctly flag a higher proportion of tampered images
  that the baseline misses.
- Target: reduce the "Not flagged by this detector" rate by at least 5 percentage
  points on the hard subset.
- Example: if baseline produces 60% "Not flagged by this detector" on hard splices,
  the policy should bring this down to 55% or lower (i.e., correctly flag at least
  5% more tampered images).

### Evaluation Method

1. **Generate synthetic records** with the following manipulations:
   - Re-encoded JPEG (double JPEG compression)
   - Low-contrast images
   - Border-level splices
   - Heavy re-compression

2. **Run the agent** (deterministic pipeline) on all synthetic records and record:
   - Verdict for each record
   - Whether the verdict was "Not flagged by this detector" (wrong for tampered)
   - All observation data

3. **Run the policy** (if built) on the same synthetic records and record:
   - Verdict for each record
   - Whether the verdict was "Not flagged by this detector" (wrong for tampered)

4. **Compare results**:
   - Baseline rate: % of tampered records flagged as "Not flagged by this detector"
   - Policy rate: % of tampered records flagged as "Not flagged by this detector"
   - Improvement: baseline_rate - policy_rate
   - Requirement: improvement >= 5 percentage points

### Notes
- Synthetic data is generated out-of-repo; commit manifest and scripts are in the repo.
- No synthetic data is included in this repository; the evaluation script generates it.
- The criterion is pre-registered: the threshold (5 percentage points) is fixed before
  data collection and must not be adjusted to achieve a positive result.
- The evaluation must report exact numbers, including failures and edge cases.

## Verification note (2026-10-27)

**Synthetic data is a plumbing check, not evidence about real forgeries.**

Audit findings against this pre-registered criterion:

1. No git history exists in this working copy, so the claim that this
   criterion was registered *before* the results file cannot be verified.
2. `scripts/evaluate_agent.py` implements the "baseline" and "policy" arms as
   **the same call to `agent.verdict.decide`** on the same hand-built records.
   No model inference runs and no policy intervention happens, so the
   improvement is 0.00 pp by construction. This is a single-system plumbing
   run, **not** the agent-vs-baseline comparison the criterion requires.
3. The synthetic records carry hand-written observation dicts with fixed
   score bands (tampered 0.7–0.9, authentic 0.3–0.5); the generated images are
   never scored. The criterion's premise ("hard subset where the baseline
   misses ~55–65%") is therefore not exercised: the measured baseline
   not-flagged rate is 0.00%.

**Phase 10 status: NOT EVALUATED** — the pre-registered agent-vs-baseline
comparison on the pre-registered metric does not exist in
`results/eval_synthetic.json`.
