# Agent policy outcome evidence

Status: **PARTIAL**. Targeted command output from the current work: `pytest tests/test_agent_policy.py -q` → 7 passed; mutation setting `uncertain = False` caused `test_uncertain_score_schedules_ela_and_metadata` to fail; restored version passed. `ruff check src/ app/ tests/ scripts/ agent/` → All checks passed.

## Outcome-changing fixture

The targeted test `test_conditional_evidence_changes_verdict` constructs one record with score 0.9 and invalid box validation. Without conditional comparison, `decide()` returns **Flagged, region unavailable**. The policy requests ELA and metadata and then schedules `compare_evidence`; the fixture records an unresolved score/box conflict, and `decide()` returns **Inconclusive**.

This establishes one synthetic controlled outcome change in the deterministic policy code. It does not yet satisfy the requested minimum of three distinct inputs, does not execute real generated splice images through a stub model, and does not establish a measured policy advantage. Therefore Phase 3 is not DONE and the system label remains **deterministic conditional pipeline**.

## Limitations

The localizer unit test uses a stub detector implementing `predict_raw`; this confirms the inference boundary without loading weights. Evaluation against `synthetic_records()` images and three distinct outcome-changing cases remains outstanding.
