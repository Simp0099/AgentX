"""agent/verdict.py: pure function — record → (Verdict, reasons).

No LLM calls. No network. Deterministic mapping from InvestigationRecord
to one of the four allowed verdicts and a list of reason strings.

Vocabulary (exactly these four; "authentic" is banned):
  "Flagged for review"
  "Flagged, region unavailable"
  "Not flagged by this detector"
  "Inconclusive"
"""

from __future__ import annotations

from agent.schemas import InvestigationRecord, Verdict


def decide(record: InvestigationRecord) -> tuple[Verdict, list[str]]:
    """Determine the verdict and reasons from an investigation record.

    Returns (verdict, reasons) where verdict is one of the four allowed
    strings and reasons is a list of explanatory strings, each referencing
    only data stored in the record.

    Rules (in priority order):
      1. Any required check ended in error or timeout  → Inconclusive,
         listing the missing check.
      2. Inspect invalid (image unsupported, too large, etc.) → Inconclusive.
      3. score < threshold and validation valid and no conflict →
         Not flagged by this detector.
      4. score >= threshold and validation valid and no conflict →
         Flagged for review.
      5. score >= threshold and validation invalid → Flagged, region unavailable.
      6. Conflict unresolved after compare_evidence → Inconclusive.
    """
    reasons: list[str] = []

    # --- Collect status of all baseline checks ---
    # observations may be flat or grouped; handle both
    all_obs = []
    for obs in record.observations:
        if isinstance(obs, list):
            all_obs.extend(obs)
        else:
            all_obs.append(obs)

    # --- Collect status of all baseline checks ---
    validate_ok = True  # assume valid unless validate_prediction finds issues
    raw_score: float | None = None
    bbox_invalid = False
    box_coverage_exceeds = False
    inspect_valid = True  # assume valid unless inspect_image finds issues
    evidence_comparison = None

    for obs in all_obs:
        tool = obs.tool
        result = obs.result or {}

        if tool == "inspect_image":
            inspect_valid = result.get("valid", True)
            if not inspect_valid:
                inspect_valid = False

        elif tool == "run_localizer":
            raw_score = result.get("tampered_prob", result.get("score", 0.0))

        elif tool == "validate_prediction":
            validate_ok = True
            issues = result.get("issues", [])
            if issues:
                # Check if any issue is about bbox invalidity
                for issue in issues:
                    if "box" in issue.lower() or "coverage" in issue.lower():
                        bbox_invalid = True
                        # Check if coverage exceeds max
                        if "coverage" in issue.lower():
                            box_coverage_exceeds = True
        elif tool == "compare_evidence":
            evidence_comparison = result

    # --- Rule 1: Any required check ended in error or timeout → Inconclusive ---
    for obs in all_obs:
        if obs.status in ("error", "timeout"):
            missing = obs.tool
            reasons.append(f"{missing} ended in {obs.status}; cannot determine verdict")
            return Verdict.INCONCLUSIVE, reasons

    # --- Rule 2: Inspect invalid → Inconclusive ---
    if not inspect_valid:
        reasons.append("inspect_image reported invalid image; cannot proceed")
        return Verdict.INCONCLUSIVE, reasons

    # --- Score is available from run_localizer ---
    if raw_score is None:
        # No localizer result; can't make a verdict beyond Not flagged
        reasons.append("no localizer score available; returning Not flagged")
        return Verdict.NOT_FLAGGED_BY_THIS_DETECTOR, reasons

    # --- Rule 3: score < threshold and validation valid and no conflict → Not flagged ---
    score_above = raw_score >= record.threshold
    validation_valid = validate_ok and not bbox_invalid and not box_coverage_exceeds

    # Explicit rule: a triggered comparison that still reports conflict is
    # inconclusive. A resolved comparison may update the threshold-only result.
    if evidence_comparison is not None:
        conflicts = evidence_comparison.get("conflicts", [])
        agreements = evidence_comparison.get("agreements", [])
        if conflicts:
            reasons.append("evidence comparison reports unresolved conflict")
            return Verdict.INCONCLUSIVE, reasons
        if agreements and not score_above and not bbox_invalid:
            reasons.append("conditional evidence agrees with tampering signals")
            return Verdict.FLAGGED_FOR_REVIEW, reasons

    if not score_above and validation_valid:
        reasons.append(f"score {raw_score:.4f} below threshold")
        reasons.append(f"threshold {record.threshold}; validation valid")
        return Verdict.NOT_FLAGGED_BY_THIS_DETECTOR, reasons

    # --- Rule 4: score >= threshold and validation valid and no conflict → Flagged for review ---
    if score_above and validation_valid:
        reasons.append(f"score {raw_score:.4f} >= threshold {record.threshold}; validation valid")
        return Verdict.FLAGGED_FOR_REVIEW, reasons

    # --- Rule 5: score >= threshold and validation invalid → Flagged, region unavailable ---
    if score_above and not validation_valid:
        reasons.append(f"score {raw_score:.4f} >= threshold {record.threshold}; validation invalid")
        return Verdict.FLAGGED_REGION_UNAVAILABLE, reasons

    # --- Rule 6: Conflict unresolved → Inconclusive ---
    # (This is reached when score >= threshold, validation is somehow ambiguous,
    #  or box coverage exceeds limit, and it wasn't caught above)
    reasons.append("conflict between score and validation; compare_evidence needed")
    return Verdict.INCONCLUSIVE, reasons
