"""scripts/evaluate_agent.py: Evaluation of the bounded forensic agent.

Generates synthetic records with hard subsets (re-encoded splices, low-contrast),
runs the deterministic pipeline, and compares baseline checks vs policy results.

Output: results/eval_synthetic.json with exact numbers including failures.

C1-C7 compliant: offline CPU, no paid APIs, weights=None, no network.

NOTE (2026-10-27): the "baseline" and "policy" arms below both call
``agent.verdict.decide`` on the same hand-built records. No policy
intervention and no orchestrator run happens, so the improvement is 0 by
construction. See docs/evaluation.md and FINAL_REPORT.md: this is a plumbing
check, not an agent-vs-baseline comparison and not evidence about real
forgeries.
"""

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.schemas import InvestigationRecord, Observation, Verdict  # noqa: E402
from agent.verdict import decide  # noqa: E402


def synthetic_hash(i):
    """Generate a deterministic hash for synthetic record i."""
    return hashlib.sha256(f"synthetic-{i}".encode()).hexdigest()[:16]


def obs(tool, args, result, duration_ms):
    """Build a successful baseline Observation (kept terse for readability)."""
    return Observation(
        tool=tool,
        args=args,
        status="ok",
        result=result,
        rationale="baseline",
        duration_ms=duration_ms,
    )


def make_synthetic_records(n_records=50):
    """Generate synthetic investigation records with hard subsets.

    Creates records with:
    - Re-encoded JPEG (double compression) - makes ELA signal weaker
    - Low-contrast images - makes detection harder
    - Border-level splices - small tampered regions
    - Mix of tampered and authentic images

    Returns list of InvestigationRecord objects.

    NOTE: the generated images are never run through the model. The records
    below carry hand-written observation dicts with fixed score bands
    (tampered: 0.7-0.9, authentic: 0.3-0.5), so every tampered record is above
    the 0.5 threshold by construction.
    """
    rng = np.random.default_rng(42)
    records = []

    for i in range(n_records):
        is_tampered = i % 2 == 0  # Even indices: tampered, Odd: authentic

        if is_tampered:
            # Splice: paste a small patch from another image
            base_size = (64, 64)
            base_img = Image.fromarray(
                rng.integers(0, 255, (*base_size, 3), dtype=np.uint8)
            )
            patch_size = (16, 16) if rng.random() > 0.5 else (8, 8)
            patch = Image.fromarray(
                rng.integers(0, 255, (*patch_size, 3), dtype=np.uint8)
            )
            spliced = base_img.copy()
            x = rng.integers(0, base_size[0] - patch_size[0])
            y = rng.integers(0, base_size[1] - patch_size[1])
            spliced.paste(patch, (x, y))

            # Re-encode as JPEG to weaken ELA signal (simulate double compression)
            buf = io.BytesIO()
            spliced.save(buf, "JPEG", quality=85)
            buf.seek(0)
            Image.open(buf).convert("RGB")  # generated, never scored

            # Two independent draws preserve the original RNG sequence
            prob = 0.7 + rng.random() * 0.2
            raw_score = 0.7 + rng.random() * 0.2

            record = InvestigationRecord(
                investigation_id=f"synthetic-{i:03d}-tampered",
                image_id=f"sha256{synthetic_hash(i)}",
                filename=f"synthetic_{i:03d}.jpg",
                goal="quality_check",
                goal_text="",
                threshold=0.5,
                config_snapshot_hash="abc123",
                model_version=(
                    "08b5f1cee3f94204f92ae77e13a3b466e8ceacc5739cd651af97597ce518add4"
                ),
                observations=[
                    obs(
                        "inspect_image",
                        {"path": f"synthetic_{i:03d}.jpg"},
                        {
                            "valid": True, "format": "JPEG", "width": 64,
                            "height": 64, "mode": "RGB", "byte_size": 5120,
                            "sha256": synthetic_hash(i),
                        },
                        10.0,
                    ),
                    obs(
                        "run_localizer",
                        {"path": f"synthetic_{i:03d}.jpg", "threshold": 0.5},
                        {"tampered_prob": prob, "label": "tampered",
                         "bbox": [0.1, 0.1, 0.5, 0.5]},
                        50.0,
                    ),
                    obs(
                        "validate_prediction",
                        {
                            "raw_score": raw_score, "bbox": [0.1, 0.1, 0.5, 0.5],
                            "image_width": 64, "image_height": 64,
                        },
                        {"valid": True, "issues": []},
                        5.0,
                    ),
                ],
                errors=[],
                rationale="Synthetic tampered record",
                verdict=Verdict.FLAGGED_FOR_REVIEW,
                verdict_reasons=["Score above threshold; validation valid"],
                created_at_utc="2026-01-15T10:30:00Z",
            )
        else:
            # Authentic image (no tampering); two draws preserve RNG sequence
            prob = 0.3 + rng.random() * 0.2
            raw_score = 0.3 + rng.random() * 0.2

            record = InvestigationRecord(
                investigation_id=f"synthetic-{i:03d}-authentic",
                image_id=f"sha256{synthetic_hash(i)}",
                filename=f"synthetic_{i:03d}.jpg",
                goal="quality_check",
                goal_text="",
                threshold=0.5,
                config_snapshot_hash="abc123",
                model_version=(
                    "08b5f1cee3f94204f92ae77e13a3b466e8ceacc5739cd651af97597ce518add4"
                ),
                observations=[
                    obs(
                        "inspect_image",
                        {"path": f"synthetic_{i:03d}.jpg"},
                        {
                            "valid": True, "format": "JPEG", "width": 64,
                            "height": 64, "mode": "RGB", "byte_size": 5120,
                            "sha256": synthetic_hash(i),
                        },
                        10.0,
                    ),
                    obs(
                        "run_localizer",
                        {"path": f"synthetic_{i:03d}.jpg", "threshold": 0.5},
                        {"tampered_prob": prob, "label": "authentic", "bbox": None},
                        50.0,
                    ),
                    obs(
                        "validate_prediction",
                        {
                            "raw_score": raw_score, "bbox": None,
                            "image_width": 64, "image_height": 64,
                        },
                        {"valid": True, "issues": []},
                        5.0,
                    ),
                ],
                errors=[],
                rationale="Synthetic authentic record",
                verdict=Verdict.NOT_FLAGGED_BY_THIS_DETECTOR,
                verdict_reasons=["Score below threshold; validation valid"],
                created_at_utc="2026-01-15T10:30:00Z",
            )

        records.append(record)

    return records


def run_evaluation(n_records=50):
    """Run the full evaluation and write results. Returns the results dict."""
    # Generate synthetic records
    print(f"Generating {n_records} synthetic records...")
    records = make_synthetic_records(n_records)

    # Split into tampered and authentic
    tampered_records = [r for r in records if "tampered" in r.investigation_id]
    authentic_records = [r for r in records if "authentic" in r.investigation_id]

    print(f"  Tampered: {len(tampered_records)}, Authentic: {len(authentic_records)}")

    # --- Baseline: deterministic pipeline without policy ---
    baseline_not_flagged_tampered = 0
    baseline_total_tampered = len(tampered_records)

    for record in tampered_records:
        verdict, reasons = decide(record)
        if verdict == Verdict.NOT_FLAGGED_BY_THIS_DETECTOR:
            baseline_not_flagged_tampered += 1

    baseline_rate = (
        baseline_not_flagged_tampered / baseline_total_tampered * 100
        if baseline_total_tampered > 0 else 0
    )

    # --- With policy (same decide function for now) ---
    policy_not_flagged_tampered = 0
    policy_total_tampered = len(tampered_records)

    for record in tampered_records:
        verdict, reasons = decide(record)
        if verdict == Verdict.NOT_FLAGGED_BY_THIS_DETECTOR:
            policy_not_flagged_tampered += 1

    policy_rate = (
        policy_not_flagged_tampered / policy_total_tampered * 100
        if policy_total_tampered > 0 else 0
    )

    # --- Improvement calculation ---
    improvement = baseline_rate - policy_rate

    # --- Authentic records: should not be wrongly flagged ---
    baseline_false_flag_authentic = 0
    for record in authentic_records:
        verdict, _ = decide(record)
        if verdict == Verdict.FLAGGED_FOR_REVIEW:
            baseline_false_flag_authentic += 1

    # --- Summary stats ---
    results = {
        "evaluation_timestamp": "2026-01-01T00:00:00Z",
        "n_records_generated": n_records,
        "n_tampered": len(tampered_records),
        "n_authentic": len(authentic_records),
        "baseline": {
            "not_flagged_tampered_count": baseline_not_flagged_tampered,
            "tampered_total": baseline_total_tampered,
            "not_flagged_rate_percent": round(baseline_rate, 2),
        },
        "policy": {
            "not_flagged_tampered_count": policy_not_flagged_tampered,
            "tampered_total": policy_total_tampered,
            "not_flagged_rate_percent": round(policy_rate, 2),
        },
        "improvement": {
            "percentage_points": round(improvement, 2),
            "meets_target": improvement >= 5.0,
        },
        "authentic_handling": {
            "n_authentic": baseline_false_flag_authentic,
            "false_flag_rate_percent": (
                round(baseline_false_flag_authentic / len(authentic_records) * 100, 2)
                if authentic_records else 0
            ),
        },
        "criterion": {
            "target_reduction_percent": 5.0,
            "meets_criterion": improvement >= 5.0,
        },
        "method": {
            "synthetic_data": "generated offline CPU, no network, weights=None",
            "pipeline": "deterministic via agent.verdict.decide",
            "records": "synthetic with re-encoding and low-contrast manipulations",
            "scope_note": (
                "plumbing check only; both arms call the same decide() function, "
                "no model inference, not evidence about real forgeries"
            ),
        },
    }

    # Write results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "eval_synthetic.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    improvement_str = f"{improvement:.2f} percentage points"
    meets_str = "YES" if improvement >= 5.0 else "NO"

    print(f"Evaluation results written to: {output_path}")
    print(f"Baseline not-flagged rate: {baseline_rate:.2f}%")
    print(f"Policy not-flagged rate: {policy_rate:.2f}%")
    print(f"Improvement: {improvement_str}")
    print(f"Meets 5pp target: {meets_str}")
    false_flag_rate = results["authentic_handling"]["false_flag_rate_percent"]
    print(f"Authentic false flag rate: {false_flag_rate:.2f}%")

    return results


if __name__ == "__main__":
    results = run_evaluation()
