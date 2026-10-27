"""Offline evaluation plumbing using generated image/mask records only.

The repository has CASIA2 labels locally, but no manifest labels a real subset
as both re-encoded and low-contrast. Results are therefore PLUMBING ONLY and
cannot evaluate the pre-registered hard-subset criterion.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import get_config  # noqa: E402
from agent.orchestrator import run_investigation  # noqa: E402
from agent.schemas import InvestigationRecord, Observation, ToolResult, Verdict  # noqa: E402
from agent.summary import SummaryBuilder  # noqa: E402
from agent.tools.inspect_image import inspect_image  # noqa: E402
from agent.tools.run_localizer import run_localizer  # noqa: E402
from agent.tools.validate_prediction import validate_prediction  # noqa: E402
from agent.verdict import decide  # noqa: E402
from src.data.dataset import AUTHENTIC, TAMPERED, synthetic_records  # noqa: E402
from src.inference import TamperingDetector  # noqa: E402


def _observation(result: ToolResult) -> Observation:
    return Observation(
        tool=result.tool,
        args={},
        status=result.status,
        result=result.result,
        duration_ms=result.duration_ms,
        rationale=result.error or "evaluation baseline check",
    )


def _make_low_contrast_reencoded(records, output_dir: Path, config: dict):
    """Return derived synthetic paths while preserving source labels and boxes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    transformed = []
    for index, (path, label, bbox) in enumerate(records):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        if label == TAMPERED and bbox is not None:
            width, height = image.size
            xmin, ymin, xmax, ymax = bbox
            left, top = int(xmin * width), int(ymin * height)
            right, bottom = max(left + 1, int(xmax * width)), max(top + 1, int(ymax * height))
            patch = np.asarray(image, dtype=np.float32)
            outside_mask = np.ones((height, width), dtype=bool)
            outside_mask[top:bottom, left:right] = False
            surrounding_mean = (
                patch[outside_mask].mean(axis=0) if outside_mask.any() else patch.mean(axis=(0, 1))
            )
            region = patch[top:bottom, left:right]
            blend = float(config["low_contrast_blend"])
            patch[top:bottom, left:right] = (1.0 - blend) * region + blend * surrounding_mean
            image = Image.fromarray(np.clip(patch, 0, 255).astype(np.uint8))
        destination = output_dir / f"sample_{index:04d}.jpg"
        image.save(destination, format="JPEG", quality=int(config["first_reencode_quality"]))
        # A second JPEG round-trip deliberately weakens compression-history cues.
        with Image.open(destination) as first_encode:
            first_encode.convert("RGB").save(
                destination, format="JPEG", quality=int(config["final_reencode_quality"])
            )
        transformed.append((str(destination), label, bbox, label == TAMPERED))
    return transformed


def _make_record(path: str, label: int, bbox, detector, threshold: float):
    image_check = inspect_image(path)
    localizer = run_localizer(path, threshold=threshold, detector=detector)
    image_data = image_check.result or {}
    localizer_data = localizer.result or {}
    validation = validate_prediction(
        run_result=localizer_data,
        image_width=image_data.get("width"),
        image_height=image_data.get("height"),
    )
    record = InvestigationRecord(
        investigation_id=f"baseline-{Path(path).stem}",
        image_id=(image_data.get("sha256") or hashlib.sha256(path.encode()).hexdigest()),
        filename=Path(path).name,
        goal="quick_check",
        goal_text="",
        threshold=threshold,
        config_snapshot_hash="synthetic-evaluation",
        model_version=str(localizer_data.get("model_version") or "unknown"),
        observations=[_observation(image_check), _observation(localizer), _observation(validation)],
        errors=[
            result.error for result in (image_check, localizer, validation) if result.status != "ok"
        ],
        rationale="baseline checks only",
        sha256=str(image_data.get("sha256") or ""),
        image_path=path,
        config_snapshot={},
    )
    record.verdict, record.verdict_reasons = decide(record)
    return record


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores, strict=True) if label == TAMPERED]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == AUTHENTIC]
    if not positives or not negatives:
        return None
    wins = sum(
        (positive > negative) + 0.5 * (positive == negative)
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _classification_metrics(labels: list[int], scores: list[float], threshold: float) -> dict:
    predicted = [int(score >= threshold) for score in scores]
    accuracy = sum(actual == guess for actual, guess in zip(labels, predicted, strict=True)) / max(
        len(labels), 1
    )
    class_f1 = []
    for class_label in (AUTHENTIC, TAMPERED):
        tp = sum(
            actual == class_label and guess == class_label
            for actual, guess in zip(labels, predicted, strict=True)
        )
        fp = sum(
            actual != class_label and guess == class_label
            for actual, guess in zip(labels, predicted, strict=True)
        )
        fn = sum(
            actual == class_label and guess != class_label
            for actual, guess in zip(labels, predicted, strict=True)
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        class_f1.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(class_f1) / 2, 4),
        "roc_auc": round(_roc_auc(labels, scores), 4)
        if _roc_auc(labels, scores) is not None
        else None,
        "threshold": threshold,
        "n": len(labels),
    }


def _box_iou(first, second) -> float:
    if first is None or second is None:
        return 0.0
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def _conditional_expected(record) -> set[str]:
    score = next(
        (
            (obs.result or {}).get("tampered_prob")
            for obs in record.observations
            if obs.tool == "run_localizer"
        ),
        None,
    )
    validation = next(
        (obs.result or {} for obs in record.observations if obs.tool == "validate_prediction"), {}
    )
    if not isinstance(score, (int, float)):
        return set()
    config = get_config()["forensics"]
    uncertain = float(config["uncertain_low"]) <= score <= float(config["uncertain_high"])
    flagged = score >= record.threshold
    issues = validation.get("issues", [])
    conflict = (flagged and any("box" in str(issue).lower() for issue in issues)) or any(
        "coverage" in str(issue).lower() for issue in issues
    )
    expected = set()
    if uncertain or flagged or conflict:
        expected.add("generate_ela")
    if uncertain or conflict:
        expected.add("inspect_metadata")
    if conflict:
        expected.add("compare_evidence")
    return expected


def _batch_fault_smoke(root: Path) -> dict:
    from agent.batch import BatchStatus, process_mixed_batch

    image_buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "gray").save(image_buffer, format="PNG")
    zip_path = root / "fault_smoke.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("valid.png", image_buffer.getvalue())
        archive.writestr("corrupt.png", b"not an image")
    result = process_mixed_batch(str(zip_path), str(root))
    statuses = [item["status"] for item in result.get("results", [])]
    return {
        "status": result.get("status"),
        "injected_corrupt_member": "corrupt" in statuses,
        "valid_member_retained": "valid" in statuses,
        "batch_completed": result.get("status") == BatchStatus.COMPLETED,
    }


def run_evaluation(n_per_class: int | None = None, threshold: float | None = None) -> dict:
    """Run synthetic baseline, summary, and full-policy arms, then write JSON."""
    evaluation_config = get_config().get("evaluation", {})
    n_per_class = int(n_per_class or evaluation_config.get("synthetic_records_per_class", 6))
    threshold = float(threshold if threshold is not None else get_config()["output"]["threshold"])
    print(f"Generating synthetic_records() with n_per_class={n_per_class}...")
    root_dir = Path(__file__).resolve().parent.parent
    checkpoint = root_dir / "checkpoints" / "best_localizer.keras"
    if checkpoint.is_file():
        detector = TamperingDetector(checkpoint)
        predictor_type = "local trained checkpoint"
    else:

        class StubDetector:
            """Controlled predictor for a plumbing run without model weights."""

            model_version = "synthetic-plumbing-stub"

            def predict_raw(self, _image):
                return {
                    "tampered_prob": 0.5,
                    "bbox_raw": None,
                    "image_size": (1, 1),
                    "model_version": self.model_version,
                }

        detector = StubDetector()
        predictor_type = "controlled stub (checkpoint unavailable)"
    labels, scores, boxes, ious = [], [], [], []
    baseline_records, full_records = [], []
    baseline_latency, full_latency = [], []

    with tempfile.TemporaryDirectory(prefix="forensic-eval-") as temporary:
        temporary_root = Path(temporary)
        source_records = synthetic_records(
            str(temporary_root / "source"),
            n=n_per_class,
            size=int(evaluation_config.get("synthetic_image_size", 128)),
            seed=42,
        )
        hard_records = _make_low_contrast_reencoded(
            source_records,
            temporary_root / "hard_subset",
            evaluation_config,
        )
        for image_path, label, ground_truth_box, _low_contrast_splice in hard_records:
            start = time.perf_counter()
            baseline = _make_record(image_path, label, ground_truth_box, detector, threshold)
            baseline_latency.append((time.perf_counter() - start) * 1000)
            labels.append(label)
            localizer = next(
                (obs.result or {} for obs in baseline.observations if obs.tool == "run_localizer"),
                {},
            )
            score = localizer.get("tampered_prob")
            if score is not None:
                scores.append(float(score))
                if label == TAMPERED:
                    boxes.append(localizer.get("bbox"))
                    ious.append(_box_iou(localizer.get("bbox"), ground_truth_box))
            baseline_records.append(baseline)

            start = time.perf_counter()
            full = run_investigation(
                image_path,
                goal="quick_check",
                config={"threshold": threshold, "max_tool_calls_per_image": 10},
                planner=False,
                detector=detector,
            )
            full_latency.append((time.perf_counter() - start) * 1000)
            full_records.append(full)

        baseline_tampered = [
            record
            for record, label in zip(baseline_records, labels, strict=True)
            if label == TAMPERED
        ]
        full_tampered = [
            record for record, label in zip(full_records, labels, strict=True) if label == TAMPERED
        ]
        baseline_not_flagged = sum(
            record.verdict == Verdict.NOT_FLAGGED_BY_THIS_DETECTOR for record in baseline_tampered
        )
        full_not_flagged = sum(
            record.verdict == Verdict.NOT_FLAGGED_BY_THIS_DETECTOR for record in full_tampered
        )
        baseline_rate = baseline_not_flagged / max(len(baseline_tampered), 1) * 100
        full_rate = full_not_flagged / max(len(full_tampered), 1) * 100

        b_summary_grounded = [
            SummaryBuilder(record).is_grounded(SummaryBuilder(record).build())
            for record in baseline_records
        ]
        full_grounded = [
            SummaryBuilder(record).is_grounded(record.summary) for record in full_records
        ]
        tool_relevance, unnecessary_rates = [], []
        workflow_complete = []
        ambiguous_results = []
        for record, label in zip(full_records, labels, strict=True):
            actual = {obs.tool for obs in record.observations if obs.status != "rejected"}
            actual_conditional = actual & {"generate_ela", "inspect_metadata", "compare_evidence"}
            expected = _conditional_expected(record)
            true_positive = len(actual_conditional & expected)
            precision = (
                true_positive / len(actual_conditional)
                if actual_conditional
                else (1.0 if not expected else 0.0)
            )
            recall = true_positive / len(expected) if expected else 1.0
            tool_relevance.append(
                2 * precision * recall / (precision + recall) if precision + recall else 0.0
            )
            unnecessary_rates.append(
                len(actual_conditional - expected) / max(len(actual_conditional), 1)
            )
            workflow_complete.append(
                {"inspect_image", "run_localizer", "validate_prediction"}.issubset(actual)
                and record.verdict is not None
            )
            score = next(
                (
                    (obs.result or {}).get("tampered_prob")
                    for obs in record.observations
                    if obs.tool == "run_localizer"
                ),
                None,
            )
            if score is not None and label == TAMPERED:
                ambiguous_results.append(record.verdict == Verdict.INCONCLUSIVE)

        metrics_scores = [
            float(
                (
                    next(
                        (
                            obs.result or {}
                            for obs in record.observations
                            if obs.tool == "run_localizer"
                        ),
                        {},
                    )
                ).get("tampered_prob")
            )
            for record in baseline_records
            if (
                next(
                    (
                        obs.result or {}
                        for obs in record.observations
                        if obs.tool == "run_localizer"
                    ),
                    {},
                )
            ).get("tampered_prob")
            is not None
        ]
        metrics_labels = [
            label
            for record, label in zip(baseline_records, labels, strict=True)
            if next(
                (
                    (obs.result or {}).get("tampered_prob")
                    for obs in record.observations
                    if obs.tool == "run_localizer"
                ),
                None,
            )
            is not None
        ]
        dataset_manifest = {
            "source": "src.data.dataset.synthetic_records",
            "synthetic_records_per_class": n_per_class,
            "hard_subset_transform": (
                "tampered patch contrast blended toward surrounding mean; "
                "all images JPEG re-encoded twice"
            ),
            "records": len(hard_records),
            "designated_ambiguous_subset": sum(item[3] for item in hard_records),
        }
        results = {
            "evaluation_status": "PLUMBING ONLY",
            "criterion_status": "NOT EVALUATED",
            "criterion_status_reason": (
                "CASIA2 labels exist locally, but no manifest identifies a real subset "
                "tagged as both re-encoded and low-contrast; synthetic records cannot "
                "evaluate the pre-registered criterion."
            ),
            "n_hard_subset": 0,
            "synthetic_plumbing": {
                "dataset_manifest": dataset_manifest,
                "predictor": predictor_type,
                "classification_metrics": _classification_metrics(
                    metrics_labels, metrics_scores, threshold
                ),
                "localization_mean_iou_on_tampered": round(float(np.mean(ious)), 4)
                if ious
                else None,
                "baseline_not_flagged_tampered": {
                    "count": baseline_not_flagged,
                    "total": len(baseline_tampered),
                    "rate_percent": round(baseline_rate, 2),
                },
                "full_policy_not_flagged_tampered": {
                    "count": full_not_flagged,
                    "total": len(full_tampered),
                    "rate_percent": round(full_rate, 2),
                },
                "difference_percentage_points": round(baseline_rate - full_rate, 2),
                "agent_metrics": {
                    "tool_selection_relevance_f1": round(float(np.mean(tool_relevance)), 4)
                    if tool_relevance
                    else None,
                    "summary_grounding_rate_baseline": round(
                        sum(b_summary_grounded) / len(b_summary_grounded), 4
                    ),
                    "summary_grounding_rate_full": round(
                        sum(full_grounded) / len(full_grounded), 4
                    ),
                    "workflow_completion_rate": round(
                        sum(workflow_complete) / len(workflow_complete), 4
                    ),
                    "unnecessary_call_rate": round(float(np.mean(unnecessary_rates)), 4)
                    if unnecessary_rates
                    else None,
                    "ambiguous_subset_inconclusive_rate": round(
                        sum(ambiguous_results) / len(ambiguous_results), 4
                    )
                    if ambiguous_results
                    else None,
                    "mean_latency_ms_baseline": round(float(np.mean(baseline_latency)), 2),
                    "mean_latency_ms_full": round(float(np.mean(full_latency)), 2),
                    "llm_tokens_per_investigation": None,
                    "llm_status": "disabled for offline plumbing run",
                },
                "batch_resilience": _batch_fault_smoke(temporary_root),
                "systems": {
                    "A": {
                        "name": "fixed baseline checks only, deterministic verdict",
                        "tampered_not_flagged_rate_percent": round(baseline_rate, 2),
                        "classification_metrics": _classification_metrics(
                            metrics_labels, metrics_scores, threshold
                        ),
                        "localization_mean_iou_on_tampered": round(float(np.mean(ious)), 4)
                        if ious
                        else None,
                    },
                    "B": {
                        "name": "baseline checks plus grounded template summary; LLM unavailable",
                        "tampered_not_flagged_rate_percent": round(baseline_rate, 2),
                        "summary_grounding_rate": round(
                            sum(b_summary_grounded) / len(b_summary_grounded), 4
                        ),
                        "llm_tokens_per_investigation": None,
                    },
                    "C": {
                        "name": "full conditional policy without an LLM planner",
                        "tampered_not_flagged_rate_percent": round(full_rate, 2),
                        "tool_selection_relevance_f1": round(float(np.mean(tool_relevance)), 4)
                        if tool_relevance
                        else None,
                        "workflow_completion_rate": round(
                            sum(workflow_complete) / len(workflow_complete), 4
                        ),
                    },
                },
            },
            "criterion": {
                "target_reduction_percentage_points": 5,
                "bootstrap_confidence_interval_95": None,
                "bootstrap_seed": 42,
                "zero_excluded": None,
                "status": "NOT EVALUATED",
            },
            "limitations": [
                "Synthetic records and transformations are plumbing checks, "
                "not benchmark evidence.",
                "The registered hard-subset metric and bootstrap interval were not evaluated.",
                "The optional LLM planner and token metrics were disabled.",
            ],
        }

    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "eval_synthetic.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"Evaluation results written to: {output_path}")
    print("Evaluation label: PLUMBING ONLY")
    print("Criterion status: NOT EVALUATED")
    record_count = results["synthetic_plumbing"]["dataset_manifest"]["records"]
    print(f"Synthetic records evaluated: {record_count}")
    return results


if __name__ == "__main__":
    run_evaluation()
