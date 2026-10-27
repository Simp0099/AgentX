"""Deterministic policy rules and bounded orchestrator behavior."""

from agent.orchestrator import run_investigation
from agent.policy import StopPolicy, next_checks
from agent.schemas import InvestigationRecord, Observation, ToolResult, Verdict
from agent.tool_registry import tool_registry
from agent.verdict import decide


def record(score=None, goal="quick_check", validation=None, conditional=None):
    observations = [
        Observation("inspect_image", {}, "ok", {"valid": True, "width": 100, "height": 100})
    ]
    if score is not None:
        observations.append(
            Observation(
                "run_localizer", {}, "ok", {"tampered_prob": score, "bbox": [0.1, 0.1, 0.8, 0.8]}
            )
        )
    if validation is not None:
        observations.append(Observation("validate_prediction", {}, "ok", validation))
    observations.extend(conditional or [])
    return InvestigationRecord(
        "id",
        "hash",
        "sample.png",
        goal,
        goal,
        0.5,
        "",
        "stub",
        observations,
        [],
        "",
        created_at_utc="2026-01-01T00:00:00Z",
    )


def test_stop_policy_never_stops_on_a_provisional_score():
    """Regression: comparing a Verdict member to its .value made should_stop always True."""
    r = record(0.9)  # inspect + localizer only; validation has not run yet

    stop, reason = StopPolicy().should_stop(r, max_steps=5)

    assert stop is False
    assert "pending" in reason


def test_stop_policy_stops_once_baseline_is_complete_and_confident():
    r = record(0.2, validation={"valid": True, "issues": []})

    stop, reason = StopPolicy().should_stop(r, max_steps=5)

    assert stop is True
    assert "evidence_consistent" in reason


def test_stop_policy_waits_for_compare_evidence_on_localization_conflict():
    r = record(0.9, validation={"valid": False, "issues": ["box inverted"]})

    stop, reason = StopPolicy().should_stop(r, max_steps=5)

    assert stop is False
    assert "compare_evidence" in reason

    # Once the comparison runs and still reports a conflict the verdict is
    # Inconclusive, so the loop keeps gathering rather than stopping.
    r.observations.append(
        Observation("compare_evidence", {}, "ok", {"agreements": [], "conflicts": ["conflict"]})
    )
    stop, _reason = StopPolicy().should_stop(r, max_steps=5)
    assert stop is False


def test_stop_policy_stops_on_budget():
    r = record(0.9)

    stop, reason = StopPolicy().should_stop(r, max_steps=1)

    assert stop is True
    assert reason == "budget exhausted"


def test_uncertain_score_schedules_ela_and_metadata():
    r = record(0.5)
    assert next_checks(r)[0][0].tool == "validate_prediction"
    r.observations.append(
        Observation("validate_prediction", {}, "ok", {"valid": True, "issues": []})
    )
    assert next_checks(r)[0][0].tool == "generate_ela"
    r.observations.append(Observation("generate_ela", {}, "ok", {"artifact": "ela.png"}))
    assert next_checks(r)[0][0].tool == "inspect_metadata"


def test_high_score_schedules_ela():
    r = record(0.9)
    assert next_checks(r)[0][0].tool == "validate_prediction"
    r.observations.append(
        Observation("validate_prediction", {}, "ok", {"valid": True, "issues": []})
    )
    assert next_checks(r)[0][0].tool == "generate_ela"


def test_origin_goal_schedules_metadata_but_non_origin_does_not():
    origin = record(0.2, "origin camera source")
    assert next_checks(origin)[0][0].tool == "validate_prediction"
    origin.observations.append(
        Observation("validate_prediction", {}, "ok", {"valid": True, "issues": []})
    )
    assert next_checks(origin)[0][0].tool == "inspect_metadata"
    ordinary = record(0.2)
    assert next_checks(ordinary)[0][0].tool == "validate_prediction"


def test_conflict_schedules_compare_and_unresolved_is_inconclusive():
    r = record(
        0.9,
        validation={"valid": False, "issues": ["box inverted"]},
        conditional=[
            Observation("generate_ela", {}, "ok", {}),
            Observation("inspect_metadata", {}, "ok", {}),
        ],
    )
    assert next_checks(r)[0][0].tool == "compare_evidence"
    r.observations.append(
        Observation("compare_evidence", {}, "ok", {"conflicts": ["score-box conflict"]})
    )
    assert decide(r)[0] == Verdict.INCONCLUSIVE


def test_conditional_evidence_changes_verdict():
    baseline = record(0.9, validation={"valid": False, "issues": ["box inverted"]})
    assert decide(baseline)[0] == Verdict.FLAGGED_REGION_UNAVAILABLE
    investigated = record(
        0.9,
        validation={"valid": False, "issues": ["box inverted"]},
        conditional=[
            Observation("generate_ela", {}, "ok", {}),
            Observation("inspect_metadata", {}, "ok", {}),
            Observation("compare_evidence", {}, "ok", {"conflicts": ["score-box conflict"]}),
        ],
    )
    assert decide(investigated)[0] == Verdict.INCONCLUSIVE


def test_budget_two_attempts_exactly_two_calls(monkeypatch):
    attempts = []

    def inspect(path):
        attempts.append("inspect_image")
        return ToolResult("inspect_image", "ok", {"valid": True, "width": 100, "height": 100})

    def localize(path, threshold):
        attempts.append("run_localizer")
        return ToolResult("run_localizer", "ok", {"tampered_prob": 0.2, "bbox": None})

    monkeypatch.setitem(tool_registry["inspect_image"], "callable", inspect)
    monkeypatch.setitem(tool_registry["run_localizer"], "callable", localize)
    record_result = run_investigation(
        "sample.png", config={"threshold": 0.5, "max_tool_calls_per_image": 2}
    )
    assert len(attempts) == 2
    assert record_result.verdict is not None


def test_invalid_inspect_is_inconclusive_without_localizer(monkeypatch):
    attempts = []

    def inspect(path):
        attempts.append("inspect_image")
        return ToolResult("inspect_image", "ok", {"valid": False})

    def localize(path, threshold):
        attempts.append("run_localizer")
        return ToolResult("run_localizer", "ok", {"tampered_prob": 0.9})

    monkeypatch.setitem(tool_registry["inspect_image"], "callable", inspect)
    monkeypatch.setitem(tool_registry["run_localizer"], "callable", localize)
    result = run_investigation(
        "sample.png", config={"threshold": 0.5, "max_tool_calls_per_image": 3}
    )
    assert result.verdict == Verdict.INCONCLUSIVE
    assert attempts == ["inspect_image"]
