"""Phase 3 policy, registry, planner fallback, and grounding contracts."""

from agent.orchestrator import run_investigation
from agent.schemas import ToolResult, Verdict
from agent.tool_registry import tool_registry, validate_args


def _install_baseline(monkeypatch, score=0.2, bbox=None):
    monkeypatch.setitem(
        tool_registry["inspect_image"],
        "callable",
        lambda path: ToolResult(
            "inspect_image",
            "ok",
            {
                "valid": True,
                "width": 100,
                "height": 100,
                "sha256": "a" * 64,
            },
        ),
    )
    monkeypatch.setitem(
        tool_registry["run_localizer"],
        "callable",
        lambda path, threshold: ToolResult(
            "run_localizer",
            "ok",
            {
                "tampered_prob": score,
                "bbox": bbox,
                "model_version": "stub-v1",
            },
        ),
    )
    monkeypatch.setitem(
        tool_registry["validate_prediction"],
        "callable",
        lambda run_result, image_width, image_height: ToolResult(
            "validate_prediction",
            "ok",
            {
                "valid": bbox is None or (bbox[0] < bbox[2] and bbox[1] < bbox[3]),
                "issues": []
                if bbox is None or (bbox[0] < bbox[2] and bbox[1] < bbox[3])
                else ["box coordinates unordered"],
            },
        ),
    )
    monkeypatch.setitem(
        tool_registry["generate_ela"],
        "callable",
        lambda path, quality=None: ToolResult("generate_ela", "ok", {"artifact": "ela.png"}),
    )
    monkeypatch.setitem(
        tool_registry["inspect_metadata"],
        "callable",
        lambda path: ToolResult("inspect_metadata", "ok", {"exif_status": "absent"}),
    )
    monkeypatch.setitem(
        tool_registry["compare_evidence"],
        "callable",
        lambda record: ToolResult(
            "compare_evidence",
            "ok",
            {
                "agreements": [],
                "conflicts": ["score-box conflict"],
            },
        ),
    )


def test_registry_rejects_unapproved_names_and_malformed_arguments():
    assert validate_args("run_shell", {})[0] is False
    assert validate_args("generate_ela", {"path": "/tmp/x", "quality": "high"})[0] is False


def test_baseline_budget_exhaustion_stops_after_two_tool_calls(monkeypatch):
    _install_baseline(monkeypatch)
    result = run_investigation(
        "sample.png", config={"threshold": 0.5, "max_tool_calls_per_image": 2}
    )
    assert [obs.tool for obs in result.observations] == ["inspect_image", "run_localizer"]


def test_conflict_case_runs_compare_and_is_inconclusive(monkeypatch):
    _install_baseline(monkeypatch, score=0.9, bbox=[0.8, 0.2, 0.1, 0.9])
    result = run_investigation(
        "sample.png", config={"threshold": 0.5, "max_tool_calls_per_image": 10}
    )
    assert "compare_evidence" in [obs.tool for obs in result.observations]
    assert result.verdict == Verdict.INCONCLUSIVE


def test_llm_unavailable_falls_back_to_deterministic_policy(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    _install_baseline(monkeypatch)
    result = run_investigation(
        "sample.png", config={"threshold": 0.5, "max_tool_calls_per_image": 5}
    )
    assert result.planner_status == "unavailable"
    assert result.verdict == Verdict.NOT_FLAGGED_BY_THIS_DETECTOR
    assert [obs.tool for obs in result.observations[:3]] == [
        "inspect_image",
        "run_localizer",
        "validate_prediction",
    ]


def test_planner_rejects_unapproved_proposal_and_counts_it_against_budget(monkeypatch):
    _install_baseline(monkeypatch)

    class FakePlanner:
        def propose(self, record, config):
            return [{"tool": "read_file", "args": {"path": "/etc/passwd"}}]

        def summarize(self, record):
            return "A concise summary."

    result = run_investigation(
        "sample.png",
        config={"threshold": 0.5, "max_tool_calls_per_image": 4},
        planner=FakePlanner(),
    )
    assert result.observations[-1].status == "rejected"
    assert result.observations[-1].rationale == "unapproved planner tool"
    assert result.planner_status == "available"


def test_malformed_planner_arguments_are_rejected(monkeypatch):
    _install_baseline(monkeypatch)

    class FakePlanner:
        def propose(self, record, config):
            return [{"tool": "generate_ela", "args": {"quality": "high"}}]

        def summarize(self, record):
            return "Summary."

    result = run_investigation(
        "sample.png",
        config={"threshold": 0.5, "max_tool_calls_per_image": 4},
        planner=FakePlanner(),
    )
    assert result.observations[-1].status == "rejected"
    assert result.observations[-1].rationale == "malformed planner arguments"


def test_ungrounded_llm_summary_is_discarded(monkeypatch):
    _install_baseline(monkeypatch)

    class FakePlanner:
        def propose(self, record, config):
            return []

        def summarize(self, record):
            return "The score was 0.999."

    result = run_investigation(
        "sample.png",
        config={"threshold": 0.5, "max_tool_calls_per_image": 5},
        planner=FakePlanner(),
    )
    assert "0.999" not in result.summary
    assert any("summary discarded" in error for error in result.errors)
