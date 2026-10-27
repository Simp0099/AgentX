# Phase 12 agent acceptance audit

Audit date: 2026-10-27. Evidence was inspected in the Phase 12 checkout. Offline
tests and checked-in reports are not represented as real provider runs.

| # | Acceptance item | Result | Evidence |
|---:|---|---|---|
| 1 | Tools have names, descriptions, typed schemas, and error dicts. | **PASS** | `pytest -q tests/test_agent_tools_api.py tests/test_agent_tools.py` exercises schema format, required typed properties, complete schema-to-`TOOLS` name equality, and error dictionaries. The test `test_tool_names_match_between_tools_and_schemas` asserts both every schema name exists and the registries match. Full-suite evidence: `pytest -q` (see final audit output). |
| 2 | A model chooses tools and different images yield different sequences. | **FAIL** | `reports/agent_eval.json` has `status: "not_run"` and `different_images_different_tool_sequences: "not_run"`. `docs/agent_evidence.md` is explicitly a controlled stub; `docs/trace_synthetic.md` explicitly records the deterministic baseline sequence for every image. Neither is evidence of model-selected variation. No provider run log with two image sequences was found. |
| 3 | System prompt states goal and limits without prescribing tool order. | **PASS** | `pytest -q tests/test_prompts.py`; `test_prompt_has_no_step_by_step_tool_order` checks prohibited order patterns and prompt tests assert goal/limits. |
| 4 | Loop runs Reason, Act, Observe to an answer or step cap; tests cover branches. | **PASS** | `pytest -q tests/test_agent_loop.py tests/test_agent_failures.py`; coverage includes direct answer, tool result on a subsequent model request, malformed call, step cap, retries/provider failures, and final answer. |
| 5 | Short-term memory preserves context across follow-ups. | **PASS** | `pytest -q tests/test_agent_memory.py`; `test_followup_reuses_history_without_another_tool_call` asserts the prior answer remains in the next model request. |
| 6 | Long-term memory reuses saved results for repeated images. | **PASS** | `pytest -q tests/test_agent_memory.py`; `test_same_image_uses_saved_record_without_tools` asserts a zero-step cached result and no second model request. |
| 7 | Tool failures reach the model and the agent retries or falls back. | **PASS** | `pytest -q tests/test_agent_failures.py tests/test_agent_loop.py`; `test_tool_error_is_returned_to_model_and_reflected_in_next_turn` checks the error in the next request, and retry-limit/provider-fallback tests cover recovery/stop behavior. |
| 8 | Agent uncertainty and limits are included in measured benchmark results. | **FAIL** | `reports/agent_eval.json` is `not_run`; `uncertain_or_needs_review_cases` is `null`, so no measured count exists. The prompt requires uncertainty/limits, but that does not satisfy the benchmark criterion. |
| 9 | Accuracy and tool-call counts are measured. | **FAIL** | `reports/agent_eval.json` exists but has `status: "not_run"`, `verdict_accuracy: null`, and `average_tool_calls_per_image: null`. The benchmark script reports that no configured provider credential is present. |
| 10 | Configured model is tested for tool calling. | **PASS** | `configs/agent.yaml` selects `nvidia/nemotron-3.5-lightning:free`; `docs/model_selection.md` has a populated candidate table with the configured model at 5/5 tool-call probes and dated measurements. This is a one-shot tool compatibility test, not forensic accuracy. |

## PDF honest-scope test (A2 and A4)

**FAIL — no qualifying real provider run was available to cite.** A `logs/agent.log`
file exists, but it contains tool-call audit rows, not model message transcripts.
`reports/agent_eval.json` records both behavior checks as `not_run`; the run
script did not have a configured provider credential. The available trace in
`docs/trace_synthetic.md` says all images followed the same deterministic
baseline path. The controlled stub evidence in `docs/agent_evidence.md` is also
not a live model run.

The closest offline unit-test transcript is deliberately not counted as proof:

1. Scripted assistant message in `tests/test_agent_loop.py`:
   `I will inspect the file before judging it.` with an `inspect_image` tool call.
2. The observed tool result is asserted in that test to contain `valid: true`
   and `width: 16`; the next scripted assistant reply is `Two of two.`.

That FakeClient test proves the tool result reaches the next model request, but
the next assistant turn returns a final answer rather than selecting a later
tool. It is neither a real provider run nor evidence that A4's subsequent tool
choice depended on the prior observation. No A2/A4 provider messages are
quoted because none were recorded. A qualifying live transcript and two
different model-selected image sequences remain required before the checklist
can pass this scope test.

## Audit gaps and risks

- The live smoke is marked `live`, skips when neither configured provider key
  name is available, and normal pytest excludes the marker via
  `pyproject.toml`. No live provider run was observed in this audit.
- The synthetic benchmark is designed for 10 authentic and 10 tampered images,
  but its current report contains no metrics because it did not run.
- Phase 11 Docker build/history verification remains unrun because Docker
  Desktop's daemon was unavailable; see `docs/docker_check.md`.
- Provider spend limits are documented for the account owner in
  `docs/deploy_checklist.md` and remain unchecked/unverified.
- Offline scripted tests establish control-flow contracts only. They cannot
  establish live model tool selection, chained evidence-dependent decisions,
  or forensic accuracy.

## Commands run for this audit

```text
$ source .venv311/bin/activate && pytest -q tests/test_agent_tools_api.py tests/test_agent_tools.py tests/test_prompts.py tests/test_agent_loop.py tests/test_agent_failures.py tests/test_agent_memory.py
64 passed in 0.46s

$ pytest -q -m live
1 skipped, 241 deselected in 21.88s

$ python scripts/benchmark_agent.py
Benchmark not run: no configured model credential is present.
benchmark_exit=2

$ ruff check .
All checks passed!

$ pytest -q
241 passed, 1 deselected, 2 warnings in 43.97s

$ secret scan over working text files and all branch history
TEXT_WORKTREE_KEYLIKE_PATHS=0
HISTORY_CHANGED_PATH_HITS=0
TRACKED_ENV_FILENAMES=1
SECRET_SCAN=PASS (no key-like pattern hits; one tracked .env.example placeholder file)
```

The key-like scan reports counts only and does not echo matching content.
