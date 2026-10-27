# Phase-by-phase forensic system report

Audit run: 2026-10-27. DONE requires passing tests, saved evidence in `docs/`, and a mutation test caught by those tests. Statuses below reflect that bar. Exact environment output is in [docs/baseline.md](docs/baseline.md).

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| 0 Environment | BLOCKED | Python 3.11 test environment runs all tests, but project venv dependency installation cannot reach PyPI; sandbox denied `.git` creation, so there is no history baseline. Evidence: `docs/baseline.md`. |
| 1 Stubs and tools | DONE | `tests/test_agent_tools.py` (6 tests cover success/error/timeout for each tool), `tests/test_inspect_image.py::test_digest_matches_file_bytes`; all pass. Digest mutation caused the byte-hash assertion to fail. Tool contract and limits documented in source/config. |
| 2 Wiring | DONE (import-level) | `tests/test_wiring.py` passes and mutation removing the orchestrator import fails. `docs/wiring.md` lists app and library-only modules. App imports but does not invoke the orchestrator; UI inference is not yet routed through it. Exactly one Gradio entrypoint found statically. |
| 3 Policy/outcome agency | PARTIAL | `tests/test_agent_policy.py` (7 tests) passes, including trigger rules, budget=2, invalid-inspect early stop, unresolved conflict, and one conditional outcome change. Disabling uncertainty rule fails its test. Only one outcome-changing input is demonstrated; no three image-backed cases were established. See `docs/agent_evidence.md`. |
| 4 Summary | DONE | `tests/test_gradio.py::test_summary_numeric_parser_rejects_unstored_number` passes; mutation that accepted every number fails the parser test. |
| 5 Optional planner | BLOCKED | No validated planner client/rejection-budget/unavailable tests; orchestration policy outcome evidence is still partial. Planner remains a library-only no-op. |
| 6 Gradio | BLOCKED | Record renderer tests pass, but Gradio is unavailable and network installation failed. App launch and HTTP/screenshot evidence are absent. See `docs/demo_checklist.md`. |
| 7 PDF | BLOCKED | ReportLab is present, but neither pypdf nor pdfplumber is installed and network installation is unavailable. Required extracted-text, thumbnail, page-count, and stable-text checks have not been verified. |
| 8 Batch/ZIP | PARTIAL | `tests/test_batch.py` passes 9 tests including streamed byte cap, traversal, symlink, nested zip, and mixed corrupt image handling. Disabling the streamed cap fails its test. UI wiring, deterministic cancellation, and worker-pool/inference-lock requirements remain incomplete. |
| 9 Email | DONE (deleted) | `tests/test_email_absent.py` passes; README states email is not implemented and no email module exists. Removing that statement fails the test. |
| 10 Evaluation | NOT EVALUATED | Git history unavailable; no preregistration order proof. Existing artifact is plumbing-only with identical verdict calls in both arms, and lacks the requested bootstrap CI/hard-subset comparison. |
| 11 Docker/CI | BLOCKED | Docker CLI cannot connect to daemon; no clean-clone builds or run. CI was inspected but not executed. |

## Raw test and lint output

Command: `/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest tests/ -q`

```text
97 passed, 2 warnings in 38.91s
```

Warnings: Keras skipped optimizer variable loading during model serialization; Keras ignored `shuffle=True` for a `tf.data.Dataset` input.

Command: `/opt/homebrew/bin/ruff check src/ app/ tests/ scripts/ agent/`

```text
All checks passed!
```

## Mutation results

All mutations listed below were temporary and restored before final tests.

| Behavior mutation | Test | Result |
|---|---|---|
| Escape helper returns input unchanged | `test_escaping_handles_script_tags` | Caught: 1 failed; restored: 1 passed |
| Image digest becomes 64 zeroes | `test_digest_matches_file_bytes` | Caught: 1 failed; restored: 1 passed |
| Remove app orchestrator import | `test_app_imports_orchestrator_and_is_single_gradio_entrypoint` | Caught: 1 failed; restored: 1 passed |
| Disable uncertain-score policy trigger | `test_uncertain_score_schedules_ela_and_metadata` | Caught: 1 failed; restored policy test passed |
| Numeric checker always returns true | `test_summary_numeric_parser_rejects_unstored_number` | Caught: 1 failed; restored: 1 passed |
| Disable ZIP streamed-byte guard | `test_streamed_uncompressed_byte_limit` | Caught: 1 failed; restored: 1 passed |
| Change README email status to available | `test_email_not_implemented_and_no_module_exists` | Caught: 1 failed; restored: 1 passed |
| Renderer escaping returns untrusted input unchanged | `test_record_renderer_escapes_untrusted_content_and_shows_verdict` | Caught: 1 failed; restored: 1 passed |

## Wiring table

The entrypoint is `app/app.py` per Dockerfile and README. Static search found one Gradio construction and one launch. The orchestrator is imported by the app but not called; imported dependencies are therefore not yet part of the app's investigation flow.

| Module | App import | Status |
|---|---|---|
| `agent/batch.py` | No | LIBRARY ONLY |
| `agent/config.py` | Yes, transitively | Loaded through orchestrator; not used for app inference |
| `agent/orchestrator.py` | Yes | Imported but not invoked |
| `agent/planner.py` | No | LIBRARY ONLY; unimplemented proposal behavior |
| `agent/policy.py` | Yes, transitively | Loaded through orchestrator; unit-tested policy path |
| `agent/rendering.py` | Yes | WIRED helper; record renderer separately tested |
| `agent/reporting.py` | No | LIBRARY ONLY |
| `agent/schemas.py` | Yes | WIRED |
| `agent/summary.py` | No | LIBRARY ONLY |
| `agent/tool_registry.py` | Yes, transitively | Loaded through orchestrator; not called by app |
| `agent/tools/common.py` | Yes, transitively | Loaded through registry; tool tests pass |
| `agent/tools/compare_evidence.py` | Yes, transitively | Loaded through registry; tool tests pass |
| `agent/tools/generate_ela.py` | Yes, transitively | Loaded through registry; tool tests pass |
| `agent/tools/inspect_image.py` | Yes, transitively | Loaded through registry; tool tests pass |
| `agent/tools/inspect_metadata.py` | Yes, transitively | Loaded through registry; tool tests pass |
| `agent/tools/run_localizer.py` | Yes, transitively | Loaded through registry; tests inject `predict_raw` detector |
| `agent/tools/validate_prediction.py` | Yes, transitively | Loaded through registry; tool tests pass |
| `agent/verdict.py` | Yes | WIRED |
| `scripts/download_data.py` | No | LIBRARY/CLI ONLY |
| `scripts/evaluate_agent.py` | No | LIBRARY/CLI ONLY |
| `scripts/make_plan_pdf.py` | No | LIBRARY/CLI ONLY |

## Phase 10 and Docker

Phase 10 is **NOT EVALUATED**. The evaluation amendment states the requested 5 percentage point target with a 95% bootstrap confidence interval and fixed seed, but cannot be called pre-registered without Git history. Existing synthetic values reproduce as 0.00% baseline, 0.00% policy, and 0.00 percentage-point improvement; both arms call the same `decide()` function, so they are not a policy-versus-baseline experiment. Synthetic data is a plumbing check, not evidence about real forgeries.

Docker status: **NOT VERIFIED**. `docker build -t deepfake-localizer:forensic-audit .` failed because the Docker daemon socket does not exist. Clean-clone builds with and without a checkpoint and `docker run` were not performed.

## System label

**deterministic conditional pipeline**

One synthetic fixture shows a comparison changing a verdict, but the required three distinct image-backed outcome cases are not available in `docs/agent_evidence.md`. Do not call the system agentic.
