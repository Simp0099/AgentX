# Evidence that the image investigator is agentic

## What the tests demonstrate

The agent loop tests use a scripted `FakeClient`; they make no network calls.
They show the model selecting tools and then receiving the result before it
chooses what to do next:

- `test_tool_result_reaches_the_next_request` has the model request
  `inspect_image`, verifies the tool result is added to the conversation, then
  verifies a second model request produces the final answer.
- `test_record_is_built_from_the_real_tool_calls` verifies that observations in
  the investigation record came from executed model tool calls.
- `test_malformed_json_arguments_are_answered_and_the_loop_continues` and
  `test_non_object_json_arguments_are_treated_as_malformed` show malformed
  calls being returned as tool errors and the model continuing.
- `test_tool_error_is_returned_to_model_and_reflected_in_next_turn` shows a
  missing image reported in the next model request, which then acknowledges
  the failure. `test_localizer_failure_keeps_ela_and_metadata_available` shows
  the model switching to ELA and metadata checks after the localizer fails.
- `test_third_identical_failing_call_is_blocked_after_two_executions` verifies
  that an unchanged failing call is blocked after two executions.
- `test_quota_skips_primary_without_retry_and_uses_first_fallback`,
  `test_transient_retries_twice_then_falls_back`,
  `test_fatal_error_does_not_fall_back`,
  `test_cooled_down_model_is_skipped_on_next_turn`, and
  `test_all_models_exhausted_returns_model_unavailable` exercise provider
  routing and its bounds. `test_retry_after_is_capped_by_configured_maximum`
  checks the retry delay cap.
- Reflection tests are in `tests/test_agent_failures.py` (there is no separate
  `test_reflection.py`). They verify reflection is off by default, and, when
  enabled, only a response with the explicit `CORRECTED:` marker can replace
  the original answer.

Command run:

```text
pytest tests/test_agent_loop.py tests/test_agent_failures.py -q
..........................                                               [100%]
26 passed in 0.19s
```

This is behavioral evidence from offline, scripted tests. It demonstrates a
model-directed tool loop and recovery paths; it does not measure real-provider
quality or establish forensic accuracy.

## Who chooses and executes tools

For the `run_agent_full` path, the model chooses whether to answer or request
one or more tools from the JSON schemas in `agent/agent_tools.py`. The model
also supplies arguments. The loop sends the full conversation, including each
tool result, on the next model call. A tool call is executed only after local
JSON parsing, registry lookup, argument validation, and path checks. The model
cannot replace the detector or inject trusted context such as the upload root.

The first tool is not forced by application code in this path. The system
prompt asks for a short plan, tells the model to gather evidence first, and
describes evidence-dependent tool use. A quick check commonly needs the three
baseline facts—image validity, localizer result, and prediction validation—so
that often means about three tool calls plus a final model call. A simple
scripted example uses one tool call and two model calls. Conditional ELA,
metadata, or evidence-comparison calls add work when the evidence or question
warrants it. Independent tool calls can be batched by the model in one model
turn. These are code/test examples, not a production usage statistic.

The agent configuration sets `max_steps: 6`; a step is one model response,
which may contain more than one tool call. Identical tool calls may execute
twice; a third identical call is blocked. History is bounded by the configured
message and character limits.

## How errors and fallback work

Tool failures are returned as JSON tool messages with an `error` and a
`hint`, associated with the tool call ID. The next model request sees the
failed call and its result, so it can change arguments, use another tool, or
explain what could not be checked. Invalid JSON arguments follow the same
message path. Once an identical failing call reaches its configured limit,
the loop stops with a retry-limit result instead of asking for it indefinitely.

Provider failures are classified in code. Transient errors retry twice by
default; a numeric or date-form `Retry-After` is honored up to
`retry_after_max_seconds`. Quota errors skip retries and put that model into a
process-local cooldown (`cooldown_seconds`, default 900). Fatal errors such as
invalid requests and authentication failures stop routing immediately. The
remaining models are tried in the configured `fallback_models` order.
If no model can answer, `run_agent_full` returns `stopped_reason` of
`model_unavailable` and a `Model unavailable:` result with attempted model
names and causes.

In the chat handler, provider unavailability (and unexpected agent exceptions)
causes a separately labelled deterministic fixed-pipeline run. The structured
reply reports `mode: fixed_pipeline`; this is not a model-selected recovery.

## Decisions that remain fixed in code

These are the decision points found in the agent and chat execution paths.
“Keep fixed” means they are safety or integrity boundaries, not that their
current numeric settings can never be tuned.

| Decision | Current hard-coded/configured behavior | Should stay fixed? |
| --- | --- | --- |
| Available model providers and fallback order | `configs/agent.yaml` defines the primary and ordered fallback list; code routes through that list. | **Yes.** A model must not be allowed to route itself to arbitrary endpoints or providers. Operators can update the reviewed config. |
| Provider error classification and recovery | Code treats quota as skip/cooldown, transient as bounded retry, and fatal as immediate stop. Retry count is two; delay cap and cooldown are config values. | **Yes.** Do not let the model decide whether auth, quota, or transient errors are safe to retry. |
| Model call and repeated-tool budgets | Six model turns by default; at most two executions for an identical tool/argument pair; history is trimmed to configured limits. | **Yes.** These bounds prevent runaway cost and loops; operators may tune them. |
| Tool allowlist, schemas, and argument/path checks | Only registered tools and schema-valid arguments execute; image paths are confined to the configured upload root; detector/context are injected by trusted code. | **Yes.** This is the tool security boundary. |
| Tool failure serialization | Tool errors are normalized to JSON error/hint data; malformed calls are rejected; the model cannot turn a failed result into an executed arbitrary action. | **Yes.** Keep errors visible and structured for recovery without allowing unsafe execution. |
| Grounded verdict and confidence/review labels | `agent.verdict.decide` maps recorded evidence, required tool status, validation issues, score, threshold, and comparison conflicts to the allowed verdicts. `_assess_answer` adds review/confidence annotations from deterministic evidence checks. | **Yes.** The model can explain evidence, but should not override evidence checks or manufacture a verdict. |
| Threshold and forensic validators | Threshold bounds and score/box/coverage validity are checked by code/config. | **Yes.** These are data-integrity rules; the user may set a threshold only within the accepted range. |
| Reflection acceptance | Reflection is off by default. If enabled, code replaces the answer only when the reviewer emits `CORRECTED:`. | **Yes.** A review pass should not silently rewrite the answer without an explicit protocol. |
| Fixed-pipeline order | The fallback `run_investigation` always checks image validity, runs the localizer, validates its prediction, then conditionally runs ELA, metadata, and evidence comparison from fixed rules. It is deliberately not tool-sequence-selected by the model. | **Yes for the fallback.** It must remain predictable when the model is unavailable. The main `run_agent_full` tool sequence is model-selected within the allowlist. |
| Fixed-pipeline conditional triggers | ELA/metadata/comparison are selected from score ranges, threshold crossings, validation issues, and configured goal keywords in `agent/policy.py`. | **Yes for fallback safety and reproducibility.** Review thresholds and trigger rules can be changed by maintainers with tests. |
| Bounded planner/orchestrator path | `agent.orchestrator.run_investigation` runs `next_checks` baseline checks before planner additions; planner proposals are restricted to `PLANNER_TOOLS`, capped by planner/tool budgets, and validated. Its LLM loop also uses deterministic stop rules and accepts planner-written summaries only when numeric claims are grounded in the record. | **Yes.** The planner can add approved checks, but must not skip baseline validation, expand the tool set, exceed budgets, or introduce unsupported numbers. This is a distinct path from the chat handler's `run_agent_full` path. |
| Final fallback summary and unavailable response | The fallback summary is rendered from the record; `Model unavailable:` and step/retry-limit messages are fixed templates. | **Yes.** They must accurately identify the execution mode and avoid presenting a generated claim as a model answer. |
| Caveat wording | `CAVEAT_TEXT` in `agent/prompts.py` is fixed, included in the model prompt and structured chat reply. | **Yes.** The limitations must not be selectively omitted by the model. Text changes should be reviewed as product/policy changes. |
| API auth, upload validation, and size limits | Token dependency, file extension/image checks, upload root, and file bounds are enforced outside the model. | **Yes.** The model must not decide access control or which files are safe to inspect. |

## Gaps and limits

- The tests use fake model responses; no test here checks a live provider,
  end-to-end provider tool-call compatibility, latency, or answer quality.
- A model can choose a poor but schema-valid tool sequence or stop too early.
  The hard step limit bounds the result but does not guarantee completeness.
- The transient/quota/fatal classifier uses exception status, headers, and
  text matching. Provider-specific error formats may be classified imperfectly.
- Cooldown is in process memory only. It resets on restart and is not shared
  across multiple worker processes.
- Reflection is disabled in the shipped config, so the corrected-answer tests
  demonstrate optional behavior, not behavior used on ordinary runs.
- The deterministic fallback policy is not agentic: its tool order and
  conditional triggers are fixed. Its purpose is to remain available when the
  model path fails.
- The separate bounded planner/orchestrator path also has a fixed baseline
  sequence and stop policy. Its planner is permitted to add approved checks,
  not replace those safety checks.
- Model prose can be incomplete or poorly phrased. The structured verdict is
  derived separately from recorded evidence, and the detector score is an
  uncalibrated review signal, not proof of manipulation.
- The tests show that errors reach the model and that calls are bounded; they
  do not show that the model always recovers correctly from a real tool error.
