# Phase 0 — baseline audit

Date: 2026-10-27. This audit records the code and test state before Phase 1.
No source code was changed during this phase.

## Test and lint baseline

Existing fresh run from the unchanged source tree:

Command: `source .venv311/bin/activate && python -m pytest tests/ -v`

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.17, pytest-9.1.1, pluggy-1.7.0 -- /Users/bhavishya/Documents/Bhavishya/Code/Projects/Personal Projects/AgentX/AI-Deepfake-Localizer-main/.venv311/bin/python
cachedir: .pytest_cache
rootdir: /Users/bhavishya/Documents/Bhavishya/Code/Projects/Personal Projects/AgentX/AI-Deepfake-Localizer-main
configfile: pyproject.toml
plugins: anyio-4.15.1
collecting ... collected 100 items

tests/test_agent_policy.py::test_uncertain_score_schedules_ela_and_metadata PASSED [  1%]
tests/test_agent_policy.py::test_high_score_schedules_ela PASSED         [  2%]
tests/test_agent_policy.py::test_origin_goal_schedules_metadata_but_non_origin_does_not PASSED [  3%]
tests/test_agent_policy.py::test_conflict_schedules_compare_and_unresolved_is_inconclusive PASSED [  4%]
tests/test_agent_policy.py::test_conditional_evidence_changes_verdict PASSED [  5%]
tests/test_agent_policy.py::test_budget_two_attempts_exactly_two_calls PASSED [  6%]
tests/test_agent_policy.py::test_invalid_inspect_is_inconclusive_without_localizer PASSED [  7%]
tests/test_agent_tools.py::test_inspect_image_success_failure_timeout PASSED [  8%]
tests/test_agent_tools.py::test_validate_prediction_success_failure_timeout PASSED [  9%]
tests/test_agent_tools.py::test_generate_ela_success_failure_timeout PASSED [ 10%]
tests/test_agent_tools.py::test_inspect_metadata_success_failure_timeout PASSED [ 11%]
tests/test_agent_tools.py::test_run_localizer_success_failure_timeout PASSED [ 12%]
tests/test_agent_tools.py::test_compare_evidence_success_failure_timeout PASSED [ 13%]
tests/test_app_handler.py::test_analyze_uses_orchestrator_record_and_renders_verdict PASSED [ 14%]
tests/test_batch.py::test_zip_slip_detection PASSED                      [ 15%]
tests/test_batch.py::test_zip_bomb_byte_counting PASSED                  [ 16%]
tests/test_batch.py::test_symlink_member_rejection PASSED                [ 17%]
tests/test_batch.py::test_mixed_valid_and_corrupt_batch_completes PASSED [ 18%]
tests/test_batch.py::test_cancellation_mid_batch PASSED                  [ 19%]
tests/test_batch.py::test_nested_zip_detection PASSED                    [ 20%]
tests/test_batch.py::test_max_zip_stream_bytes PASSED                    [ 21%]
tests/test_batch.py::test_streamed_uncompressed_byte_limit PASSED        [ 22%]
tests/test_batch.py::test_absolute_zip_path_rejected PASSED              [ 23%]
tests/test_ela.py::test_preserves_size_and_mode PASSED                   [ 24%]
tests/test_ela.py::test_flags_spliced_region PASSED                      [ 25%]
tests/test_ela.py::test_array_shape_and_range PASSED                     [ 26%]
tests/test_ela.py::test_accepts_numpy_and_path PASSED                    [ 27%]
tests/test_ela.py::test_flat_image_does_not_blow_up PASSED               [ 28%]
tests/test_ela.py::test_fixed_scale_is_deterministic PASSED              [ 29%]
tests/test_email_absent.py::test_email_not_implemented_and_no_module_exists PASSED [ 30%]
tests/test_gradio.py::test_verdict_no_authentic PASSED                   [ 31%]
tests/test_gradio.py::test_all_four_verdicts_present PASSED              [ 32%]
tests/test_gradio.py::test_summary_contains_uncalibrated_label PASSED    [ 33%]
tests/test_gradio.py::test_summary_numeric_parser_rejects_unstored_number PASSED [ 34%]
tests/test_gradio.py::test_record_renderer_escapes_untrusted_content_and_shows_verdict PASSED [ 35%]
tests/test_gradio.py::test_escaping_handles_script_tags PASSED           [ 36%]
tests/test_gradio.py::test_verdict_decision_scenarios PASSED             [ 37%]
tests/test_inference_contract.py::TestPredictRaw::test_valid_image_returns_dict_with_expected_keys PASSED [ 38%]
tests/test_inference_contract.py::TestPredictRaw::test_tampered_image_has_nonzero_prob PASSED [ 39%]
tests/test_inference_contract.py::TestPredictRaw::test_predict_raw_no_threshold_applied PASSED [ 40%]
tests/test_inference_contract.py::TestPredict::test_predict_uses_provided_threshold PASSED [ 41%]
tests/test_inference_contract.py::TestPredict::test_predict_defaults_to_config_threshold PASSED [ 42%]
tests/test_inference_contract.py::TestPredict::test_predict_no_threshold_on_authentic PASSED [ 43%]
tests/test_inference_contract.py::TestPredict::test_predict_no_threshold_on_tampered PASSED [ 44%]
tests/test_inference_contract.py::TestPredict::test_predict_returns_bounding_box_when_tampered PASSED [ 45%]
tests/test_inference_contract.py::TestPredict::test_predict_bbox_none_when_authentic PASSED [ 46%]
tests/test_inference_contract.py::TestEdgeCases::test_corrupt_image_clean_error PASSED [ 47%]
tests/test_inference_contract.py::TestEdgeCases::test_unsupported_extension PASSED [ 48%]
tests/test_inference_contract.py::TestEdgeCases::test_tiny_image_1x1 PASSED [ 49%]
tests/test_inference_contract.py::TestEdgeCases::test_very_large_image PASSED [ 50%]
tests/test_inference_contract.py::TestEdgeCases::test_malformed_model_output_rejected PASSED [ 51%]
tests/test_inference_contract.py::TestEdgeCases::test_threshold_not_written_to_shared_state PASSED [ 52%]
tests/test_inspect_image.py::TestInspectImageSuccess::test_valid_png_image PASSED [ 53%]
tests/test_inspect_image.py::TestInspectImageSuccess::test_sha256_is_64_chars PASSED [ 54%]
tests/test_inspect_image.py::TestInspectImageFailure::test_nonexistent_file PASSED [ 55%]
tests/test_inspect_image.py::TestInspectImageFailure::test_empty_string_path PASSED [ 56%]
tests/test_inspect_image.py::TestInspectImageWithPath::test_with_path_object PASSED [ 57%]
tests/test_inspect_image.py::TestInspectImageWithPath::test_path_object_direct PASSED [ 58%]
tests/test_inspect_image.py::TestInspectImageEdgeCases::test_small_image_no_crash PASSED [ 59%]
tests/test_inspect_image.py::TestInspectImageEdgeCases::test_digest_matches_file_bytes PASSED [ 60%]
tests/test_model.py::test_output_shapes PASSED                           [ 61%]
tests/test_model.py::test_label_output_range PASSED                      [ 62%]
tests/test_model.py::test_bbox_output_range PASSED                       [ 63%]
tests/test_model.py::test_bbox_never_degenerate PASSED                   [ 64%]
tests/test_model.py::test_corner_center_roundtrip PASSED                 [ 65%]
tests/test_model.py::test_masked_bbox_loss_authentic_zero PASSED         [ 66%]
tests/test_model.py::test_masked_bbox_loss_tampered_positive PASSED      [ 67%]
tests/test_model.py::test_masked_bbox_loss_perfect_prediction_zero PASSED [ 68%]
tests/test_model.py::test_mixed_batch_loss PASSED                        [ 69%]
tests/test_model.py::test_model_serialization PASSED                     [ 70%]
tests/test_model.py::test_six_channel_input_required PASSED              [ 71%]
tests/test_pipeline.py::test_training_writes_artifacts PASSED            [ 72%]
tests/test_pipeline.py::test_test_split_is_held_out PASSED               [ 73%]
tests/test_pipeline.py::test_checkpoint_loads_in_a_clean_process PASSED  [ 74%]
tests/test_pipeline.py::test_evaluate_produces_metrics_and_plots PASSED  [ 75%]
tests/test_pipeline.py::test_inference_on_single_image PASSED            [ 76%]
tests/test_reporting.py::test_pdf_300_char_filename_script_escape PASSED [ 77%]
tests/test_reporting.py::test_pdf_no_exif_reports_absent PASSED          [ 78%]
tests/test_reporting.py::test_pdf_thumbnail_for_large_image PASSED       [ 79%]
tests/test_reporting.py::test_pdf_multi_page_record PASSED               [ 80%]
tests/test_reporting.py::test_pdf_identical_record_same_text PASSED      [ 81%]
tests/test_reporting.py::test_escape_text PASSED                         [ 82%]
tests/test_reporting.py::test_pdf_extracted_text_escaping_absent_exif_and_repeatability PASSED [ 83%]
tests/test_validate_prediction.py::TestValidatePredictionSuccess::test_valid_score_and_bbox PASSED [ 84%]
tests/test_validate_prediction.py::TestValidatePredictionSuccess::test_authentic_with_no_bbox PASSED [ 85%]
tests/test_validate_prediction.py::TestValidatePredictionSuccess::test_zero_score_authentic PASSED [ 86%]
tests/test_validate_prediction.py::TestValidatePredictionSuccess::test_full_score_tampered PASSED [ 87%]
tests/test_validate_prediction.py::TestValidatePredictionFailure::test_nan_score PASSED [ 88%]
tests/test_validate_prediction.py::TestValidatePredictionFailure::test_score_above_one PASSED [ 89%]
tests/test_validate_prediction.py::TestValidatePredictionFailure::test_score_below_zero PASSED [ 90%]
tests/test_validate_prediction.py::TestValidatePredictionFailure::test_inverted_bbox_x PASSED [ 91%]
tests/test_validate_prediction.py::TestValidatePredictionFailure::test_inverted_bbox_y PASSED [ 92%]
tests/test_validate_prediction.py::TestValidatePredictionFailure::test_bbox_out_of_range_xmin PASSED [ 93%]
tests/test_validate_prediction.py::TestValidatePredictionFailure::test_bbox_out_of_range_xmax PASSED [ 94%]
tests/test_validate_prediction.py::TestValidatePredictionFailure::test_bbox_zero_area PASSED [ 95%]
tests/test_validate_prediction.py::TestValidatePredictionAreaCoverage::test_box_area_below_minimum PASSED [ 96%]
tests/test_validate_prediction.py::TestValidatePredictionAreaCoverage::test_box_covers_too_much PASSED [ 97%]
tests/test_validate_prediction.py::TestValidatePredictionIntegration::test_mixed_validity PASSED [ 98%]
tests/test_wiring.py::test_app_imports_orchestrator_and_is_single_gradio_entrypoint PASSED [ 99%]
tests/test_wiring.py::test_gradio_app_wires_batch_zip_processor PASSED   [100%]

=============================== warnings summary ===============================
tests/test_model.py::test_model_serialization
  /Users/bhavishya/Documents/Bhavishya/Code/Projects/Personal Projects/AgentX/AI-Deepfake-Localizer-main/.venv311/lib/python3.11/site-packages/keras/src/saving/saving_lib.py:869: UserWarning: Skipping variable loading for optimizer 'adam', because it has 46 variables whereas the saved optimizer has 2 variables.
    saveable.load_own_variables(store)

tests/test_pipeline.py::test_training_writes_artifacts
  /Users/bhavishya/Documents/Bhavishya/Code/Projects/Personal Projects/AgentX/AI-Deepfake-Localizer-main/.venv311/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:74: UserWarning: `shuffle=True` was passed, but will be ignored since the data `x` was provided as a tf.data.Dataset. The Dataset is expected to already be shuffled (via `.shuffle(buffer_size)`).
    self.data_adapter = data_adapters.get_data_adapter(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 100 passed, 2 warnings in 43.17s =======================
```

Command: `source .venv311/bin/activate && ruff check src/ app/ tests/ scripts/ agent/`

```text
All checks passed!
```

Result: **100 passed, 2 warnings**; Ruff: **All checks passed!** The two warnings
are Keras optimizer-variable loading and Keras ignoring `shuffle=True` for an
already-built `tf.data.Dataset`.

## Verified model and inference contract

- Model builder: `src.models.localizer.build_model()` in `src/models/localizer.py`.
- Model input: `(N, 224, 224, 6)` by default; RGB is channels 0–2 and the ELA
  image is channels 3–5. `configs/default.yaml` sets `data.image_size: [224,224]`.
- Model outputs: dictionary keys `label` and `bbox`. The label head is sigmoid
  with shape `(N,1)` and represents the tampered score. The box head has shape
  `(N,5)`: normalized `[xmin, ymin, xmax, ymax, has_box]`. The box loss is
  masked by `has_box`, so it trains only on tampered records.
- Inference class: `src.inference.TamperingDetector`. The orchestrator tool
  `agent/tools/run_localizer.py` calls its `predict_raw()` method.
- Default checkpoint: `checkpoints/best_localizer.keras`, from
  `output.checkpoint_dir` and `output.checkpoint_name` in
  `configs/default.yaml`; `app/app.py` permits an environment override through
  `CHECKPOINT`.
- `predict_raw(image)` returns `tampered_prob`, `bbox_raw`, `image_size`, and
  `model_version`. It applies no threshold. It currently sorts box corners and
  always returns the box; it does not validate score finiteness/range or box
  coordinates before returning them.
- `predict(image, threshold=None)` returns `label`, `confidence`,
  `tampered_prob`, `bbox`, `ela_image`, and `image`. `confidence` is the score
  for the selected class; the raw sigmoid is returned separately as
  `tampered_prob`. The per-call threshold defaults to
  `output.threshold: 0.5`. The box is omitted when the threshold does not flag
  the image. `predict_with_overlay()` adds `overlay`.
- The inference lock `_inference_lock` is a process-wide `threading.Lock` used
  around each model call.
- `src/data/ela.py::compute_ela()` computes ELA at the image's original size;
  `ela_array()` resizes the computed ELA map afterward. `preprocess()` uses the
  full-resolution ELA result before forming the resized six-channel input.
- **Model-version discrepancy:** `TamperingDetector.__init__()` currently
  computes SHA-256 over `str(checkpoint.resolve())`, the path string, rather
  than over checkpoint file bytes. The value is stable for a path but does not
  identify checkpoint contents.
- **Input-validation discrepancy:** image pixel and file-byte limits are in
  `configs/default.yaml` (`40,000,000` pixels and `50,000,000` bytes) and
  `agent/tools/inspect_image.py` applies them. Direct `TamperingDetector`
  preprocessing does not enforce those limits. The inference test named
  `test_very_large_image` exercises 4000×3000 (12 MP), below the configured
  limit; `test_malformed_model_output_rejected` does not mock malformed model
  output.

## Threshold-race check

The brief's claimed `app/app.py::analyze()` shared-threshold mutation is not
present in the audited code. `app/app.py::analyze(image, threshold)` delegates
to `app.handler.analyze(image, threshold, detector=DETECTOR)`. The handler
passes the threshold in each investigation's config. `TamperingDetector`
accepts `threshold` per `predict()` call and has no mutable `self.threshold`;
model calls are locked. **The stated race was not reproducible from current
code.** `tests/test_inference_contract.py::test_threshold_not_written_to_shared_state`
uses two threads, but only asserts that both calls return valid results; it does
not assert each verdict against its own threshold. There is no thread test that
calls the Gradio handler with two thresholds.

A separate threshold issue exists in the command-line path: `main()` passes
`--threshold` to `TamperingDetector(...)`, whose constructor does not store or
use that value, then calls `predict_with_overlay()` without the per-call
threshold. In that path the YAML default is used, so the CLI argument has no
effect.

## Clean-clone Docker check

The recent clean-clone verification in `docs/docker_check.md` started with only
tracked `checkpoints/.gitkeep`; no checkpoint file was present. The build
succeeded, contrary to the requested failure expectation. Captured build
result:

```text
#16 naming to docker.io/library/deepfake-localizer:latest done
#16 unpacking to docker.io/library/deepfake-localizer:latest done
#16 DONE 0.0s
```

`.gitignore` contains `checkpoints/*` and `!checkpoints/.gitkeep`; `Dockerfile`
contains `COPY checkpoints/ ./checkpoints/`. Thus the empty directory survives
clone and the image builds without a model checkpoint. A later build after
copying the trained checkpoint also succeeded; see `docs/docker_check.md`.

## Phase 0 outcome

The acceptance document exists and records observed behavior, test/lint
outputs, inference contract, and the threshold-race check. Phase 0 passes.
The stale race claim, checkpoint hash implementation, inference output
validation, direct input limits, and CLI threshold handling are listed as
verified Phase 1 work items; no code changes were made in this phase.
