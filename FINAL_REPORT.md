# Evidence-based phase report

This report uses the requested evidence files: `docs/baseline.md`, `docs/agent_evidence.md`, `docs/demo_checklist.md`, `docs/pdf_check.txt`, `docs/evaluation.md`, `results/`, and `docs/docker_check.md`.

System label: **agentic conditional investigation pipeline**. `docs/agent_evidence.md` records three synthetic image cases in which the conditional evidence comparison changes the verdict from “Flagged, region unavailable” to “Inconclusive.” This label describes those observed cases; it is not a performance claim.

## Per-phase status

| Phase | Status | Tests and evidence |
|---|---|---|
| 0. Baseline audit | DONE | `docs/baseline.md` records 100 baseline tests, Ruff output, actual inference contract, Docker behavior, and the threshold-race investigation. It found no threshold race in the audited code. |
| 1. Inference stabilization | DONE | `tests/test_inference_contract.py::test_model_version_hashes_checkpoint_contents`, `test_malformed_model_score_is_rejected`, `test_image_above_configured_pixel_limit_is_rejected`, `test_cli_passes_threshold_to_prediction`; `tests/test_app_handler.py::test_simultaneous_analyze_calls_keep_thresholds_isolated`. |
| 2. Forensic tools | DONE | `tests/test_agent_tools.py::test_inspect_image_success_failure_timeout`, `test_malformed_model_result_fails_prediction_validation`, `test_tools_use_configured_default_timeout`, `test_absent_metadata_is_not_a_conflict`. Tools return bounded `ToolResult` values. |
| 3. Bounded policy and verdict | DONE | `tests/test_orchestrator_contract.py::test_registry_rejects_unapproved_names_and_malformed_arguments`, `test_baseline_budget_exhaustion_stops_after_two_tool_calls`, `test_conflict_case_runs_compare_and_is_inconclusive`, `test_llm_unavailable_falls_back_to_deterministic_policy`, `test_malformed_planner_arguments_are_rejected`, `test_ungrounded_llm_summary_is_discarded`. |
| 4. Gradio integration | DONE | `tests/test_app_handler.py::test_multiple_upload_component_outputs_one_image_without_error`; `docs/demo_checklist.md` records `HTTP/1.1 200 OK` and the first 20 response lines. Multiple uploads, goal selection, score label, executed/not-run checks, errors, and conditional evidence displays are wired. |
| 5. PDF reporting | DONE | `tests/test_reporting.py::test_pdf_300_char_filename_script_escape`, `test_pdf_no_exif_reports_absent`, `test_pdf_thumbnail_for_large_image`, `test_pdf_multi_page_record`, `test_pdf_extracted_text_escaping_absent_exif_and_repeatability`, `test_batch_pdf_contains_required_summary_columns`, `test_pdf_metadata_parse_error_is_not_reported_as_absent`. `docs/pdf_check.txt` records escaped filename, `EXIF: absent`, repeatability, and thumbnail checks as PASS. |
| 6. Batch and ZIP | DONE | `tests/test_batch.py::test_zip_slip_detection`, `test_zip_bomb_compression_ratio_rejected`, `test_symlink_member_rejection`, `test_nested_zip_detection`, `test_mixed_valid_and_corrupt_batch_completes`, `test_cancellation_mid_batch`, `test_valid_image_batch_uses_safe_names_and_removes_job_temp`; 13 selected batch/ZIP tests passed, including the UI wiring test. |
| 7. Email | PARTIAL (optional) | Email remains unimplemented and off. `tests/test_email_absent.py::test_email_not_implemented_and_no_module_exists` verifies absence; the email-send acceptance cases are not implemented. |
| 8. Evaluation | NOT EVALUATED | `results/eval_run.txt` and `results/eval_synthetic.json` are labeled `PLUMBING ONLY`; the registered real hard-subset criterion and bootstrap interval are `NOT EVALUATED`. The synthetic run uses `synthetic_records()` and reports no improvement (0.00 percentage points). |
| Docker and CI | PARTIAL | `docs/docker_check.md` records successful clean-clone builds with and without a checkpoint and startup logs. CI Ruff covers `src/ app/ tests/ scripts/ agent/`; CI pytest collects `tests/` only. |

## Evaluation result

Local CASIA2 labels are present, but no manifest identifies a real labeled subset tagged as both re-encoded and low-contrast. The synthetic run compared the baseline and full deterministic policy on 12 generated/transformed images. The tampered-image “Not flagged by this detector” rate was 100% in both arms; the difference was 0.00 percentage points. The 95% bootstrap CI was not run. No improvement claim is supported.

## Raw pytest output

Command: `source .venv311/bin/activate && python -m pytest tests/ -v`

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.17, pytest-9.1.1, pluggy-1.7.0 -- /Users/bhavishya/Documents/Bhavishya/Code/Projects/Personal Projects/AgentX/AI-Deepfake-Localizer-main/.venv311/bin/python
cachedir: .pytest_cache
rootdir: /Users/bhavishya/Documents/Bhavishya/Code/Projects/Personal Projects/AgentX/AI-Deepfake-Localizer-main
configfile: pyproject.toml
plugins: anyio-4.15.1
collecting ... collected 121 items

tests/test_agent_policy.py::test_uncertain_score_schedules_ela_and_metadata PASSED [  0%]
tests/test_agent_policy.py::test_high_score_schedules_ela PASSED         [  1%]
tests/test_agent_policy.py::test_origin_goal_schedules_metadata_but_non_origin_does_not PASSED [  2%]
tests/test_agent_policy.py::test_conflict_schedules_compare_and_unresolved_is_inconclusive PASSED [  3%]
tests/test_agent_policy.py::test_conditional_evidence_changes_verdict PASSED [  4%]
tests/test_agent_policy.py::test_budget_two_attempts_exactly_two_calls PASSED [  4%]
tests/test_agent_policy.py::test_invalid_inspect_is_inconclusive_without_localizer PASSED [  5%]
tests/test_agent_tools.py::test_inspect_image_success_failure_timeout PASSED [  6%]
tests/test_agent_tools.py::test_validate_prediction_success_failure_timeout PASSED [  7%]
tests/test_agent_tools.py::test_generate_ela_success_failure_timeout PASSED [  8%]
tests/test_agent_tools.py::test_inspect_metadata_success_failure_timeout PASSED [  9%]
tests/test_agent_tools.py::test_inspect_metadata_reports_present_exif_fields PASSED [  9%]
tests/test_agent_tools.py::test_run_localizer_success_failure_timeout PASSED [ 10%]
tests/test_agent_tools.py::test_malformed_model_result_fails_prediction_validation PASSED [ 11%]
tests/test_agent_tools.py::test_tools_use_configured_default_timeout PASSED [ 12%]
tests/test_agent_tools.py::test_compare_evidence_success_failure_timeout PASSED [ 13%]
tests/test_agent_tools.py::test_absent_metadata_is_not_a_conflict PASSED [ 14%]
tests/test_app_handler.py::test_analyze_uses_orchestrator_record_and_renders_verdict PASSED [ 14%]
tests/test_app_handler.py::test_simultaneous_analyze_calls_keep_thresholds_isolated PASSED [ 15%]
tests/test_app_handler.py::test_multiple_upload_component_outputs_one_image_without_error PASSED [ 16%]
tests/test_batch.py::test_zip_slip_detection PASSED                      [ 17%]
tests/test_batch.py::test_zip_bomb_byte_counting PASSED                  [ 18%]
tests/test_batch.py::test_symlink_member_rejection PASSED                [ 19%]
tests/test_batch.py::test_mixed_valid_and_corrupt_batch_completes PASSED [ 19%]
tests/test_batch.py::test_cancellation_mid_batch PASSED                  [ 20%]
tests/test_batch.py::test_nested_zip_detection PASSED                    [ 21%]
tests/test_batch.py::test_streamed_uncompressed_byte_limit PASSED        [ 22%]
tests/test_batch.py::test_zip_bomb_compression_ratio_rejected PASSED     [ 23%]
tests/test_batch.py::test_absolute_zip_path_rejected PASSED              [ 23%]
tests/test_batch.py::test_valid_image_batch_uses_safe_names_and_removes_job_temp PASSED [ 24%]
tests/test_ela.py::test_preserves_size_and_mode PASSED                   [ 25%]
tests/test_ela.py::test_flags_spliced_region PASSED                      [ 26%]
tests/test_ela.py::test_array_shape_and_range PASSED                     [ 27%]
tests/test_ela.py::test_accepts_numpy_and_path PASSED                    [ 28%]
tests/test_ela.py::test_flat_image_does_not_blow_up PASSED               [ 28%]
tests/test_ela.py::test_fixed_scale_is_deterministic PASSED              [ 29%]
tests/test_email_absent.py::test_email_not_implemented_and_no_module_exists PASSED [ 30%]
tests/test_gradio.py::test_verdict_no_authentic PASSED                   [ 31%]
tests/test_gradio.py::test_all_four_verdicts_present PASSED              [ 32%]
tests/test_gradio.py::test_summary_contains_uncalibrated_label PASSED    [ 33%]
tests/test_gradio.py::test_summary_numeric_parser_rejects_unstored_number PASSED [ 33%]
tests/test_gradio.py::test_record_renderer_escapes_untrusted_content_and_shows_verdict PASSED [ 34%]
tests/test_gradio.py::test_escaping_handles_script_tags PASSED           [ 35%]
tests/test_gradio.py::test_verdict_decision_scenarios PASSED             [ 36%]
tests/test_inference_contract.py::test_predict_raw_returns_unthresholded_contract_and_model_input PASSED [ 37%]
tests/test_inference_contract.py::test_predict_threshold_is_per_call_and_confidence_is_predicted_class PASSED [ 38%]
tests/test_inference_contract.py::test_predict_defaults_to_configured_threshold PASSED [ 38%]
tests/test_inference_contract.py::test_malformed_model_score_is_rejected[nan] PASSED [ 39%]
tests/test_inference_contract.py::test_malformed_model_score_is_rejected[-0.01] PASSED [ 40%]
tests/test_inference_contract.py::test_malformed_model_score_is_rejected[1.01] PASSED [ 41%]
tests/test_inference_contract.py::test_invalid_model_box_is_returned_as_none[bbox0] PASSED [ 42%]
tests/test_inference_contract.py::test_invalid_model_box_is_returned_as_none[bbox1] PASSED [ 42%]
tests/test_inference_contract.py::test_invalid_model_box_is_returned_as_none[bbox2] PASSED [ 43%]
tests/test_inference_contract.py::test_invalid_model_box_is_returned_as_none[bbox3] PASSED [ 44%]
tests/test_inference_contract.py::test_model_version_hashes_checkpoint_contents PASSED [ 45%]
tests/test_inference_contract.py::test_cli_passes_threshold_to_prediction PASSED [ 46%]
tests/test_inference_contract.py::test_corrupt_image_path_returns_clean_pillow_error PASSED [ 47%]
tests/test_inference_contract.py::test_missing_image_returns_clean_file_error PASSED [ 47%]
tests/test_inference_contract.py::test_unsupported_extension_is_rejected PASSED [ 48%]
tests/test_inference_contract.py::test_tiny_image_is_supported PASSED    [ 49%]
tests/test_inference_contract.py::test_image_above_configured_pixel_limit_is_rejected PASSED [ 50%]
tests/test_inference_contract.py::test_file_above_configured_byte_limit_is_rejected PASSED [ 51%]
tests/test_inference_contract.py::test_predict_with_overlay_adds_overlay_only_when_box_is_valid PASSED [ 52%]
tests/test_inference_contract.py::test_ela_is_full_resolution_then_resized PASSED [ 52%]
tests/test_inspect_image.py::TestInspectImageSuccess::test_valid_png_image PASSED [ 53%]
tests/test_inspect_image.py::TestInspectImageSuccess::test_sha256_is_64_chars PASSED [ 54%]
tests/test_inspect_image.py::TestInspectImageFailure::test_nonexistent_file PASSED [ 55%]
tests/test_inspect_image.py::TestInspectImageFailure::test_empty_string_path PASSED [ 56%]
tests/test_inspect_image.py::TestInspectImageWithPath::test_with_path_object PASSED [ 57%]
tests/test_inspect_image.py::TestInspectImageWithPath::test_path_object_direct PASSED [ 57%]
tests/test_inspect_image.py::TestInspectImageEdgeCases::test_small_image_no_crash PASSED [ 58%]
tests/test_inspect_image.py::TestInspectImageEdgeCases::test_digest_matches_file_bytes PASSED [ 59%]
tests/test_model.py::test_output_shapes PASSED                           [ 60%]
tests/test_model.py::test_label_output_range PASSED                      [ 61%]
tests/test_model.py::test_bbox_output_range PASSED                       [ 61%]
tests/test_model.py::test_bbox_never_degenerate PASSED                   [ 62%]
tests/test_model.py::test_corner_center_roundtrip PASSED                 [ 63%]
tests/test_model.py::test_masked_bbox_loss_authentic_zero PASSED         [ 64%]
tests/test_model.py::test_masked_bbox_loss_tampered_positive PASSED      [ 65%]
tests/test_model.py::test_masked_bbox_loss_perfect_prediction_zero PASSED [ 66%]
tests/test_model.py::test_mixed_batch_loss PASSED                        [ 66%]
tests/test_model.py::test_model_serialization PASSED                     [ 67%]
tests/test_model.py::test_six_channel_input_required PASSED              [ 68%]
tests/test_orchestrator_contract.py::test_registry_rejects_unapproved_names_and_malformed_arguments PASSED [ 69%]
tests/test_orchestrator_contract.py::test_baseline_budget_exhaustion_stops_after_two_tool_calls PASSED [ 70%]
tests/test_orchestrator_contract.py::test_conflict_case_runs_compare_and_is_inconclusive PASSED [ 71%]
tests/test_orchestrator_contract.py::test_llm_unavailable_falls_back_to_deterministic_policy PASSED [ 71%]
tests/test_orchestrator_contract.py::test_planner_rejects_unapproved_proposal_and_counts_it_against_budget PASSED [ 72%]
tests/test_orchestrator_contract.py::test_malformed_planner_arguments_are_rejected PASSED [ 73%]
tests/test_orchestrator_contract.py::test_ungrounded_llm_summary_is_discarded PASSED [ 74%]
tests/test_pipeline.py::test_training_writes_artifacts PASSED            [ 75%]
tests/test_pipeline.py::test_test_split_is_held_out PASSED               [ 76%]
tests/test_pipeline.py::test_checkpoint_loads_in_a_clean_process PASSED  [ 76%]
tests/test_pipeline.py::test_evaluate_produces_metrics_and_plots PASSED  [ 77%]
tests/test_pipeline.py::test_inference_on_single_image PASSED            [ 78%]
tests/test_reporting.py::test_pdf_300_char_filename_script_escape PASSED [ 79%]
tests/test_reporting.py::test_pdf_no_exif_reports_absent PASSED          [ 80%]
tests/test_reporting.py::test_pdf_thumbnail_for_large_image PASSED       [ 80%]
tests/test_reporting.py::test_pdf_multi_page_record PASSED               [ 81%]
tests/test_reporting.py::test_pdf_identical_record_same_text PASSED      [ 82%]
tests/test_reporting.py::test_escape_text PASSED                         [ 83%]
tests/test_reporting.py::test_pdf_extracted_text_escaping_absent_exif_and_repeatability PASSED [ 84%]
tests/test_reporting.py::test_batch_pdf_contains_required_summary_columns PASSED [ 85%]
tests/test_reporting.py::test_pdf_metadata_parse_error_is_not_reported_as_absent PASSED [ 85%]
tests/test_validate_prediction.py::TestValidatePredictionSuccess::test_valid_score_and_bbox PASSED [ 86%]
tests/test_validate_prediction.py::TestValidatePredictionSuccess::test_authentic_with_no_bbox PASSED [ 87%]
tests/test_validate_prediction.py::TestValidatePredictionSuccess::test_zero_score_authentic PASSED [ 88%]
tests/test_validate_prediction.py::TestValidatePredictionSuccess::test_full_score_tampered PASSED [ 89%]
tests/test_validate_prediction.py::TestValidatePredictionFailure::test_nan_score PASSED [ 90%]
tests/test_validate_prediction.py::TestValidatePredictionFailure::test_score_above_one PASSED [ 90%]
tests/test_validate_prediction.py::TestValidatePredictionFailure::test_score_below_zero PASSED [ 91%]
tests/test_validate_prediction.py::TestValidatePredictionFailure::test_inverted_bbox_x PASSED [ 92%]
tests/test_validate_prediction.py::TestValidatePredictionFailure::test_inverted_bbox_y PASSED [ 93%]
tests/test_validate_prediction.py::TestValidatePredictionFailure::test_bbox_out_of_range_xmin PASSED [ 94%]
tests/test_validate_prediction.py::TestValidatePredictionFailure::test_bbox_out_of_range_xmax PASSED [ 95%]
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
======================= 121 passed, 2 warnings in 40.18s =======================

```

Result: **121 passed, 2 warnings**. The warnings were Keras optimizer-variable loading and the ignored `shuffle=True` option for an already-created `tf.data.Dataset`.

## Raw Ruff output

Command: `ruff check src/ app/ tests/ scripts/ agent/`

```text
All checks passed!
```

## Git log

Command: `git log --oneline -- docs/evaluation.md results/`

```text
ea7a3273 Record synthetic evaluation plumbing run
396756fe Pre-register hard-subset evaluation criterion
48d31e70 baseline before verification
```
