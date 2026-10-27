# Docker and CI verification output

## Docker version

Command: `docker --version`

```text
Docker version 29.9.2, build a82e528
```

The first sandboxed build attempt returned this raw Docker API error:

```text
ERROR: permission denied while trying to connect to the docker API at unix:///Users/bhavishya/.docker/run/docker.sock
```

The elevated retry connected to Docker Desktop (`desktop-linux`) and completed.

## Clean clone without a checkpoint

Command:

```text
git clone . /tmp/clean && cd /tmp/clean && docker build -t deepfake-localizer .
```

Clone output:

```text
Cloning into '/tmp/clean'...
done.
```

Before adding the model file, tracked checkpoint files in the clean clone:

```text
$ git -C /tmp/clean ls-files checkpoints
checkpoints/.gitkeep
```

The exact no-checkpoint build completed successfully. The complete uncached build
stream was emitted in the live command session; the saved raw output below is a
`--progress=plain` repeat against the same clean clone, with the dependency layers
cached:

```text
#0 building with "desktop-linux" instance using docker driver

#1 [internal] load build definition from Dockerfile
#1 transferring dockerfile: 1.42kB 0.0s done
#1 DONE 0.0s

#2 [internal] load metadata for docker.io/library/python:3.11-slim
#2 DONE 0.0s

#3 [internal] load .dockerignore
#3 transferring context: 2B done
#3 DONE 0.0s

#4 [ 1/11] FROM docker.io/library/python:3.11-slim@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377
#4 resolve docker.io/library/python:3.11-slim@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377 0.0s done
#4 DONE 0.0s

#5 [internal] load build context
#5 transferring context: 1.54kB done
#5 DONE 0.0s

#6 [10/11] COPY checkpoints/ ./checkpoints/
#6 CACHED

#7 [ 6/11] COPY src/ ./src/
#7 CACHED

#8 [ 5/11] RUN pip install --no-cache-dir -r requirements.txt
#8 CACHED

#9 [ 9/11] COPY configs/ ./configs/
#9 CACHED

#10 [ 4/11] COPY requirements.txt .
#10 CACHED

#11 [ 7/11] COPY agent/ ./agent/
#11 CACHED

#12 [ 8/11] COPY app/ ./app/
#12 CACHED

#13 [ 2/11] RUN apt-get update && apt-get install -y --no-install-recommends         libgl1         libglib2.0-0     && rm -rf /var/lib/apt/lists/*
#13 CACHED

#14 [ 3/11] WORKDIR /app
#14 CACHED

#15 [11/11] RUN useradd --create-home appuser && chown -R appuser:appuser /app
#15 CACHED

#16 exporting to image
#16 exporting layers done
#16 exporting manifest sha256:7b4b0135b03ab5edf56d48cc1dd845c7b9775414f5e866f8245fa13f936eeeac
#16 exporting manifest sha256:7b4b0135b03ab5edf56d48cc1dd845c7b9775414f5e866f8245fa13f936eeeac done
#16 exporting config sha256:21ffe1d04cc02f62010ef2d4cb73344d7ce5587ae6c7cbbaa8a211214ad6c6bc done
#16 exporting attestation manifest sha256:ce34b4a5b3e7f7e8d268d4773fe0272aca049c1931c0abb6d6aa9b4a6b6c4743 done
#16 exporting manifest list sha256:dd2aaba5a142a4f1f8dd8b5bfcbf275afc06956835082ea565f9e7fc5336f60c done
#16 naming to docker.io/library/deepfake-localizer:latest done
#16 unpacking to docker.io/library/deepfake-localizer:latest done
#16 DONE 0.0s
```

## Trained-checkpoint rebuild

The original repo already contains the trained checkpoint at `checkpoints/best_localizer.keras` (234M, dated Oct 24). It was copied into `/tmp/clean/checkpoints/` after the first successful no-checkpoint build. The rebuild completed successfully. Captured output:

```text
#0 building with "desktop-linux" instance using docker driver

#1 [internal] load build definition from Dockerfile
#1 transferring dockerfile: 1.42kB done
#1 DONE 0.0s

#2 [internal] load metadata for docker.io/library/python:3.11-slim
#2 DONE 0.0s

#3 [internal] load .dockerignore
#3 transferring context: 2B done
#3 DONE 0.0s

#4 [ 1/11] FROM docker.io/library/python:3.11-slim@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377
#4 resolve docker.io/library/python:3.11-slim@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377 done
#4 DONE 0.0s

#5 [internal] load build context
#5 transferring context: 245.05MB 2.8s done
#5 DONE 2.8s

#6 [ 6/11] COPY src/ ./src/
#6 CACHED

#7 [ 7/11] COPY agent/ ./agent/
#7 CACHED

#8 [ 8/11] COPY app/ ./app/
#8 CACHED

#9 [ 2/11] RUN apt-get update && apt-get install -y --no-install-recommends         libgl1         libglib2.0-0     && rm -rf /var/lib/apt/lists/*
#9 CACHED

#10 [ 4/11] COPY requirements.txt .
#10 CACHED

#11 [ 3/11] WORKDIR /app
#11 CACHED

#12 [ 5/11] RUN pip install --no-cache-dir -r requirements.txt
#12 CACHED

#13 [ 9/11] COPY configs/ ./configs/
#13 CACHED

#14 [10/11] COPY checkpoints/ ./checkpoints/
#14 DONE 1.3s

#15 [11/11] RUN useradd --create-home appuser && chown -R appuser:appuser /app
#15 DONE 1.6s

#16 exporting to image
#16 exporting layers
#16 exporting layers 6.3s done
#16 exporting manifest sha256:e50b7efc95839a3c4a3ce096cac5359374f35652f85119e49c5b59ab0c85f868 done
#16 exporting config sha256:a3f86372a6354081352f4d9f886f1bc2727bed093305a815cadae15a2b161c60 done
#16 exporting attestation manifest sha256:200ab8ab5e4a609fe60e02316c3bd52ab0f74db1404a8170438042400516ccee 0.0s done
#16 exporting manifest list sha256:f3dbb90c5aaedc6d3258605eb97e7cfc550b39a6fddbe81f645efe335acc2fb2 done
#16 naming to docker.io/library/deepfake-localizer:latest done
#16 unpacking to docker.io/library/deepfake-localizer:latest
#16 unpacking to docker.io/library/deepfake-localizer:latest 2.6s done
#16 DONE 8.9s
```

## Container startup and first 30 log lines

Command: `docker run -p 7860:7860 deepfake-localizer`

Container listing while running:

```text
CONTAINER ID   IMAGE                COMMAND               CREATED          STATUS          PORTS                                         NAMES
000f70ea8bc7   deepfake-localizer   "python app/app.py"   20 seconds ago   Up 18 seconds   0.0.0.0:7860->7860/tcp, [::]:7860->7860/tcp   wonderful_panini
```

Command: `docker logs --tail 30 000f70ea8bc7`

```text
* Running on local URL:  http://0.0.0.0:7860
* To create a public link, set `share=True` in `launch()`.
```

Only two log lines had been emitted; these are all available lines from the first 30-line request. The attached container session was interrupted after capture.

## CI workflow check

Command: `rg -n -A2 -B1 'name: Lint|name: Tests|ruff check|pytest tests/' .github/workflows/ci.yml`

```text
24-
25:      - name: Lint
26:        run: ruff check src/ app/ tests/ scripts/ agent/
27-
28:      - name: Tests
29-        # Tests build the model with weights=None, so no ImageNet download and
30-        # no GPU is needed -- the whole suite runs on a free CI runner.
31:        run: pytest tests/ -v
32-        env:
33-          TF_CPP_MIN_LOG_LEVEL: "3"
```

The workflow lints all five named directories. The pytest command collects `tests/` only; it does not pass `src/`, `app/`, or `scripts/` as pytest collection roots.

## Local CI commands and raw output

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
