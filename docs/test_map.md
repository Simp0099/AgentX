# Test map and falsifiability

This map records tests present in `tests/` and what their assertions actually establish. “MISSING” means no test asserts the specified behavior. Mutation results are based on this audit turn; dependency-blocked and unrun mutations are explicitly marked.

| Requirement | Test function and file | Asserted behavior | Mutation that should fail / result |
|---|---|---|---|
| Uncertain score schedules ELA and metadata | MISSING | No policy test | N/A |
| Score >= threshold schedules ELA | MISSING | No policy test | N/A |
| Origin goal schedules metadata; non-origin does not | MISSING | No policy test | N/A |
| Conflict schedules compare; unresolved becomes Inconclusive | MISSING | No policy/orchestrator test | N/A |
| Budget 2 gives exactly 2 attempted calls | MISSING | No orchestrator test | N/A |
| Invalid inspect gives Inconclusive and never localizes | MISSING | No orchestrator test | N/A |
| Each check at most once per image | MISSING | No orchestrator test | N/A |
| Verdict always one of four strings | `test_all_four_verdicts_present`, `test_verdict_no_authentic`, `test_verdict_decision_scenarios` — `tests/test_gradio.py` | Enum has four specified values, none contains “authentic”; four fixtures map to four verdicts | Rename one enum value or break a decision rule. Mutation NOT RUN in this turn (test collection is blocked by missing dependencies). |
| Record JSON round-trip | MISSING | No `from_dict` or round-trip test | N/A |
| Two threads at thresholds 0.2 and 0.9 each match own threshold | `test_threshold_not_written_to_shared_state` — `tests/test_inference_contract.py` | Concurrent predictions with 0.1 and 0.9 return valid-shaped outputs; does not use 0.2 or assert threshold-specific labels | Mutate threshold handling to shared state; test should fail. NOT RUN: TensorFlow unavailable. |
| Six tools each have success, failure, timeout tests | PARTIAL: `test_valid_png_image`, `test_nonexistent_file` (`tests/test_inspect_image.py`); validation cases in `tests/test_validate_prediction.py`; none for timeout, and no full matrix for remaining tools | Some image inspection and validation cases exist; inspect_image currently returns a hardcoded result for most paths | Change expected success/error behavior. NOT RUN because full pytest collection fails; timeout mutations have no test. |
| Summary numbers match stored observations | MISSING | No test validates every number; `check_numbers_against_record` checks only score substring | Change an unvalidated displayed number; no test expected to fail. |
| Planner rejects unapproved/malformed, rejection consumes budget, unavailable completes, disabled equals Phase 3 | MISSING | No planner test | N/A |
| Gradio render shows verdict and uncalibrated label, no authentic verdict | PARTIAL: `test_all_four_verdicts_present`, `test_verdict_no_authentic`, `test_summary_contains_uncalibrated_label` — `tests/test_gradio.py` | Checks enum vocabulary and SummaryBuilder label; does not render app output or assert verdict in rendered result | Remove rendered verdict/label. No render-level test exists. |
| PDF 300-char hostile filename escaped | `test_pdf_300_char_filename_script_escape`, `test_escape_text` — `tests/test_reporting.py` | Generates PDF and asserts escaped markup contains entities | Restore identity escaping; NOT RUN this turn (ReportLab unavailable). |
| PDF no EXIF says “absent” | `test_pdf_no_exif_reports_absent` — `tests/test_reporting.py` | Only checks nonempty PDF; does not extract/assert text | Remove “EXIF: absent”; current assertion still passes. |
| PDF 6000x4000 yields thumbnail | `test_pdf_thumbnail_for_large_image` — `tests/test_reporting.py` | Only checks nonempty PDF; source does not load/create a thumbnail | Remove thumbnail logic; assertion still passes. |
| PDF multi-page breaks | `test_pdf_multi_page_record` — `tests/test_reporting.py` | Only checks nonempty PDF; no page count asserted | Remove PageBreaks; assertion may still pass. |
| PDF regeneration gives identical text | `test_pdf_identical_record_same_text` — `tests/test_reporting.py` | Compares PDF byte sizes, not extracted text | Change text but preserve size; no reliable failure. |
| Batch zip-slip | `test_zip_slip_detection` — `tests/test_batch.py` | Encoded traversal member fails with traversal error | Disable traversal rejection; NOT RUN this turn. |
| Batch zip bomb streamed-byte count | MISSING for streamed bytes; `test_zip_bomb_byte_counting` only exceeds member count | Rejects a 6000-member archive; does not exercise streamed-byte limit | Disable streamed byte cap; no test expected to fail. |
| Batch nested zip | `test_nested_zip_detection` — `tests/test_batch.py` | Prints status; contains no assertion | Accept/reject nested archive differently; test cannot catch it. |
| Batch symlink | `test_symlink_member_rejection` — `tests/test_batch.py` | Accepts either COMPLETED or FAILED on a normal file; does not construct symlink | Disable symlink rejection; test cannot catch it. |
| Batch mixed batch completes | `test_mixed_valid_and_corrupt_batch_completes` — `tests/test_batch.py` | Asserts overall status COMPLETED | Force batch status FAILED; NOT RUN this turn. |
| Batch mid-batch cancel | `test_cancellation_mid_batch` — `tests/test_batch.py` | Asserts only that some result exists; completed run passes too | Disable cancellation; test can still pass. |
| Email behaviors or deletion note | MISSING | Email module and email tests absent; README states email unavailable | N/A |

## Mutation execution record

No production mutations were applied during this audit turn. This is deliberate: most mapped requirements have no asserting test, and the available environment cannot collect the full suite (`tensorflow` and `reportlab` are missing). The existing report’s assertion that every listed mutation was run is not supported by evidence available in this checkout. See `FINAL_REPORT.md` and `docs/baseline.md` for exact command output.
