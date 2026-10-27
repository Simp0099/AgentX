# Demo checklist

Status: **BLOCKED — runtime launch not performed**.

Evidence: the current Python 3.11 environment has no `gradio` module (`find_spec("gradio")` returned false). Installing it in `.venv311` failed because DNS/network access to `pypi.org` is unavailable. The app import and HTTP response/screenshot therefore cannot be truthfully reported. Static check finds the single Blocks construction and launch in `app/app.py`; this is not a runtime verification.

Record renderer test command: `python3.11 -m pytest tests/test_gradio.py::test_record_renderer_escapes_untrusted_content_and_shows_verdict -q` → 1 passed. Mutation replacing escaping with identity → 1 failed; restored implementation → 1 passed.
