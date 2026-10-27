# Gradio demo verification

Command: `source .venv311/bin/activate && python app/app.py` (Python 3.11.17, `.venv311`). The updated app started successfully.

Command: `curl -i --max-time 10 http://localhost:7860 | head -20`

HTTP status: **200 OK**. First 20 response lines:

```text
HTTP/1.1 200 OK
date: Tue, 27 Oct 2026 10:49:19 GMT
server: uvicorn
content-length: 72055
content-type: text/html; charset=utf-8

<!doctype html>
<html
	lang="en"
	style="
		margin: 0;
		padding: 0;
		min-height: 100%;
		display: flex;
		flex-direction: column;
	"
>
	<head>
		<meta charset="utf-8" />
		<base href="./" />
```

The Gradio component uses multiple image uploads, a Quick check / Detailed / Custom text goal selector, and a per-investigation threshold. Results use per-image galleries and show the verdict, `tampered score (uncalibrated)`, executed checks, checks not run, and errors. ELA is shown only when `generate_ela` ran. The overlay gallery is populated only for a flagged verdict with a valid box; its caption is `Model-predicted tampered region (ELA-based)`. The batch ZIP controls remain connected to `agent.batch.process_batch()`.

Automated UI output check: `tests/test_app_handler.py::test_multiple_upload_component_outputs_one_image_without_error`.
