# Gradio demo verification

Command: `python app/app.py` (Python 3.11.17, `.venv311`). The updated app started successfully.

Command: `curl -i --max-time 10 http://localhost:7860 | head -20`

HTTP status: **200 OK**. First 20 response lines:

```text
HTTP/1.1 200 OK
date: Tue, 27 Oct 2026 08:50:23 GMT
server: uvicorn
content-length: 50474
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

The handler calls `run_investigation()` through `app.handler.analyze` and renders the verdict, the label `tampered score (uncalibrated)`, and executed checks. The batch ZIP controls call `agent.batch.process_batch()`.
