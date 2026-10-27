# Gradio demo verification

## Launch command

Command: `python app/app.py` (a temporary PATH shim resolves `python` to Python 3.11.17).

```text
Traceback (most recent call last):
  File "app/app.py", line 9, in <module>
    import gradio as gr
ModuleNotFoundError: No module named 'gradio'
```

The app exited with status 1 before opening the listener.

## HTTP request

Command: `curl -i --max-time 3 http://localhost:7860`

```text
curl: (7) Failed to connect to localhost port 7860 after 0 ms: Couldn't connect to server
```

HTTP status line: **none** (connection refused; server did not start). Response body: **none**. Therefore the UI handler test passes, but live Gradio behavior is not verified in this environment.
