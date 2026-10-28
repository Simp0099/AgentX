"""Gradio entry point for orchestrated image investigation."""

import json
import logging
import os
import smtplib
import sys
import urllib.error
import urllib.request
from email.message import EmailMessage
from html import escape
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402

from agent.batch import process_batch  # noqa: E402
from agent.env import get_secret  # noqa: E402
from app.chat_handler import agent_chat as _agent_chat  # noqa: E402
from app.handler import analyze as _analyze  # noqa: E402
from app.ui_settings import (  # noqa: E402
    apply_analysis_settings,
    load_ui_settings,
    recent_runs_html,
    record_ui_run,
    save_analysis_settings,
    save_general_settings,
)

CHECKPOINT = os.environ.get("CHECKPOINT", "checkpoints/best_localizer.keras")
DETECTOR = None
LOAD_ERROR = None
LOGGER = logging.getLogger(__name__)
UI_SETTINGS = load_ui_settings()
apply_analysis_settings(UI_SETTINGS)
try:
    from src.inference import TamperingDetector

    DETECTOR = TamperingDetector(CHECKPOINT)
except Exception as exc:  # noqa: BLE001 - missing checkpoint is shown in UI
    LOAD_ERROR = str(exc)


def analyze(image, threshold):
    """Run investigation through the orchestrator and render its record."""
    return _analyze(image, threshold, detector=DETECTOR)


def chat(message, image, state):
    """Route a chat turn through the testable chat handler."""
    result = _agent_chat(message, image, state, detector=DETECTOR)
    return (
        render_chat_card(result),
        result["tool_log"],
        result["ela_image_path"],
        result.get("_state", state),
    )


def render_chat_card(data):
    """Render the structured chat reply as an evidence card."""
    verdict = data.get("verdict") or "Inconclusive"
    icon = {
        "Flagged for review": "🚩",
        "Flagged, region unavailable": "⚠️",
        "Not flagged by this detector": "✅",
        "Inconclusive": "❔",
    }.get(verdict, "❔")
    score = data.get("tampered_score_uncalibrated")
    score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "Unavailable"
    image = data.get("image") or {}
    width, height = image.get("width"), image.get("height")
    size_text = (
        f"{width} × {height} px" if width is not None and height is not None else "Unavailable"
    )
    duration = data.get("processing_seconds")
    duration_text = f"{duration:.3f} s" if isinstance(duration, (int, float)) else "Unavailable"
    region = data.get("region")
    if region:
        horizontal = region["horizontal"]
        vertical = region["vertical"]
        region_html = (
            "<h3>Suspected tampered region</h3><table><thead><tr>"
            "<th>Axis</th><th>From (px)</th><th>To (px)</th></tr></thead><tbody>"
            f"<tr><td>Horizontal</td><td>{escape(str(horizontal[0]))}</td>"
            f"<td>{escape(str(horizontal[1]))}</td></tr>"
            f"<tr><td>Vertical</td><td>{escape(str(vertical[0]))}</td>"
            f"<td>{escape(str(vertical[1]))}</td></tr></tbody></table>"
        )
        coverage = data.get("coverage_percent")
        if isinstance(coverage, (int, float)):
            region_html += f"<p>Covers about {coverage:g}%</p>"
    else:
        region_html = "<h3>Suspected tampered region</h3><p>No region localized</p>"
    answer = escape(str(data.get("answer_text") or "")).replace("\n", "<br>")
    caveat = escape(str(data.get("caveat") or "")).replace("\n", "<br>")
    return (
        '<article class="evidence-card">'
        f"<h2>{icon} {escape(verdict)}</h2>"
        "<table><tbody>"
        f"<tr><th>Verdict</th><td>{escape(verdict)}</td></tr>"
        f"<tr><th>Tampered score (uncalibrated)</th><td>{score_text}</td></tr>"
        f"<tr><th>Image size</th><td>{size_text}</td></tr>"
        f"<tr><th>Processing time</th><td>{duration_text}</td></tr>"
        f"</tbody></table>{region_html}"
        f"<blockquote>{caveat}</blockquote>"
        f"<p>{answer}</p></article>"
    )


def analyze_uploads(files, threshold, goal, custom_text, *, include_ui_outputs=False):
    """Run each uploaded image through the same investigation handler."""
    if not files:
        empty = (
            [],
            [],
            [],
            "Upload one or more images to begin.",
            _stat_cards(0, 0, 0),
            recent_runs_html(),
        )
        return empty if include_ui_outputs else empty[:4]
    if not isinstance(files, list):
        files = [files]
    originals, ela_maps, overlays, details = [], [], [], []
    flagged = 0
    errors = 0
    history_files = []
    for uploaded in files:
        path = getattr(uploaded, "name", uploaded)
        try:
            max_bytes = int(UI_SETTINGS["max_file_size_mb"] * 1_000_000)
            if Path(path).stat().st_size > max_bytes:
                raise ValueError(
                    f"Image exceeds the configured {UI_SETTINGS['max_file_size_mb']} MB limit."
                )
            with Image.open(path) as source:
                original = source.convert("RGB")
            original.filename = Path(path).name
            ela, _scores, overlay, detail = _analyze(
                original,
                threshold,
                detector=DETECTOR,
                goal=goal,
                custom_text=custom_text,
            )
            caption = escape_filename(Path(path).name)
            originals.append((original, caption))
            if ela is not None:
                ela_maps.append((ela, caption))
            if overlay is not None:
                overlay_caption = f"{caption} — Model-predicted tampered region (ELA-based)"
                overlays.append((overlay, overlay_caption))
            details.append(f"## {caption}\n\n{detail}")
            if "### Verdict: **Flagged" in detail:
                flagged += 1
            verdict = next(
                (
                    label
                    for label in (
                        "Flagged for review",
                        "Flagged, region unavailable",
                        "Not flagged by this detector",
                        "Inconclusive",
                    )
                    if f"**{label}**" in detail
                ),
                "Inconclusive",
            )
            history_files.append(
                {
                    "filename": Path(path).name,
                    "verdict": verdict,
                    "score": _scores.get("tampered score (uncalibrated)") if _scores else None,
                }
            )
        except Exception as exc:  # one malformed upload must not hide other results
            errors += 1
            details.append(f"## Upload failed\n\n{type(exc).__name__}: {escape_filename(str(exc))}")
            history_files.append({"filename": Path(path).name, "verdict": "Failed", "score": None})
    stats = _stat_cards(len(originals) + errors, flagged, errors)
    result = (
        originals,
        ela_maps,
        overlays,
        "\n\n---\n\n".join(details),
        stats,
        recent_runs_html(),
    )
    if include_ui_outputs:
        record_ui_run(history_files)
        result = (*result[:5], recent_runs_html())
        return result
    return result[:4]


def analyze_uploads_for_ui(files, threshold, goal, custom_text):
    return analyze_uploads(files, threshold, goal, custom_text, include_ui_outputs=True)


def _stat_cards(files, flagged, errors):
    return (
        '<div class="stat-row">'
        f'<div class="stat-card"><strong>{int(files)}</strong><span>Files analyzed</span></div>'
        f'<div class="stat-card"><strong>{int(flagged)}</strong><span>Flagged</span></div>'
        f'<div class="stat-card"><strong>{int(errors)}</strong><span>Errors</span></div>'
        "</div>"
    )


def escape_filename(value):
    """Escape filenames before rendering them in Markdown captions."""
    from agent.rendering import escape_untrusted

    return escape_untrusted(value)


def analyze_batch(zip_file):
    """Validate and unpack an uploaded image archive through the batch API."""
    if zip_file is None:
        return "Upload a ZIP file to begin."
    zip_path = getattr(zip_file, "name", zip_file)
    result = process_batch(str(zip_path), "artifacts/batch", detector=DETECTOR)
    lines = [f"Batch status: {result['status']}"]
    if result.get("error"):
        lines.append(f"Error: {result['error']}")
    for item in result.get("results", []):
        if item.get("reason"):
            lines.append(f"- {item['filename']}: {item['status']} ({item['reason']})")
            continue
        verdict = f" ({item['verdict']})" if item.get("verdict") else ""
        lines.append(f"- {item['filename']}: {item['status']}{verdict}")
    if result.get("batch_pdf"):
        lines.append(f"Batch PDF: {result['batch_pdf']}")
    if not result.get("results") and not result.get("error"):
        lines.append("- No image files found.")
    return "\n".join(lines)


def confirm_email_recipients(addresses):
    """Show a recipient count before the user confirms report delivery."""
    recipients = [item.strip() for item in (addresses or "").split(",") if item.strip()]
    if not recipients:
        return "Enter at least one recipient."
    if len(recipients) > 10:
        return "A report can be sent to at most 10 recipients."
    return f"Ready to send to {len(recipients)} recipient(s). Click Send to confirm."


def send_report_from_ui(report_path, addresses, note):
    """Send a report after an explicit user click."""
    if not report_path:
        return "Generate a report first."
    recipients = [item.strip() for item in (addresses or "").split(",") if item.strip()]
    try:
        from agent.email_report import send_report_email

        send_report_email(
            recipients,
            f"Report {Path(report_path).stem}",
            note or "Report attached.",
            Path(report_path),
        )
    except Exception as exc:
        return f"Email failed: {exc}"
    return f"Report sent to {len(recipients)} recipient(s)."


def toggle_analysis_panel(is_open):
    next_open = not bool(is_open)
    return next_open, gr.update(visible=next_open)


def save_analysis_controls(
    threshold, file_mb, zip_members, batch_seconds, workers, run_ela, run_metadata
):
    global UI_SETTINGS
    UI_SETTINGS, message = save_analysis_settings(
        threshold, file_mb, zip_members, batch_seconds, workers, run_ela, run_metadata
    )
    return message


def save_general_controls(history_keep, appearance):
    saved, message = save_general_settings(history_keep, appearance)
    history = Path("artifacts/history")
    if history.exists():
        runs = sorted(
            (path for path in history.glob("*.json") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for old in runs[saved["history_keep"] :]:
            old.unlink(missing_ok=True)
    return message


def smtp_settings_for_ui():
    """Return display-only SMTP settings and password status, never its value."""
    from agent.email_report import email_settings

    settings = email_settings()
    password_set = bool(get_secret("SMTP_PASSWORD"))
    return (
        str(settings["host"]),
        str(settings["port"]),
        bool(settings["starttls"]),
        get_secret("SMTP_USERNAME") or "",
        get_secret("SMTP_FROM") or get_secret("SMTP_USERNAME") or "",
        f"SMTP_PASSWORD: {'Set' if password_set else 'Missing'}",
    )


def send_test_email_from_ui(recipient):
    """Send a no-attachment test message to the typed address."""
    from agent.email_report import email_settings, smtp_error_message, smtp_exception_code

    settings = email_settings()
    missing = list(settings["missing"])
    username = get_secret("SMTP_USERNAME") or ""
    password = "".join((get_secret("SMTP_PASSWORD") or "").split())
    if not recipient or "@" not in recipient:
        return "Enter a valid address to receive the test email."
    if missing:
        return "Missing SMTP variables: " + ", ".join(missing)
    message = EmailMessage()
    message["Subject"] = "Deepfake Localizer email test"
    message["From"] = str(settings["from"])
    message["To"] = recipient.strip()
    message.set_content("Your email delivery setup is working.")
    try:
        if settings["starttls"]:
            with smtplib.SMTP(str(settings["host"]), int(settings["port"]), timeout=20) as smtp:
                smtp.starttls()
                smtp.login(username, password)
                smtp.send_message(message)
        else:
            with smtplib.SMTP_SSL(str(settings["host"]), int(settings["port"]), timeout=20) as smtp:
                smtp.login(username, password)
                smtp.send_message(message)
    except Exception as exc:
        LOGGER.warning(
            "test email failed error_type=%s smtp_code=%s",
            type(exc).__name__,
            smtp_exception_code(exc),
        )
        return smtp_error_message(exc)
    return f"Test email sent to {recipient.strip()}."


def model_provider_status():
    from agent.agent_config import get_agent_config, model_entries
    from agent.loop import LAST_USED_MODELS

    providers = []
    for endpoint in model_entries(get_agent_config()):
        provider = str(endpoint["provider"])
        variable = str(endpoint["api_key_env"])
        providers.append(
            {
                "provider": provider,
                "variable": variable,
                "key": bool(get_secret(variable)),
                "model": str(endpoint["model"]),
                "last_used": LAST_USED_MODELS.get(provider, "Not used yet"),
            }
        )
    return providers


def test_provider_connection(provider):
    """Send a one-token request to the selected OpenAI-compatible provider."""
    from agent.agent_config import get_agent_config, model_entries

    endpoint = next(
        (item for item in model_entries(get_agent_config()) if item["provider"] == provider),
        None,
    )
    if endpoint is None:
        return f"Provider {provider} is not configured."
    variable = str(endpoint["api_key_env"])
    key = get_secret(variable)
    if not key:
        return f"Key missing: {variable}. Add it to .env and restart the app."
    payload = json.dumps(
        {
            "model": endpoint["model"],
            "messages": [{"role": "user", "content": "Reply OK."}],
            "max_tokens": 1,
            "temperature": 0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        str(endpoint["base_url"]).rstrip("/") + "/chat/completions",
        data=payload,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
            "accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read(1024)
        from agent.loop import LAST_USED_MODELS

        LAST_USED_MODELS[str(endpoint["provider"])] = str(endpoint["model"])
        return f"Connection succeeded. Model: {endpoint['model']} (1-token test)."
    except urllib.error.HTTPError as exc:
        LOGGER.warning("provider test failed provider=%s http_status=%s", provider, exc.code)
        return f"Connection failed with HTTP {exc.code}. Check {variable} and provider quota."
    except Exception as exc:
        LOGGER.warning(
            "provider test failed provider=%s error_type=%s", provider, type(exc).__name__
        )
        return f"Connection failed ({type(exc).__name__}). Check network and provider settings."


def test_provider_connection_ui(provider):
    from agent.loop import LAST_USED_MODELS

    message = test_provider_connection(provider)
    return message, LAST_USED_MODELS.get(provider, "Not used yet")


def history_settings_status():
    history = Path("artifacts/history")
    count = sum(1 for path in history.glob("*.json") if path.is_file()) if history.exists() else 0
    return f"{count} saved run(s). Retention is applied when new history is saved."


def clear_run_history():
    history = Path("artifacts/history")
    if history.exists():
        for path in history.glob("*.json"):
            if path.is_file():
                path.unlink()
    return "Recent run history cleared.", history_settings_status(), recent_runs_html()


def wire_batch_button(batch_button, batch_input, batch_output):
    return batch_button.click(analyze_batch, inputs=[batch_input], outputs=[batch_output])


DESCRIPTION = """
# Deepfake / Tampering Region Localizer

Upload one or more images to run the deterministic forensic checks. The investigation
record supplies the verdict, uncalibrated score, and executed-check log.

**Limitations.** Trained on CASIA v2 splices. Heavy re-compression,
screenshots, social-media re-encoding, and fully-generated images may not
produce reliable splice signals. Treat output as a signal, not proof.
"""

APP_CSS = Path(__file__).with_name("gradio.css").read_text(encoding="utf-8")

with gr.Blocks(title="Deepfake Tampering Localizer", fill_width=True) as demo:
    settings_open = gr.State(False)
    with gr.Row(elem_classes=["app-header"]):
        gr.Markdown("# Deepfake / Tampering Region Localizer")
        settings_toggle = gr.Button("⚙", elem_id="header-settings-button")
    with gr.Tabs():
        with gr.Tab("Analysis"):
            gr.Markdown(DESCRIPTION)
            if DETECTOR is None:
                gr.Markdown(f"> No trained checkpoint loaded: `{LOAD_ERROR}`")
            with gr.Row(elem_classes=["analysis-layout"]):
                with gr.Column(elem_classes=["left-column"]):
                    image_input = gr.File(
                        type="filepath",
                        file_count="multiple",
                        file_types=["image"],
                        label="Upload one or more images",
                    )
                    goal = gr.Dropdown(
                        choices=["Quick check", "Detailed", "Custom text"],
                        value="Quick check",
                        label="Investigation goal",
                    )
                    custom_goal = gr.Textbox(label="Custom goal", lines=2, max_lines=3)
                    run = gr.Button(
                        "Analyze media", variant="primary", elem_classes=["analyze-button"]
                    )
                    with gr.Column(
                        visible=False, elem_id="analysis-settings-panel"
                    ) as analysis_settings_panel:
                        gr.Markdown("### Analysis settings")
                        threshold = gr.Slider(
                            0.0,
                            1.0,
                            value=UI_SETTINGS["threshold"],
                            step=0.01,
                            label="Detection threshold",
                            info=(
                                f"Active saved default: {UI_SETTINGS['threshold']:.2f}. "
                                "This value is passed to the run."
                            ),
                        )
                        file_size = gr.Number(
                            value=UI_SETTINGS["max_file_size_mb"],
                            minimum=1,
                            maximum=500,
                            step=1,
                            label="Max file size per image (MB)",
                        )
                        zip_members = gr.Number(
                            value=UI_SETTINGS["max_zip_members"],
                            minimum=1,
                            maximum=5000,
                            step=1,
                            label="Max ZIP members",
                        )
                        batch_seconds = gr.Number(
                            value=UI_SETTINGS["batch_time_limit_seconds"],
                            minimum=1,
                            maximum=3600,
                            step=1,
                            label="Batch time limit (seconds)",
                        )
                        max_workers = gr.Number(
                            value=UI_SETTINGS["max_workers"],
                            minimum=1,
                            maximum=32,
                            step=1,
                            label="Max workers",
                        )
                        run_ela = gr.Checkbox(value=UI_SETTINGS["run_ela"], label="Run ELA")
                        run_metadata = gr.Checkbox(
                            value=UI_SETTINGS["run_metadata_inspection"],
                            label="Run metadata inspection",
                        )
                        analysis_save = gr.Button("Save analysis settings")
                        analysis_save_status = gr.Markdown()
                    settings_toggle.click(
                        toggle_analysis_panel,
                        inputs=[settings_open],
                        outputs=[settings_open, analysis_settings_panel],
                        js=(
                            "(isOpen) => { const tab = [...document.querySelectorAll('[role=tab]')]"
                            ".find(item => item.textContent.trim() === 'Analysis'); "
                            "if (tab) tab.click(); return [isOpen]; }"
                        ),
                    )
                    analysis_save.click(
                        save_analysis_controls,
                        inputs=[
                            threshold,
                            file_size,
                            zip_members,
                            batch_seconds,
                            max_workers,
                            run_ela,
                            run_metadata,
                        ],
                        outputs=[analysis_save_status],
                    )
                    with gr.Accordion("Batch ZIP analysis", open=False):
                        batch_input = gr.File(label="Upload an image ZIP", file_types=[".zip"])
                        batch_button = gr.Button("Process batch")
                        batch_output = gr.Textbox(label="Batch result", lines=5, interactive=False)
                        wire_batch_button(batch_button, batch_input, batch_output)
                with gr.Column(elem_classes=["right-column"]):
                    stats_output = gr.HTML(_stat_cards(0, 0, 0))
                    with gr.Row(elem_classes=["right-workspace"]):
                        with gr.Column(elem_classes=["right-main"]):
                            original_output = gr.Gallery(
                                label="Results by file",
                                columns=2,
                                height="auto",
                                elem_classes=["result-gallery"],
                            )
                            overlay_output = gr.Gallery(
                                label="Region overlays",
                                columns=2,
                                height="auto",
                                elem_classes=["result-gallery"],
                            )
                            ela_output = gr.Gallery(
                                label="ELA evidence",
                                columns=2,
                                height="auto",
                                elem_classes=["ela-gallery"],
                            )
                            with gr.Accordion("Ask about this image", open=False):
                                chat_image = gr.Image(type="filepath", label="Image for follow-up")
                                chat_message = gr.Textbox(
                                    label="Question", placeholder="Is this edited?", lines=2
                                )
                                chat_submit = gr.Button("Ask", variant="primary")
                                chat_answer = gr.HTML(label="Answer")
                                with gr.Accordion("Tool calls and evidence", open=False):
                                    chat_tool_log = gr.JSON(
                                        label="Calls, arguments, status, and result preview"
                                    )
                                chat_ela = gr.Image(label="ELA (Error Level Analysis)")
                            chat_state = gr.State(
                                {
                                    "history": [],
                                    "image_path": None,
                                    "image_hash": None,
                                    "filename": "image",
                                }
                            )
                            chat_submit.click(
                                chat,
                                inputs=[chat_message, chat_image, chat_state],
                                outputs=[chat_answer, chat_tool_log, chat_ela, chat_state],
                            )
                        with gr.Column(elem_classes=["results-sidebar"]):
                            detail_output = gr.Markdown("Detailed findings appear after analysis.")
                            gr.Markdown("### Recent runs")
                            recent_runs_output = gr.HTML(recent_runs_html())
                            gr.Markdown("### Email report")
                            email_report_path = gr.Textbox(label="Existing report PDF path")
                            email_addresses = gr.Textbox(label="Recipients (comma-separated)")
                            email_note = gr.Textbox(label="Optional note", lines=2, max_lines=3)
                            email_confirmation = gr.Markdown()
                            with gr.Row():
                                email_confirm = gr.Button("Confirm recipients")
                                email_send = gr.Button("Send")
                            email_status = gr.Markdown()
                            email_confirm.click(
                                confirm_email_recipients,
                                inputs=[email_addresses],
                                outputs=[email_confirmation],
                            )
                            email_send.click(
                                send_report_from_ui,
                                inputs=[email_report_path, email_addresses, email_note],
                                outputs=[email_status],
                            )
            run.click(
                analyze_uploads_for_ui,
                inputs=[image_input, threshold, goal, custom_goal],
                outputs=[
                    original_output,
                    ela_output,
                    overlay_output,
                    detail_output,
                    stats_output,
                    recent_runs_output,
                ],
            )
            image_input.change(
                lambda files: (
                    getattr(files[0], "name", files[0])
                    if isinstance(files, list) and files
                    else getattr(files, "name", files)
                ),
                inputs=[image_input],
                outputs=[chat_image],
            )

        with gr.Tab("Settings"):
            gr.Markdown("## Account and app settings")
            gr.Markdown("### Email delivery")
            gr.Markdown(
                "SMTP values below are read from `.env`. Edit `.env` and restart the app "
                "to change them; the password is never displayed.",
                elem_classes=["settings-note"],
            )
            with gr.Row():
                smtp_host = gr.Textbox(label="SMTP host", interactive=False)
                smtp_port = gr.Textbox(label="SMTP port", interactive=False)
                smtp_starttls = gr.Checkbox(label="STARTTLS", interactive=False)
            with gr.Row():
                smtp_username = gr.Textbox(label="SMTP username", interactive=False)
                smtp_from = gr.Textbox(label="Sender address", interactive=False)
                smtp_password_status = gr.Textbox(label="Password status", interactive=False)
            smtp_recipient = gr.Textbox(label="Test email recipient", placeholder="you@example.com")
            test_email_button = gr.Button("Test email")
            test_email_status = gr.Markdown()
            test_email_button.click(
                send_test_email_from_ui, inputs=[smtp_recipient], outputs=[test_email_status]
            )
            gr.Markdown("### Model providers")
            for provider in model_provider_status():
                provider_label = {
                    "openrouter": "OpenRouter",
                    "gemini": "Gemini",
                    "opencode_zen": "OpenCode Zen",
                }.get(provider["provider"], provider["provider"])
                with gr.Group(elem_classes=["provider-card"]):
                    gr.Markdown(
                        f"**{escape(provider_label)}** · `{escape(provider['variable'])}` · "
                        f"Key {'set' if provider['key'] else 'missing'}"
                    )
                    gr.Markdown(f"Configured model: `{escape(provider['model'])}` · Last used: ")
                    last_used = gr.Markdown(escape(provider["last_used"]))
                    test_button = gr.Button(f"Test {provider_label} connection (1 token)")
                    test_status = gr.Markdown()
                    test_button.click(
                        fn=lambda name=provider["provider"]: test_provider_connection_ui(name),
                        outputs=[test_status, last_used],
                    )
            gr.Markdown("### History")
            general_defaults = load_ui_settings()
            history_keep = gr.Number(
                value=general_defaults["history_keep"],
                minimum=1,
                maximum=100,
                step=1,
                label="Recent runs to keep",
            )
            history_status = gr.Markdown(value=history_settings_status())
            clear_history_button = gr.Button("Clear history", variant="stop")
            clear_history_status = gr.Markdown()
            clear_history_button.click(
                clear_run_history,
                outputs=[clear_history_status, history_status, recent_runs_output],
            )
            gr.Markdown("### Appearance")
            appearance = gr.Radio(
                ["Light", "Dark"], value=general_defaults["appearance"], label="Color theme"
            )
            appearance_toggle = gr.Button("☼  Toggle light / dark")
            general_save = gr.Button("Save settings")
            general_save_status = gr.Markdown()
            general_save.click(
                save_general_controls,
                inputs=[history_keep, appearance],
                outputs=[general_save_status],
                js=(
                    "(keep, mode) => { document.body.dataset.theme = "
                    "String(mode).toLowerCase(); return [keep, mode]; }"
                ),
            )
            demo.load(
                smtp_settings_for_ui,
                inputs=[],
                outputs=[
                    smtp_host,
                    smtp_port,
                    smtp_starttls,
                    smtp_username,
                    smtp_from,
                    smtp_password_status,
                ],
            )
            appearance.change(
                fn=None,
                inputs=[appearance],
                js=(
                    "(mode) => { document.body.dataset.theme = "
                    "String(mode).toLowerCase(); return []; }"
                ),
            )
            appearance_toggle.click(
                fn=lambda mode: "Dark" if mode == "Light" else "Light",
                inputs=[appearance],
                outputs=[appearance],
                js=(
                    "(mode) => { const next = mode === 'Light' ? 'Dark' : 'Light'; "
                    "document.body.dataset.theme = next.toLowerCase(); return [mode]; }"
                ),
            )
            demo.load(
                fn=None,
                inputs=[appearance],
                js=(
                    "(mode) => { document.body.dataset.theme = "
                    "String(mode).toLowerCase(); return []; }"
                ),
            )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        css=APP_CSS,
    )
