"""Dependency-free helpers for safely rendering untrusted user text."""

import html


def escape_untrusted(text: str | None) -> str:
    """Escape untrusted strings before placing them in HTML or Markdown."""
    if not text:
        return ""
    return html.escape(str(text), quote=False)


def render_record(record) -> str:
    """Render a record as safe Markdown containing verdict and raw score label."""
    verdict = record.verdict.value if record.verdict else "Inconclusive"
    filename = escape_untrusted(record.filename)
    goal_text = escape_untrusted(record.goal_text)
    scores = [
        observation.result.get("tampered_prob")
        for observation in record.observations
        if observation.tool == "run_localizer" and observation.result
    ]
    score_line = (
        f"**tampered score (uncalibrated):** {float(scores[0]):.3f}"
        if scores else "**tampered score (uncalibrated):** unavailable"
    )
    return (
        f"### Verdict: {escape_untrusted(verdict)}\n\n"
        f"**File:** {filename}\n\n**Goal:** {goal_text}\n\n{score_line}"
    )
