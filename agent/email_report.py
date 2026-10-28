"""Send existing PDF reports through an SMTP server."""

from __future__ import annotations

import logging
import re
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

from agent.env import get_secret

_LOGGER = logging.getLogger(__name__)
_MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
_ADDRESS = re.compile(r"^[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+$")


class EmailNotConfigured(RuntimeError):
    """SMTP configuration is missing; messages name variables only."""


def email_settings() -> dict[str, object]:
    """Return effective SMTP settings and missing variable names only."""
    username = get_secret("SMTP_USERNAME")
    password = get_secret("SMTP_PASSWORD")
    missing = [
        name
        for name, value in (("SMTP_USERNAME", username), ("SMTP_PASSWORD", password))
        if not value
    ]
    return {
        "configured": not missing,
        "missing": missing,
        "host": get_secret("SMTP_HOST") or "smtp.gmail.com",
        "port": int(get_secret("SMTP_PORT") or "465"),
        "starttls": (get_secret("SMTP_STARTTLS") or "false").strip().lower() == "true",
        "from": get_secret("SMTP_FROM") or username or "",
    }


def smtp_error_message(exc: Exception) -> str:
    """Map common SMTP errors to actionable text without exposing credentials."""
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        if exc.smtp_code in (534, 535):
            return (
                "Gmail rejected the login. Use a 16-character App Password from a Google account "
                "with 2-Step Verification on, not your normal password. "
                "Remove any spaces from the password."
            )
        return (
            f"SMTP authentication failed (code {exc.smtp_code}). "
            "Check SMTP_USERNAME and SMTP_PASSWORD."
        )
    if isinstance(exc, smtplib.SMTPSenderRefused):
        return (
            f"SMTP refused sender address {exc.sender!s} (code {exc.smtp_code}). Check SMTP_FROM."
        )
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        refusals = []
        for address, response in exc.recipients.items():
            code = response[0] if isinstance(response, tuple) and response else "unknown"
            refusals.append(f"{address} (code {code})")
        return "SMTP refused recipient address " + ", ".join(refusals) + "."
    if isinstance(exc, (smtplib.SMTPServerDisconnected, ssl.SSLError, TimeoutError, OSError)):
        return (
            "Could not connect to smtp.gmail.com:465. Check your network or try "
            "SMTP_PORT=587 with SMTP_STARTTLS=true."
        )
    return f"Email delivery failed ({type(exc).__name__}). Check SMTP settings and try again."


def smtp_exception_code(exc: Exception) -> int | str | None:
    """Return SMTP response codes without including any server response text."""
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        codes = [
            response[0]
            for response in exc.recipients.values()
            if isinstance(response, tuple) and response
        ]
        return ",".join(map(str, codes)) or None
    return getattr(exc, "smtp_code", None)


def send_report_email(to: list[str], subject: str, body: str, attachment_path: Path) -> None:
    """Send one report PDF to at most ten validated recipients."""
    if not to or len(to) > 10:
        raise ValueError("Provide between 1 and 10 recipients.")
    if any(not _ADDRESS.fullmatch(address) for address in to):
        raise ValueError("One or more recipient addresses are invalid.")
    reports_dir = (Path.cwd() / "artifacts" / "reports").resolve()
    candidate = Path(attachment_path).expanduser().resolve()
    if not candidate.is_relative_to(reports_dir) or not candidate.is_file():
        raise ValueError("Report attachment is not in the reports directory.")
    if candidate.stat().st_size > _MAX_ATTACHMENT_BYTES:
        raise ValueError("Report attachment exceeds 20 MB.")

    settings = {
        "SMTP_HOST": get_secret("SMTP_HOST") or "smtp.gmail.com",
        "SMTP_PORT": get_secret("SMTP_PORT") or "465",
        "SMTP_USERNAME": get_secret("SMTP_USERNAME"),
        "SMTP_PASSWORD": "".join((get_secret("SMTP_PASSWORD") or "").split()),
        "SMTP_FROM": get_secret("SMTP_FROM") or get_secret("SMTP_USERNAME"),
        "SMTP_STARTTLS": get_secret("SMTP_STARTTLS") or "false",
    }
    missing = [key for key in ("SMTP_USERNAME", "SMTP_PASSWORD") if not settings[key]]
    if missing:
        raise EmailNotConfigured("Missing SMTP configuration: " + ", ".join(missing))
    try:
        port = int(settings["SMTP_PORT"])
    except ValueError:
        raise ValueError("SMTP_PORT must be a number.") from None

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings["SMTP_FROM"] or settings["SMTP_USERNAME"]
    message["To"] = ", ".join(to)
    message.set_content(body)
    message.add_attachment(
        candidate.read_bytes(), maintype="application", subtype="pdf", filename=candidate.name
    )
    try:
        if settings["SMTP_STARTTLS"].strip().lower() == "true":
            with smtplib.SMTP(settings["SMTP_HOST"], port, timeout=20) as smtp:
                smtp.starttls()
                smtp.login(settings["SMTP_USERNAME"], settings["SMTP_PASSWORD"])
                smtp.send_message(message)
        else:
            with smtplib.SMTP_SSL(settings["SMTP_HOST"], port, timeout=20) as smtp:
                smtp.login(settings["SMTP_USERNAME"], settings["SMTP_PASSWORD"])
                smtp.send_message(message)
    except Exception as exc:
        smtp_code = smtp_exception_code(exc)
        _LOGGER.warning(
            "report email failed error_type=%s smtp_code=%s", type(exc).__name__, smtp_code
        )
        raise
    _LOGGER.info("report email sent report_id=%s recipients=%d", candidate.stem, len(to))
