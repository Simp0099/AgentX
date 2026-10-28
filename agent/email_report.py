"""Send existing PDF reports through an SMTP server."""

from __future__ import annotations

import logging
import re
import smtplib
import socket
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
    port = int(get_secret("SMTP_PORT") or "587")
    starttls = port == 587 if port in (465, 587) else (
        (get_secret("SMTP_STARTTLS") or "true").strip().lower() == "true"
    )
    return {
        "configured": not missing,
        "missing": missing,
        "host": get_secret("SMTP_HOST") or "smtp.gmail.com",
        "port": port,
        "starttls": starttls,
        "from": get_secret("SMTP_FROM") or username or "",
    }


def smtp_error_message(exc: Exception, port: int | None = None) -> str:
    """Map common SMTP errors to actionable text without exposing credentials."""
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        if exc.smtp_code == 535:
            return (
                "Gmail rejected the login. Use a 16-character App Password, "
                "not your account password."
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
    active_port = port if port is not None else int(email_settings()["port"])
    if isinstance(exc, (TimeoutError, socket.timeout, ConnectionRefusedError)):
        return f"Network blocked outbound SMTP on port {active_port}; try 587"
    return str(exc) or f"Email delivery failed ({type(exc).__name__})."


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


def effective_smtp_config() -> dict[str, str]:
    """Return printable effective settings with the password always masked."""
    settings = email_settings()
    username = get_secret("SMTP_USERNAME") or "<missing>"
    mode = "STARTTLS" if settings["starttls"] else "implicit SSL"
    return {
        "host": str(settings["host"]),
        "port": str(settings["port"]),
        "mode": mode,
        "username": username,
        "password": "********" if get_secret("SMTP_PASSWORD") else "<missing>",
    }


def send_message_via_smtp(message: EmailMessage) -> None:
    """Send one prepared email using port-selected TLS and shared credentials."""
    settings = email_settings()
    if settings["missing"]:
        raise EmailNotConfigured("Missing SMTP configuration: " + ", ".join(settings["missing"]))
    port = int(settings["port"])
    if port not in (465, 587):
        raise ValueError("SMTP_PORT must be 465 (implicit SSL) or 587 (STARTTLS).")
    username = get_secret("SMTP_USERNAME") or ""
    password = "".join((get_secret("SMTP_PASSWORD") or "").split())
    try:
        if port == 465:
            with smtplib.SMTP_SSL(str(settings["host"]), port, timeout=20) as smtp:
                smtp.login(username, password)
                smtp.send_message(message)
        else:
            with smtplib.SMTP(str(settings["host"]), port, timeout=20) as smtp:
                smtp.starttls()
                smtp.login(username, password)
                smtp.send_message(message)
    except Exception as exc:
        _LOGGER.warning(
            "smtp send failed error_type=%s smtp_code=%s port=%s",
            type(exc).__name__,
            smtp_exception_code(exc),
            port,
        )
        raise


def send_test_email(to: str) -> None:
    """Send the short setup message through the same transport as reports."""
    if not _ADDRESS.fullmatch(to):
        raise ValueError("Enter a valid recipient address.")
    settings = email_settings()
    message = EmailMessage()
    message["Subject"] = "Deepfake Localizer email test"
    message["From"] = str(settings["from"])
    message["To"] = to
    message.set_content("Your email delivery setup is working.")
    send_message_via_smtp(message)


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

    settings = email_settings()
    if settings["missing"]:
        raise EmailNotConfigured("Missing SMTP configuration: " + ", ".join(settings["missing"]))

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings["from"]
    message["To"] = ", ".join(to)
    message.set_content(body)
    message.add_attachment(
        candidate.read_bytes(), maintype="application", subtype="pdf", filename=candidate.name
    )
    send_message_via_smtp(message)
    _LOGGER.info("report email sent report_id=%s recipients=%d", candidate.stem, len(to))
