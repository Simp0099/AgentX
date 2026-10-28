from __future__ import annotations

import logging
import smtplib
import ssl

import pytest
from fastapi.testclient import TestClient

from agent import api, email_report


class FakeSMTP:
    instance = None

    def __init__(self, *args, **kwargs):
        self.args = args
        FakeSMTP.instance = self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def login(self, username, password):
        self.credentials = (username, password)

    def starttls(self):
        self.tls = True

    def send_message(self, message):
        self.message = message


@pytest.fixture
def configured(monkeypatch, tmp_path):
    reports = tmp_path / "artifacts" / "reports"
    reports.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(email_report.smtplib, "SMTP_SSL", FakeSMTP)
    for key, value in {
        "SMTP_USERNAME": "user@example.com",
        "SMTP_PASSWORD": "secret-value",
        "SMTP_FROM": "reports@example.com",
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "465",
    }.items():
        monkeypatch.setenv(key, value)
    report = reports / ("deepfake-localizer-" + "a" * 32 + ".pdf")
    report.write_bytes(b"%PDF-fake")
    return report


def test_email_attaches_pdf_and_sets_from(configured):
    email_report.send_report_email(["to@example.com"], "Subject", "Body", configured)
    message = FakeSMTP.instance.message
    assert message["From"] == "reports@example.com"
    attachment = next(part for part in message.walk() if part.get_filename())
    assert attachment.get_content_type() == "application/pdf"
    assert attachment.get_payload(decode=True) == b"%PDF-fake"


def test_email_limits_recipients_and_attachment_size(configured, monkeypatch):
    with pytest.raises(ValueError, match="10 recipients"):
        email_report.send_report_email([f"{n}@example.com" for n in range(11)], "", "", configured)
    monkeypatch.setattr(email_report, "_MAX_ATTACHMENT_BYTES", 1)
    with pytest.raises(ValueError, match="20 MB"):
        email_report.send_report_email(["to@example.com"], "", "", configured)


def test_unconfigured_error_lists_names_only(configured, monkeypatch):
    monkeypatch.delenv("SMTP_USERNAME")
    monkeypatch.delenv("SMTP_PASSWORD")
    with pytest.raises(email_report.EmailNotConfigured) as error:
        email_report.send_report_email(["to@example.com"], "", "", configured)
    assert "SMTP_USERNAME" in str(error.value) and "SMTP_PASSWORD" in str(error.value)
    assert "secret-value" not in str(error.value)


def test_secret_is_not_logged(configured, caplog):
    with caplog.at_level(logging.INFO):
        email_report.send_report_email(
            ["private@example.com"], "Subject", "secret-value", configured
        )
    assert "secret-value" not in caplog.text
    assert "private@example.com" not in caplog.text


def test_email_api_auth_unknown_and_valid(configured, monkeypatch):
    monkeypatch.setenv(api.ENV_API_TOKEN, "api-token")
    report_id = configured.stem
    client = TestClient(api.app)
    body = {"to": ["to@example.com"], "message": "note"}
    assert client.post(f"/reports/{report_id}/email", json=body).status_code == 401
    headers = {api.TOKEN_HEADER: "api-token"}
    assert client.post("/reports/unknown/email", json=body, headers=headers).status_code == 404
    response = client.post(f"/reports/{report_id}/email", json=body, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"status": "sent", "recipients": 1}


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            smtplib.SMTPAuthenticationError(535, b"secret-value"),
            "Gmail rejected the login. Use a 16-character App Password",
        ),
        (
            smtplib.SMTPAuthenticationError(534, b"secret-value"),
            "SMTP authentication failed (code 534)",
        ),
        (
            smtplib.SMTPSenderRefused(550, b"secret-value", "sender@example.com"),
            "sender@example.com (code 550)",
        ),
        (
            smtplib.SMTPRecipientsRefused({"refused@example.com": (550, b"secret-value")}),
            "refused@example.com (code 550)",
        ),
        (
            smtplib.SMTPServerDisconnected("secret-value"),
            "secret-value",
        ),
        (ssl.SSLError("secret-value"), "secret-value"),
        (TimeoutError("secret-value"), "Network blocked outbound SMTP on port 465; try 587"),
        (ConnectionRefusedError("secret-value"), "Network blocked outbound SMTP on port 465; try 587"),
    ],
)
def test_smtp_failures_are_mapped_and_logs_never_contain_password(
    configured, monkeypatch, caplog, error, expected
):
    class FailedSMTP(FakeSMTP):
        def login(self, username, password):
            raise error

    monkeypatch.setattr(email_report.smtplib, "SMTP_SSL", FailedSMTP)
    monkeypatch.setattr(email_report.smtplib, "SMTP", FailedSMTP)
    monkeypatch.setenv("SMTP_PASSWORD", " secret-value ")
    with caplog.at_level(logging.WARNING):
        with pytest.raises(type(error)) as caught:
            email_report.send_report_email(
                ["to@example.com"], "Subject", "secret-value", configured
            )
    mapped = email_report.smtp_error_message(caught.value, port=465)
    assert expected in mapped
    assert "secret-value" not in caplog.text
    assert "to@example.com" not in caplog.text
    assert "WARNING" in caplog.text


def test_password_whitespace_is_stripped_and_from_defaults(configured, monkeypatch):
    monkeypatch.setenv("SMTP_PASSWORD", " app pass word ")
    monkeypatch.delenv("SMTP_FROM")
    email_report.send_report_email(["to@example.com"], "Subject", "Body", configured)
    assert FakeSMTP.instance.credentials[1] == "apppassword"
    assert FakeSMTP.instance.message["From"] == "user@example.com"


def test_587_uses_starttls_transport(configured, monkeypatch):
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setattr(email_report.smtplib, "SMTP", FakeSMTP)
    email_report.send_report_email(["to@example.com"], "Subject", "Body", configured)
    assert FakeSMTP.instance.args[:2] == ("smtp.example.com", 587)
    assert FakeSMTP.instance.tls is True


def test_test_email_uses_shared_sender(configured, monkeypatch):
    email_report.send_test_email("me@example.com")
    assert FakeSMTP.instance.message["To"] == "me@example.com"
    assert "test" in FakeSMTP.instance.message["Subject"].lower()


def test_email_status_and_test_email_routes_are_token_protected(configured, monkeypatch):
    monkeypatch.setenv(api.ENV_API_TOKEN, "api-token")
    client = TestClient(api.app)
    assert client.get("/email/status").status_code == 401
    headers = {api.TOKEN_HEADER: "api-token"}
    status = client.get("/email/status", headers=headers)
    assert status.status_code == 200
    assert status.json()["configured"] is True
    assert "secret-value" not in status.text
    result = client.post("/email/test", json={"to": "me@example.com"}, headers=headers)
    assert result.status_code == 200
    assert result.json() == {"status": "sent"}


def test_json_export_route_returns_the_current_run(monkeypatch):
    monkeypatch.setenv(api.ENV_API_TOKEN, "api-token")
    client = TestClient(api.app)
    payload = {
        "files": [{"filename": "sample.png", "verdict": "Inconclusive"}],
        "batch": None,
        "errors": [],
        "threshold": 0.5,
        "processing_seconds": 2.1,
    }
    assert client.post("/api/results/export", json=payload).status_code == 401
    response = client.post(
        "/api/results/export", json=payload, headers={api.TOKEN_HEADER: "api-token"}
    )
    assert response.status_code == 200
    assert response.json() == payload
