from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path


class DashboardParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: list[str] = []
        self.headers: list[str] = []
        self.in_header = False

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])
        if tag == "th":
            self.in_header = True

    def handle_endtag(self, tag):
        if tag == "th":
            self.in_header = False

    def handle_data(self, data):
        if self.in_header and data.strip():
            self.headers.append(data.strip())


def test_dashboard_has_one_email_and_report_flow_and_real_tables():
    source = (Path(__file__).resolve().parents[1] / "app" / "dashboard.html").read_text()
    parser = DashboardParser()
    parser.feed(source)

    assert parser.ids.count("report-actions-card") == 1
    assert parser.ids.count("side-email-send") == 1
    assert parser.ids.count("side-test-email") == 1
    assert parser.ids.count("export-pdf") == 1
    assert parser.ids.count("download-results") == 1
    assert "id=\"email-report\"" not in source
    assert "id=\"test-email\"" not in source
    assert "id=\"side-export\"" not in source
    assert "id=\"side-download\"" not in source
    assert "Metric" in parser.headers and "Value" in parser.headers
    assert all(header in parser.headers for header in ("Provider", "Key variable", "Status", "Model"))
    assert all(header in parser.headers for header in ("File", "Verdict", "Score", "Time"))
    assert "Coverage %" in parser.headers
