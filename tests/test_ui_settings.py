from __future__ import annotations

import json

from app.ui_settings import load_ui_settings, recent_runs_html, save_analysis_settings


def test_analysis_settings_save_load_round_trip(tmp_path):
    settings_file = tmp_path / "artifacts" / "settings.json"
    saved, message = save_analysis_settings(
        0.73,
        64,
        250,
        240,
        4,
        False,
        True,
        path=settings_file,
        apply=False,
    )

    loaded = load_ui_settings(settings_file)
    assert saved["threshold"] == loaded["threshold"] == 0.73
    assert loaded["max_file_size_mb"] == 64
    assert loaded["max_zip_members"] == 250
    assert loaded["batch_time_limit_seconds"] == 240
    assert loaded["max_workers"] == 4
    assert loaded["run_ela"] is False
    assert loaded["run_metadata_inspection"] is True
    assert "artifacts/settings.json" in message
    assert "password" not in json.dumps(loaded).lower()


def test_recent_run_markup_stacks_fields_and_escapes_names(tmp_path):
    history = tmp_path / "history"
    history.mkdir()
    long_name = "image-" + ("very-long-" * 20) + "<unsafe>.png"
    (history / "run.json").write_text(
        json.dumps(
            {
                "created_at": "2026-10-28T12:00:00Z",
                "files": [{"filename": long_name, "verdict": "Flagged for review", "score": 0.81}],
            }
        ),
        encoding="utf-8",
    )

    markup = recent_runs_html(history, limit=10)

    assert "recent-run-name" in markup
    assert "recent-run-meta" in markup
    assert "recent-run-time" in markup
    assert "&lt;unsafe&gt;" in markup
    assert 'title="' in markup
    assert "Score 0.810" in markup
