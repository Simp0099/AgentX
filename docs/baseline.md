# Environment baseline (Step 1 / Phase 0)

Run date: 2026-10-27. A venv exists at `.venv311`; it was created with Python 3.15.0.

## Venv installation and collection

Command: `source .venv311/bin/activate && pip install -r requirements.txt`

```text
ERROR: Could not find a version that satisfies the requirement tensorflow==2.21.0 (from versions: none)
ERROR: No matching distribution found for tensorflow==2.21.0
```

The package index could not resolve `pypi.org` (DNS/network unavailable). Consequently, the venv has no pytest:

```text
source .venv311/bin/activate && python -m pytest --collect-only -q
.venv311/bin/python: No module named pytest
```

**Phase 0 status: BLOCKED.** The requested isolated environment is incomplete. Do not treat the global interpreter results below as a successful venv installation.

## Available global interpreter evidence

Python 3.11.17 was already installed and has project test dependencies. Versions:

```text
tensorflow==2.21.0
keras==3.15.1
numpy==2.4.6
Pillow==13.0.0
PyYAML==6.0.3
scikit-learn==1.9.1
matplotlib==3.11.2
reportlab==5.1.0
pytest==9.1.1
ruff 0.17.2
```

Command: `/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest --collect-only -q`

```text
97 tests collected in 19.30s
```

Earlier full global run: `97 passed, 2 warnings in 38.91s`. Ruff command `/opt/homebrew/bin/ruff check src/ app/ tests/ scripts/ agent/` returned `All checks passed!`. These global results do not remove the venv installation blocker.

## Git availability

```text
GIT_ABSENT
fatal: not a git repository
```

No history or commit-order claims are made. A prior `git init` attempt was denied by the workspace sandbox (`Operation not permitted` while creating `.git`).
