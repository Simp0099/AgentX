"""agent/env.py: secret-safe loading of model-provider credentials.

Two sources are consulted, in order, and neither ever overrides a variable that
is already present in the shell (exported values always win):

1. The file named by the ``AGENT_ENV_FILE`` environment variable (a path shared
   with another project so one set of free-tier keys can be reused).
2. The repo-local ``.env``.

Nothing in this module logs, prints, returns or stores a credential *value*.
``describe`` reports only whether a name is set and how long its value is, so a
caller can confirm configuration without putting a secret in a log or a commit.
"""

from __future__ import annotations

import os
from pathlib import Path

# Name of the shell variable that points at an external env file. Only the name
# is ever referenced; its value is a path, and values are never printed.
ENV_FILE_VAR = "AGENT_ENV_FILE"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_ENV = _REPO_ROOT / ".env"

_loaded_files: list[str] = []


def load_env(force: bool = False) -> list[str]:
    """Load credentials from ``AGENT_ENV_FILE`` then the repo-local ``.env``.

    Existing shell variables are never overridden, so an operator can always
    win over a file. Loading is idempotent: repeat calls are no-ops unless
    ``force`` is true.

    Args:
        force: re-read the files even if a previous call already did.

    Returns:
        The paths that were actually read, in load order. Paths are not
        secrets, so they are safe to report.
    """
    global _loaded_files
    if _loaded_files and not force:
        return list(_loaded_files)

    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dependency is pinned in requirements
        return []

    loaded: list[str] = []

    # 1. External file shared with another project (path comes from the shell).
    external = os.environ.get(ENV_FILE_VAR)
    if external:
        external_path = Path(external).expanduser()
        if external_path.is_file():
            # override=False keeps shell-exported values authoritative.
            load_dotenv(external_path, override=False)
            loaded.append(str(external_path))

    # 2. Repo-local .env (gitignored).
    if _LOCAL_ENV.is_file():
        load_dotenv(_LOCAL_ENV, override=False)
        loaded.append(str(_LOCAL_ENV))

    _loaded_files = loaded
    return list(loaded)


# Load dotenv values before provider modules read os.environ.
load_env()


def describe(name: str) -> str:
    """Return ``"set (length N)"`` or ``"missing"`` for one variable.

    The value itself is never returned, so this is safe for logs and reports.

    Args:
        name: the environment variable name, e.g. ``GEMINI_API_KEY``.
    """
    value = os.environ.get(name)
    if not value:
        return "missing"
    return f"set (length {len(value)})"


def get_secret(name: str) -> str | None:
    """Return a credential value for an API client, or ``None`` when unset.

    Callers must pass the returned value straight to the provider client and
    must never log, echo or persist it.
    """
    load_env()
    return os.environ.get(name) or None


def require(name: str) -> str:
    """Return a credential value, or raise when the variable is unset.

    Raises:
        RuntimeError: naming the missing variable, never a value.
    """
    value = get_secret(name)
    if value is None:
        raise RuntimeError(f"{name} is not set; see docs/model_selection.md")
    return value
