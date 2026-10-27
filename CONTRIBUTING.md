# Contributing

Contributions are welcome — bug fixes, documentation improvements, new
dataset loaders, and architecture experiments alike.

## Getting started

1. Fork the repository and clone your fork.
2. Follow [INSTALL.md](INSTALL.md) to set up the environment.
3. Run the test suite to confirm a clean baseline:

   ```bash
   pytest tests/ -v
   ruff check src/ app/ tests/ scripts/
   ```

## Branch naming

Use a short, descriptive prefix:

- `feat/<short-description>` — new functionality
- `fix/<short-description>` — bug fixes
- `docs/<short-description>` — documentation only
- `refactor/<short-description>` — internal changes with no behavior change

## Making changes

- Keep changes focused; unrelated fixes belong in separate PRs.
- Add or update tests for any behavior change — `tests/` covers ELA
  correctness, model I/O, and pipeline integration, and new functionality
  should extend that coverage rather than reduce it.
- Do not hardcode hyperparameters, paths, or magic numbers in `src/` — add
  them to `configs/default.yaml` instead.
- Update the relevant docs (`README.md`, `TRAINING.md`, `DATASET.md`, etc.)
  when behavior or CLI flags change.

## Coding style

- Formatting and linting are enforced by [ruff](https://docs.astral.sh/ruff/)
  (config in `pyproject.toml`). Run `ruff check src/ app/ tests/ scripts/`
  before committing.
- Target Python 3.10+.
- Prefer explicit, self-documenting code over comments explaining what code
  does; use comments for *why*, especially around non-obvious numerical
  choices (loss weights, thresholds, layer choices).

## Tests

```bash
pytest tests/ -v
```

Tests build the model with `weights=None`, so they run on CPU without
downloading ImageNet weights — keep new tests fast and GPU-free so CI stays
free-tier friendly. If you add a new module with a core invariant worth
protecting, consider adding a `python -m src.<module>` self-check in
addition to a pytest test, following the existing pattern in
`src/data/ela.py` and `src/models/localizer.py`.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) style:

```text
feat: add cross-dataset evaluation support
fix: correct IoU calculation for zero-area boxes
docs: clarify CASIA v2 folder layout
refactor: extract box decoding into a shared utility
```

## Pull requests

- Describe what changed and why, not just what.
- Link any related issue.
- Confirm `pytest` and `ruff` both pass — CI will re-check this on push, but
  a clean local run saves a round-trip.
- Screenshots or `reports/qualitative.png` output are appreciated for
  changes that affect model behavior.

## Reporting issues

Please include:

- A minimal reproduction (command run, config used, error output)
- TensorFlow version and OS (`python -c "import tensorflow as tf; print(tf.__version__)"`)
- Whether the issue is CPU- or GPU-specific
