# Dependencies - All Free and Open-Source

Every dependency listed below is free software under a permissive license.
No paid APIs, no SDKs for paid services, no managed databases.

| Package | License | Purpose | Cost |
|---------|---------|---------|------|
| `keras` | Apache-2.0 | Deep learning framework - core dependency | $0 |
| `tensorflow` | Apache-2.0 | TensorFlow Keras - model inference | $0 |
| `numpy` | BSD-3-Clause | Numerical computing | $0 |
| `PIL/Pillow` | BSD-3-Clause | Image processing - ELA, metadata | $0 |
| `gradio` | MIT | Web UI framework | $0 |
| `yaml` | MIT | PyYAML - config loading | $0 |
| `pytest` | MIT/PSF | Testing framework | $0 |
| `ruff` | MIT | Linter | $0 |
| `uuid` | PSF | Built-in - UUID generation | $0 |
| `hashlib` | PSF | Built-in - SHA-256 | $0 |
| `tempfile` | PSF | Built-in - temp directory management | $0 |
| `concurrent.futures` | PSF | Built-in - tool timeout enforcement | $0 |
| `os` | PSF | Built-in - path operations | $0 |
| `sys` | PSF | Built-in - system parameters | $0 |

## Built-in Modules (no install needed)
- `uuid`: Universally Unique Identifier generation, PSF license, $0
- `hashlib`: SHA-256 hashing, PSF license, $0
- `tempfile`: Temporary directory management, PSF license, $0
- `concurrent.futures`: Tool timeout enforcement, PSF license, $0
- `os`: Path operations, PSF license, $0
- `sys`: System parameters, PSF license, $0

## Verification
- All installed packages verified via `pip show <package>`
- License information from each package's metadata
- No packages with commercial restrictions found
- Ollama is optional: activated only if OLLAMA_HOST environment variable is set
- No test starts Ollama, calls a network endpoint, or sends email
