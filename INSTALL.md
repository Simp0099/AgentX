# Installation

## Requirements

- Python 3.10+
- ~4 GB free disk space (TensorFlow + dependencies; ResNet50 ImageNet
  weights add ~100 MB on first run)
- No GPU required. Inference and testing run on CPU. Training on CPU is
  slow but workable (see [TRAINING.md](TRAINING.md)); a CUDA-capable GPU or
  a hosted GPU runtime (e.g. Colab) is recommended for full training runs.

## 1. Clone the repository

```bash
git clone https://github.com/Simp0099/Ai-Deepfake-Tampering-Localizer.git
cd Ai-Deepfake-Tampering-Localizer
```

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs TensorFlow 2.21, NumPy, Pillow, PyYAML, scikit-learn,
matplotlib, Gradio (demo app), and pytest/ruff (dev tooling).

## 4. Verify the install

```bash
python -c "import tensorflow as tf; print('TF', tf.__version__)"
```

Then run the self-checks and test suite:

```bash
python -m src.data.ela          # ELA must flag a known splice
python -m src.data.dataset      # mask→bbox exact, 6-channel batch
python -m src.models.localizer  # output shapes + masked loss behaviour

pytest tests/ -v
ruff check src/ app/ tests/ scripts/
```

All tests build the model with `weights=None`, so no ImageNet download or
GPU is required to validate the install.

## GPU support (optional, for training)

TensorFlow 2.21 uses CUDA-enabled wheels automatically when a compatible
NVIDIA GPU and drivers are present — no separate install is needed on Linux.
Confirm detection with:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

An empty list means TensorFlow will fall back to CPU; training will still
run, just slower (see hardware guidance in [TRAINING.md](TRAINING.md)).

Apple Silicon (M-series) users should **not** install `tensorflow-metal`.
It is not required by this project and current versions have shown
compatibility issues with recent TensorFlow releases; the model trains
correctly on CPU. If it is already installed and causing import errors:

```bash
pip uninstall tensorflow-metal -y
```

## Common installation issues

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run modules with `python -m src.train`, not `python src/train.py`. The `-m` form puts the repo root on `sys.path`. |
| Import error mentioning `_pywrap_tensorflow_internal.so` on macOS | `pip uninstall tensorflow-metal -y`, then re-verify the import. |
| `pip install` fails resolving TensorFlow | Confirm Python is 3.10 or newer: `python --version`. |
| Tests fail with a GPU/CUDA error | The test suite is CPU-only by design (`weights=None`); check `CUDA_VISIBLE_DEVICES` isn't forcing a broken GPU context. |

## Next steps

- [DATASET.md](DATASET.md) — download and prepare training data
- [TRAINING.md](TRAINING.md) — train the model
- [INFERENCE.md](INFERENCE.md) — run inference on your own images
