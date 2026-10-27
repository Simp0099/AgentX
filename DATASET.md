# Dataset

The raw dataset is **not committed** to this repository — it is several
gigabytes, and redistribution terms are unclear for some mirrors. Download it
locally with the steps below.

## Supported datasets

| Dataset | Status | Task |
|---|---|---|
| CASIA v2 | ✅ Supported | Splicing detection + localization |
| FaceForensics++ | 🔜 Planned | Face-swap / reenactment detection |
| CelebDF | 🔜 Planned | Face-swap detection |
| DFDC | 🔜 Planned | Deepfake detection (video) |
| GenImage | 🔜 Planned | Generative-image detection |
| Synthbuster | 🔜 Planned | Diffusion-image detection |
| UniversalFakeDetect | 🔜 Planned | Cross-generator generalization |

Only CASIA v2 is implemented in `src/data/dataset.py` today. The remaining
datasets are future work (see [docs/limitations.md](docs/limitations.md)) and
are not yet wired into the loader or training pipeline.

## CASIA v2

CASIA v2.0 is a standard academic benchmark for image splicing detection:
approximately 7,400 authentic and 5,100 tampered images, with pixel-level
ground-truth masks for the tampered subset. The original host
(`forensics.idealtest.org`) has been offline for some time, so this project
downloads a Kaggle mirror.

```bash
pip install kaggle
# Place your API token at ~/.kaggle/kaggle.json
# (kaggle.com/settings -> Create New Token)
chmod 600 ~/.kaggle/kaggle.json

python scripts/download_data.py --dest data
```

### Expected folder structure

```text
data/
└── CASIA2/
    ├── Au/                        # authentic images       (label 0)
    ├── Tp/                        # tampered images         (label 1)
    └── CASIA 2 Groundtruth/       # binary masks for Tp/  → bounding boxes
```

The loader (`build_index` in `src/data/dataset.py`) probes several
capitalizations of these folder names, since mirrors are inconsistent. If
your download uses different names, either rename to match the table above
or add the variant to the candidate lists in `build_index`.

### Alternate mirrors

| Kaggle dataset | Notes |
|---|---|
| `sophatvathana/casia-dataset` | Default in `scripts/download_data.py`; includes ground-truth masks |
| `divg07/casia-20-image-tampering-detection-dataset` | Classification labels only — **no masks**, so localization cannot be trained |
| `alaakhaled/casia-v2` | Alternate mirror; verify the mask folder is present before use |

The hard requirement is **ground-truth masks**. Without them, there is no
localization target, and only the classification head can be trained.

### Known data quirks

- A small number of CASIA masks are blank or have no matching `Tp/` image.
  The loader reports these as `dropped(no/blank mask)` and skips them.
- Some mask filenames carry a `_gt` suffix; the loader normalizes for this.
- Class balance is roughly 2:1 authentic:tampered, which is why training
  applies `compute_class_weight('balanced', ...)`.

## Dataset preparation and splitting

`build_index()` walks the folder structure above and converts each
tampered-image mask to a tight bounding box via `mask_to_bbox()`.
`split_records()` then performs a stratified 60/20/20 train/validation/test
split (fractions configurable in `configs/default.yaml`), using a fixed seed
for reproducibility. The test split is written to
`checkpoints/test_split.json` after training so evaluation always scores the
same held-out images regardless of retraining.

Verify the index builds correctly before training:

```bash
python -c "from src.data.dataset import build_index; build_index('data/CASIA2')"
```

Expected output shape: `authentic=7491  tampered=5123  dropped(no/blank mask)=N`

## Fallback: synthetic splices (no download)

For smoke-testing the pipeline, CI, or debugging the box loss against
known-correct targets:

```bash
python -c "from src.data.dataset import synthetic_records; synthetic_records('data/synthetic', n=200)"
```

This pastes random patches into JPEG-settled backgrounds, so the
ground-truth box is exact by construction. It is **not** a substitute for
CASIA v2 for benchmarking — the splices are unrealistic and metrics on this
data are meaningless as a performance claim. Use it only to verify plumbing.
