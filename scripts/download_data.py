"""Download CASIA v2 via the Kaggle API.

Usage:
    python scripts/download_data.py --dest data
    python scripts/download_data.py --dest data --verify-only

Requires:
    pip install kaggle
    ~/.kaggle/kaggle.json  (token from https://www.kaggle.com/settings)

Dataset mirrors move around; see data/README.md for alternatives and for the
folder layout the loader expects.
"""

import argparse
import subprocess
import sys
from pathlib import Path

KAGGLE_DATASET = "sophatvathana/casia-dataset"

# Mirrors are inconsistent about capitalization and naming; the loader probes the
# same candidate lists, so keep these in sync with dataset.py::build_index.
FOLDER_CANDIDATES = {
    "authentic": ["Au", "au", "authentic"],
    "tampered": ["Tp", "tp", "tampered"],
    "masks": ["CASIA 2 Groundtruth", "CASIA2 Groundtruth", "Gt", "groundtruth", "mask"],
}


def download(dest: Path, dataset: str = KAGGLE_DATASET) -> None:
    try:
        import kaggle  # noqa: F401  -- the import itself triggers the credential check
    except ImportError as exc:
        raise SystemExit("kaggle not installed. Run: pip install kaggle") from exc
    except OSError as exc:
        # Kaggle raises OSError when ~/.kaggle/kaggle.json is missing or unreadable.
        raise SystemExit(
            f"Kaggle credentials not found ({exc}). Create a token at "
            "https://www.kaggle.com/settings and save it to ~/.kaggle/kaggle.json"
        ) from exc

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset} -> {dest}  (several GB; this takes a while)")

    # subprocess rather than os.system so a non-zero exit is caught here instead
    # of falling through to a confusing "folders not found" message.
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"kaggle download failed (exit {result.returncode}). The mirror may have "
            "moved or been made private -- see data/README.md for alternatives."
        )

    verify(dest)


def verify(dest: Path) -> bool:
    """Report whether the expected CASIA folders are present. True if usable."""
    root = dest / "CASIA2"
    if not root.is_dir():
        found = sorted(p.name for p in dest.iterdir() if p.is_dir()) if dest.is_dir() else []
        print(f"WARNING: no CASIA2/ under {dest}. Directories present: {found}")
        print("Point --data-root at the real folder, or rename it to CASIA2/.")
        return False

    ok = True
    for kind, names in FOLDER_CANDIDATES.items():
        match = next((root / n for n in names if (root / n).is_dir()), None)
        if match:
            count = sum(1 for _ in match.iterdir())
            print(f"  {kind:10s} found {match.name!r}  ({count} files)")
        else:
            print(f"  {kind:10s} MISSING  (looked for {names})")
            ok = False

    if ok:
        print(f"\nOK. Train with:  python -m src.train --data-root {root}")
    else:
        print("\nGround-truth masks are required for the localization head. Without "
              "them only the classifier can be trained -- see data/README.md.")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Download CASIA v2 from Kaggle")
    parser.add_argument("--dest", default="data", type=Path)
    parser.add_argument("--dataset", default=KAGGLE_DATASET,
                        help="Kaggle dataset slug, if the default mirror is gone")
    parser.add_argument("--verify-only", action="store_true",
                        help="check an existing download without re-fetching")
    args = parser.parse_args()

    if args.verify_only:
        sys.exit(0 if verify(args.dest) else 1)
    download(args.dest, args.dataset)


if __name__ == "__main__":
    main()
