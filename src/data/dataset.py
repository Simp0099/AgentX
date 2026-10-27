"""CASIA v2 loading, mask->bbox conversion, and tf.data pipeline.

Coordinate convention is inherited from Project 2 (Object_Detection_TensorFlow):
boxes are ``[xmin, ymin, xmax, ymax]``, normalized to [0, 1] against image size.
Keeping that exact order means Project 2's ``intersection_over_union`` and its
drawing helpers work here unchanged.

Authentic images have no tampered region. They carry a zero box and a
``has_box=0`` flag; the box loss is masked out for them (see models/losses.py),
because regressing "where is the splice" on an image with no splice is not a
meaningful target.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from .ela import ela_array

AUTHENTIC, TAMPERED = 0, 1
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}


def mask_to_bbox(mask, threshold=127):
    """Tightest box around the nonzero region of a ground-truth mask.

    Args:
        mask: PIL Image, path, or 2D array. Nonzero/bright = tampered pixel.
        threshold: pixel value above which a pixel counts as tampered.

    Returns:
        ``(xmin, ymin, xmax, ymax)`` normalized to [0, 1], or ``None`` if the
        mask is empty (a handful of CASIA masks are blank/corrupt).
    """
    if isinstance(mask, (str, Path)):
        mask = Image.open(mask)
    if isinstance(mask, Image.Image):
        mask = np.asarray(mask.convert("L"))
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.max(axis=2)

    rows = np.any(mask > threshold, axis=1)
    cols = np.any(mask > threshold, axis=0)
    if not rows.any() or not cols.any():
        return None

    ymin, ymax = np.flatnonzero(rows)[[0, -1]]
    xmin, xmax = np.flatnonzero(cols)[[0, -1]]
    h, w = mask.shape[:2]
    # +1 on the max edge so a single-pixel region still has nonzero area.
    return (xmin / w, ymin / h, (xmax + 1) / w, (ymax + 1) / h)


def _index_masks(mask_dir):
    """Map a normalized stem -> mask path.

    CASIA mask filenames track their Tp image but inconsistently: some carry a
    ``_gt`` suffix, some differ in case or extension. Normalizing the stem and
    indexing once is far cheaper than globbing per image.
    """
    index = {}
    for path in Path(mask_dir).iterdir():
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = path.stem.lower()
        if stem.endswith("_gt"):
            stem = stem[:-3]
        index[stem] = path
    return index


def build_index(root, verbose=True):
    """Scan a CASIA2 root into a list of ``(image_path, label, bbox)`` records.

    Expects ``root/Au`` (authentic), ``root/Tp`` (tampered) and a groundtruth
    folder. Folder names vary between Kaggle mirrors, so candidates are probed.
    Tampered images whose mask is missing or blank are dropped -- they have no
    usable localization target.
    """
    root = Path(root)
    au_dir = next((root / c for c in ("Au", "au", "authentic") if (root / c).is_dir()), None)
    tp_dir = next((root / c for c in ("Tp", "tp", "tampered") if (root / c).is_dir()), None)
    gt_dir = next(
        (
            root / c
            for c in ("CASIA 2 Groundtruth", "CASIA2 Groundtruth", "Gt", "groundtruth", "mask")
            if (root / c).is_dir()
        ),
        None,
    )
    if au_dir is None or tp_dir is None:
        raise FileNotFoundError(
            f"Could not find Au/ and Tp/ under {root}. Contents: "
            f"{[p.name for p in root.iterdir()][:20] if root.is_dir() else 'missing'}"
        )
    if gt_dir is None:
        raise FileNotFoundError(f"No groundtruth mask folder under {root}")

    records = []
    for path in sorted(au_dir.iterdir()):
        if path.suffix.lower() in IMAGE_EXTS:
            records.append((str(path), AUTHENTIC, (0.0, 0.0, 0.0, 0.0)))
    n_authentic = len(records)

    masks = _index_masks(gt_dir)
    dropped = 0
    for path in sorted(tp_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        mask_path = masks.get(path.stem.lower())
        bbox = mask_to_bbox(mask_path) if mask_path else None
        if bbox is None:
            dropped += 1
            continue
        records.append((str(path), TAMPERED, bbox))

    if verbose:
        n_tampered = len(records) - n_authentic
        print(f"authentic={n_authentic}  tampered={n_tampered}  dropped(no/blank mask)={dropped}")
    return records


def load_example(path, size=(224, 224), ela_quality=90):
    """Load one image as the model's 6-channel input: RGB stacked with its ELA map.

    RGB is left in [0, 255] because the ResNet50 preprocessing layer in the model
    expects that range; the ELA channels are scaled to match.
    """
    image = Image.open(path).convert("RGB")
    ela = ela_array(image, size=size, quality=ela_quality) * 255.0
    rgb = np.asarray(image.resize(size), dtype=np.float32)
    return np.concatenate([rgb, ela], axis=-1)


def split_records(records, val_frac=0.2, test_frac=0.2, seed=42):
    """Stratified train/val/test split.

    Stratifying on label matters: CASIA v2 is roughly 2:1 authentic:tampered, and
    an unstratified split can leave a val set with a visibly different balance,
    which makes the val metrics lie.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([r[1] for r in records])
    train, val, test = [], [], []
    for label in (AUTHENTIC, TAMPERED):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n_test = int(len(idx) * test_frac)
        n_val = int(len(idx) * val_frac)
        test += [records[i] for i in idx[:n_test]]
        val += [records[i] for i in idx[n_test : n_test + n_val]]
        train += [records[i] for i in idx[n_test + n_val :]]
    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


def make_dataset(records, size=(224, 224), batch_size=32, shuffle=False, ela_quality=90):
    """Records -> batched tf.data.Dataset yielding ``(x, {"label":, "bbox":})``.

    ELA runs a JPEG re-encode per image, which is pure Python, so it goes through
    ``numpy_function`` inside ``map`` with parallel calls rather than being
    vectorized. This is the pipeline's bottleneck; ``cache_arrays`` in train.py
    is the escape hatch for small datasets.
    """
    paths = np.array([r[0] for r in records])
    labels = np.array([r[1] for r in records], dtype=np.float32)
    boxes = np.array([r[2] for r in records], dtype=np.float32)
    # has_box mirrors the label today, but it is kept as its own tensor so an
    # authentic-image-with-annotation case later would not need a loss rewrite.
    has_box = labels.copy()

    def _load(path, label, box, flag):
        x = tf.numpy_function(
            lambda p: load_example(p.decode(), size=size, ela_quality=ela_quality),
            [path],
            tf.float32,
        )
        x.set_shape((*size, 6))
        return x, {
            "label": tf.reshape(label, (1,)),
            "bbox": tf.concat([box, tf.reshape(flag, (1,))], axis=0),
        }

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, boxes, has_box))
    if shuffle:
        ds = ds.shuffle(min(len(records), 2048), reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def synthetic_records(out_dir, n=48, size=256, seed=0):
    """Generate a tiny spliced-image dataset with exact ground-truth boxes.

    Not a research substitute for CASIA -- it exists so the model, training loop,
    and CI can be exercised end to end without a multi-GB download, and so the
    box targets are known to be correct when debugging the loss.
    """
    out_dir = Path(out_dir)
    (out_dir / "Au").mkdir(parents=True, exist_ok=True)
    (out_dir / "Tp").mkdir(parents=True, exist_ok=True)
    (out_dir / "Gt").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    for i in range(n):
        # Low-frequency background so JPEG has something compressible to settle on.
        small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
        host = Image.fromarray(small).resize((size, size), Image.BICUBIC)
        au_path = out_dir / "Au" / f"au_{i:04d}.jpg"
        host.save(au_path, "JPEG", quality=75)
        host = Image.open(au_path).convert("RGB")

        bw = int(rng.integers(size // 8, size // 3))
        bh = int(rng.integers(size // 8, size // 3))
        x = int(rng.integers(0, size - bw))
        y = int(rng.integers(0, size - bh))
        patch = Image.fromarray(rng.integers(0, 255, (bh, bw, 3), dtype=np.uint8))

        spliced = host.copy()
        spliced.paste(patch, (x, y))
        spliced.save(out_dir / "Tp" / f"tp_{i:04d}.jpg", "JPEG", quality=95)

        mask = np.zeros((size, size), dtype=np.uint8)
        mask[y : y + bh, x : x + bw] = 255
        Image.fromarray(mask).save(out_dir / "Gt" / f"tp_{i:04d}_gt.png")

    return build_index(out_dir, verbose=False)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        records = synthetic_records(tmp, n=8)
        assert len(records) == 16, len(records)

        # Round-trip check: a known mask must come back as the box we drew.
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[25:75, 50:150] = 255
        box = mask_to_bbox(mask)
        assert np.allclose(box, (0.25, 0.25, 0.75, 0.75)), box
        assert mask_to_bbox(np.zeros((10, 10), dtype=np.uint8)) is None

        train, val, test = split_records(records, seed=1)
        assert len(train) + len(val) + len(test) == len(records)

        x, y = next(iter(make_dataset(records, batch_size=4)))
        assert x.shape == (4, 224, 224, 6), x.shape
        assert y["label"].shape == (4, 1) and y["bbox"].shape == (4, 5)
        assert 0.0 <= float(tf.reduce_min(x)) and float(tf.reduce_max(x)) <= 255.0
        print("ok  mask->bbox exact, 6-channel batch", tuple(x.shape))
