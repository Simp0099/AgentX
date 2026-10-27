from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from src.data.ela import compute_ela, ela_array


@pytest.fixture
def spliced():
    """A JPEG-settled host image with an uncompressed patch pasted in.

    Returns (image, patch_box) where patch_box is (left, top, right, bottom).
    """
    rng = np.random.default_rng(0)
    base = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    buf = BytesIO()
    base.save(buf, "JPEG", quality=75)
    buf.seek(0)
    host = Image.open(buf).convert("RGB")

    patch = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    host.paste(patch, (96, 96))
    return host, (96, 96, 160, 160)


def test_preserves_size_and_mode(spliced):
    image, _ = spliced
    ela = compute_ela(image)
    assert ela.size == image.size
    assert ela.mode == "RGB"


def test_flags_spliced_region(spliced):
    """The core premise: a pasted region must light up brighter than its host."""
    image, (left, top, right, bottom) = spliced
    ela = np.asarray(compute_ela(image, quality=90).convert("L"), dtype=np.float32)
    inside = ela[top:bottom, left:right].mean()
    outside = np.concatenate([ela[:top].ravel(), ela[bottom:].ravel()]).mean()
    assert inside > outside * 2, f"weak signal: {inside=:.1f} {outside=:.1f}"


def test_array_shape_and_range(spliced):
    image, _ = spliced
    arr = ela_array(image, size=(224, 224))
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_accepts_numpy_and_path(spliced, tmp_path):
    image, _ = spliced
    assert compute_ela(np.asarray(image)).size == image.size

    path = tmp_path / "sample.jpg"
    image.save(path, "JPEG", quality=75)
    assert compute_ela(str(path)).size == image.size


def test_flat_image_does_not_blow_up():
    """A uniform image has zero difference; auto-scale must not divide by zero."""
    flat = Image.new("RGB", (64, 64), (128, 128, 128))
    arr = np.asarray(compute_ela(flat))
    assert arr.max() == 0


def test_fixed_scale_is_deterministic(spliced):
    image, _ = spliced
    a = np.asarray(compute_ela(image, scale=20))
    b = np.asarray(compute_ela(image, scale=20))
    assert np.array_equal(a, b)
