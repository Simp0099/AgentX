"""Error Level Analysis (ELA) preprocessing.

Why this works, in one paragraph: JPEG compression is lossy and *idempotent-ish*.
Save a JPEG again at the same quality and most pixels barely change, because they
are already sitting on the quantization grid the encoder would have snapped them
to. But a region that was pasted in from another image (or painted, or generated)
has been through a *different* compression history, so it is not yet on that grid.
Re-saving it moves those pixels noticeably more. ELA makes that visible: re-encode
the image at a known quality, subtract from the original, and amplify. Untouched
areas go near-black; regions with a different compression history light up.
"""

from io import BytesIO

import numpy as np
from PIL import Image, ImageChops


def compute_ela(image, quality=90, scale=None):
    """Return the ELA map of ``image`` as an RGB PIL Image.

    Args:
        image: PIL Image, numpy array (HWC uint8), or path to an image file.
        quality: JPEG quality used for the re-save. 90 is the usual default;
            it must be *above* the typical quality of the source images so the
            re-save is not itself destructive enough to swamp the signal.
        scale: brightness amplification factor. ``None`` auto-scales so the
            brightest difference pixel maps to 255, which is what makes faint
            splices visible. Pass a fixed number for deterministic output.

    Returns:
        PIL.Image in RGB mode, same size as the input.
    """
    if isinstance(image, (str, bytes)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    image = image.convert("RGB")

    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    diff = ImageChops.difference(image, recompressed)

    if scale is None:
        peak = max(band[1] for band in diff.getextrema())
        # A perfectly flat image (or one already at exactly this quality) has no
        # difference at all. Amplifying by 255 there would just amplify noise.
        scale = 255.0 / peak if peak else 1.0

    return Image.eval(diff, lambda px: min(255, int(px * scale)))


def ela_array(image, size=(224, 224), quality=90, scale=None):
    """ELA map as a float32 array in [0, 1], resized to ``size``.

    This is the form the model consumes. Kept separate from :func:`compute_ela`
    so the demo app can show the human-readable PIL version.
    """
    ela = compute_ela(image, quality=quality, scale=scale).resize(size)
    return np.asarray(ela, dtype=np.float32) / 255.0


if __name__ == "__main__":
    # Self-check: a hand-made splice must produce a brighter ELA response inside
    # the pasted region than outside it. This is the whole premise of the model,
    # so if it ever stops holding, everything downstream is meaningless.
    rng = np.random.default_rng(0)
    base = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))

    # Round-trip the base through JPEG so it has a settled compression history.
    buf = BytesIO()
    base.save(buf, "JPEG", quality=75)
    buf.seek(0)
    host = Image.open(buf).convert("RGB")

    # Paste in a patch that has NOT been through that same compression.
    spliced = host.copy()
    patch = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    spliced.paste(patch, (96, 96))

    ela = np.asarray(compute_ela(spliced, quality=90).convert("L"), dtype=np.float32)
    inside = ela[96:160, 96:160].mean()
    outside = np.concatenate([ela[:96].ravel(), ela[160:].ravel()]).mean()

    assert inside > outside, f"ELA failed to flag splice: {inside=:.1f} {outside=:.1f}"
    assert compute_ela(spliced).size == spliced.size
    assert ela_array(spliced).shape == (224, 224, 3)
    print(f"ok  spliced region {inside:.1f} vs background {outside:.1f}")
