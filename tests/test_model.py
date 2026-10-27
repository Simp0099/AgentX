"""Model forward-pass tests — no weights download, no GPU required.

These run in CI on every push. They verify shapes, loss behaviour, and
serialization — the things that silently break when Keras or TF is upgraded.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.models.localizer import build_model, masked_bbox_loss


@pytest.fixture(scope="module")
def model():
    # weights=None keeps CI offline and fast (~2 s vs ~30 s with imagenet).
    return build_model(weights=None)


def test_output_shapes(model):
    out = model(tf.zeros((3, 224, 224, 6)), training=False)
    assert out["label"].shape == (3, 1)
    assert out["bbox"].shape == (3, 5)


def test_label_output_range(model):
    out = model(tf.zeros((4, 224, 224, 6)), training=False)
    labels = out["label"].numpy()
    assert labels.min() >= 0.0 and labels.max() <= 1.0


def test_bbox_output_range(model):
    """Sigmoid on the bbox head must keep all predictions in [0, 1]."""
    out = model(tf.zeros((4, 224, 224, 6)), training=False)
    bbox = out["bbox"].numpy()
    assert bbox.min() >= 0.0 and bbox.max() <= 1.0


def test_bbox_never_degenerate(model):
    """Every predicted box must have strictly positive area.

    Regression test for a real defect: a head predicting corners directly can
    saturate to xmin >= xmax (observed: [1.0, 0.001, 0.99, 0.001]), which is a
    zero-area box. IoU is then exactly 0 regardless of localization quality, and
    a saturated sigmoid has no gradient left to recover with -- training appears
    to run while the localization head is dead. The centre/size parameterization
    makes positive area structural.
    """
    # Random inputs, not zeros: a constant input can mask a degeneracy that only
    # appears once activations vary.
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 255, (8, 224, 224, 6)).astype(np.float32)
    bbox = model(x, training=False)["bbox"].numpy()

    widths = bbox[:, 2] - bbox[:, 0]
    heights = bbox[:, 3] - bbox[:, 1]
    assert (widths > 0).all(), f"zero/negative width: {widths}"
    assert (heights > 0).all(), f"zero/negative height: {heights}"


def test_corner_center_roundtrip():
    """corners -> centre/size -> corners must be the identity."""
    from src.models.localizer import center_size_to_corners, corners_to_center_size

    corners = np.array([[0.2, 0.3, 0.6, 0.8],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.45, 0.45, 0.55, 0.55]], dtype=np.float32)
    result = center_size_to_corners(tf.constant(corners_to_center_size(corners))).numpy()
    np.testing.assert_allclose(result, corners, atol=1e-6)


def test_masked_bbox_loss_authentic_zero():
    """Authentic images (has_box=0) must contribute exactly zero box loss."""
    y_true = tf.constant([[0.1, 0.2, 0.8, 0.9, 0.0]])  # has_box = 0
    y_pred = tf.constant([[0.5, 0.5, 0.5, 0.5, 0.5]])
    assert float(masked_bbox_loss(y_true, y_pred)) == 0.0


def test_masked_bbox_loss_tampered_positive():
    """Tampered images (has_box=1) must produce a positive, finite loss."""
    y_true = tf.constant([[0.1, 0.2, 0.8, 0.9, 1.0]])
    y_pred = tf.constant([[0.5, 0.5, 0.5, 0.5, 0.5]])
    loss = float(masked_bbox_loss(y_true, y_pred))
    assert loss > 0.0 and loss == loss  # positive and not NaN


def test_masked_bbox_loss_perfect_prediction_zero():
    """Perfect box prediction on a tampered image must give near-zero loss."""
    y_true = tf.constant([[0.1, 0.2, 0.8, 0.9, 1.0]])
    y_pred = tf.constant([[0.1, 0.2, 0.8, 0.9, 1.0]])
    assert float(masked_bbox_loss(y_true, y_pred)) < 1e-6


def _huber(error, delta=0.1):
    """Reference Huber, mirroring masked_bbox_loss's delta."""
    error = np.abs(np.asarray(error, dtype=np.float64))
    return np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))


def test_mixed_batch_loss():
    """Mixed batch: loss must reflect the tampered row only, not the batch mean."""
    y_true = tf.constant([
        [0.1, 0.1, 0.5, 0.5, 1.0],  # tampered
        [0.0, 0.0, 0.0, 0.0, 0.0],  # authentic
    ])
    y_pred = tf.constant([
        [0.2, 0.2, 0.6, 0.6, 0.5],
        [0.9, 0.9, 0.9, 0.9, 0.5],  # large error, but authentic -> must be masked
    ])
    loss = float(masked_bbox_loss(y_true, y_pred))
    # Only the first row contributes; the second row's much larger error must not.
    expected = float(_huber([0.1, 0.1, 0.1, 0.1]).mean())
    assert abs(loss - expected) < 1e-5, f"{loss=:.6f} {expected=:.6f}"


def test_model_serialization(tmp_path, model):
    """Checkpoint must reload without custom_objects and produce identical output."""
    path = tmp_path / "test.keras"
    model.save(path)
    loaded = tf.keras.models.load_model(path)
    x = tf.zeros((2, 224, 224, 6))
    orig = model(x, training=False)
    reloaded = loaded(x, training=False)
    np.testing.assert_allclose(
        orig["label"].numpy(), reloaded["label"].numpy(), atol=1e-5
    )
    np.testing.assert_allclose(
        orig["bbox"].numpy(), reloaded["bbox"].numpy(), atol=1e-5
    )


def test_six_channel_input_required(model):
    """Passing a 3-channel tensor must raise, not silently produce wrong output."""
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        model(tf.zeros((1, 224, 224, 3)), training=False)
