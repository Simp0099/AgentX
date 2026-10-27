"""Dual-head deepfake localizer: real/tampered classification + tampered-region box.

Architecture lineage
--------------------
* ResNet50 backbone, frozen then fine-tuned      — Project 1 (Pet Face Classification)
* Dual-head pattern (shared trunk -> 2 heads)    — Project 2 (Object Detection)
* Class-weighted training, EarlyStopping         — Project 1
* ``[xmin, ymin, xmax, ymax]`` box convention    — Project 2

New here
--------
* 6-channel input: RGB stacked with its ELA map, with a dedicated ELA branch.
* Binary classification head instead of 10-class softmax.
* Box loss masking: authentic images have no tampered region, so their box loss
  is zeroed via a ``has_box`` flag packed into the target.
* Sigmoid on the box head, since normalized boxes live in [0, 1].

Deliberately NOT here: geometric augmentation. Project 1 could flip and rotate
freely because its target was a single class label. Here the target includes a
box, and Keras preprocessing layers cannot transform the box alongside the
image, so a flip would silently teach the model the wrong location. Resampling
also smooths the compression artifacts ELA exists to detect. See README
"Limitations" for the box-aware augmentation this leaves on the table.
"""

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Layer,
    LayerNormalization,
    MaxPooling2D,
)

# ── serializable helpers ──────────────────────────────────────────────────────
# These are real Layer subclasses rather than Lambda wrappers so that
# ``load_model`` works without ``safe_mode=False`` or a custom_objects dict.
# A checkpoint that cannot be reloaded is a checkpoint that does not exist.

@tf.keras.utils.register_keras_serializable(package="localizer")
class ChannelSlice(Layer):
    """Slice a contiguous run of channels off the last axis."""

    def __init__(self, start, stop, **kwargs):
        super().__init__(**kwargs)
        self.start, self.stop = start, stop

    def call(self, inputs):
        return inputs[..., self.start:self.stop]

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.stop - self.start)

    def get_config(self):
        return {**super().get_config(), "start": self.start, "stop": self.stop}


@tf.keras.utils.register_keras_serializable(package="localizer")
class ResNetPreprocess(Layer):
    """ResNet50's ``preprocess_input`` as a named, serializable layer."""

    def call(self, inputs):
        return tf.keras.applications.resnet50.preprocess_input(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


# ── box parameterization ──────────────────────────────────────────────────────
# The box head predicts (cx, cy, w, h) -- centre and size -- not the corners.
#
# Predicting corners directly through a sigmoid has a failure mode that is easy
# to hit and hard to diagnose: nothing couples xmin to xmax, so the head can
# saturate to xmin=1.0, xmax=0.99, giving a box of zero or negative area. IoU is
# then exactly 0 no matter how well the model actually localizes, and because a
# saturated sigmoid has a vanishing gradient, MSE cannot pull it back. Training
# looks like it is running while the localization head is silently dead.
#
# With (cx, cy, w, h) all in [0, 1] via sigmoid, width and height are positive by
# construction, so every prediction is a valid box with real area.

@tf.keras.utils.register_keras_serializable(package="localizer")
def center_size_to_corners(box, clip=False):
    """``(cx, cy, w, h)`` -> ``(xmin, ymin, xmax, ymax)``.

    ``clip`` is off inside the model on purpose. ``clip_by_value`` has zero
    gradient outside its range, so clipping in the training graph is a third way
    to strand the box head: once a coordinate drifts past the boundary it can
    never come back. Predictions are clipped at inference instead, where there is
    no gradient to preserve.
    """
    cx, cy, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    corners = tf.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)
    return tf.clip_by_value(corners, 0.0, 1.0) if clip else corners


def corners_to_center_size(box):
    """``(xmin, ymin, xmax, ymax)`` -> ``(cx, cy, w, h)``. Numpy, for targets."""
    import numpy as np
    box = np.asarray(box, dtype=np.float32)
    xmin, ymin, xmax, ymax = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    return np.stack([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], axis=-1)


@tf.keras.utils.register_keras_serializable(package="localizer")
class CornerDecode(Layer):
    """Decode the raw ``(cx, cy, w, h, has_box)`` head into corner form.

    ``softplus`` on width and height, not sigmoid: it is positive everywhere, so
    boxes keep real area, but it never saturates, so the gradient survives no
    matter how large the incoming activation gets. Centres stay linear.

    Kept inside the model so ``model(x)["bbox"]`` yields corners everywhere --
    inference, evaluation, and the demo all read the same convention as
    Project 2's IoU code, with no caller-side conversion to forget.
    """

    def call(self, inputs):
        # Sigmoid on the centre: it is a position in [0, 1], and bounding it
        # removes the "predict -45" failure mode entirely. Saturation is far less
        # dangerous here than on size, because a centre pinned at an edge is
        # still a usable box, whereas a size pinned at 0 is not a box at all.
        centre = tf.sigmoid(inputs[..., :2])
        # Softplus on size: positive everywhere so boxes keep real area, but it
        # never saturates, so the gradient survives however large the incoming
        # activation gets. +1e-3 floor keeps IoU from degenerating on zero area.
        size = tf.nn.softplus(inputs[..., 2:4]) + 1e-3
        corners = center_size_to_corners(tf.concat([centre, size], axis=-1))
        return tf.concat([corners, inputs[..., 4:5]], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape


# ── loss ──────────────────────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package="localizer")
def masked_bbox_loss(y_true, y_pred):
    """Huber loss on ``[xmin, ymin, xmax, ymax]``, for tampered images only.

    ``y_true`` is 5-dim: the corner box, then ``has_box`` (1 tampered, 0
    authentic). Authentic images contribute exactly zero -- "where is the splice"
    has no answer on an untampered image, and training against a zero box would
    teach the head to collapse every prediction toward the origin.

    Huber rather than Project 2's plain MSE: MSE's gradient shrinks with the
    error, so a head that starts far from the target creeps toward it. Huber
    keeps a constant gradient outside delta, which matters much more here than in
    Project 2, where digits sat on a blank canvas and the box was nearly free to
    guess.
    """
    has_box = y_true[..., 4:5]
    box_true = y_true[..., :4]
    box_pred = y_pred[..., :4]
    error = box_true - box_pred
    delta = 0.1  # ~10% of the normalized image dimension
    per_coord = tf.where(
        tf.abs(error) <= delta,
        0.5 * tf.square(error),
        delta * (tf.abs(error) - 0.5 * delta),
    )
    loss = tf.reduce_mean(per_coord, axis=-1, keepdims=True)
    # Normalize by the number of tampered samples, not the batch size, so the
    # loss scale does not drift with the authentic/tampered ratio of each batch.
    return tf.reduce_sum(loss * has_box) / (tf.reduce_sum(has_box) + 1e-7)


# ── model ─────────────────────────────────────────────────────────────────────

def build_model(input_shape=(224, 224, 6), fine_tune_from=None, weights="imagenet",
                bbox_loss_weight=1.0, learning_rate=None):
    """Build and compile the dual-head localizer.

    Args:
        input_shape: ``(H, W, 6)`` -- channels 0:3 RGB, 3:6 ELA.
        fine_tune_from: ``None`` freezes the whole backbone (phase 1). An int
            unfreezes from that backbone layer index onward; negative counts
            from the end, so ``-30`` mirrors Project 1's ``layers[:-30]``.
        weights: ``"imagenet"`` normally; ``None`` in tests/CI to avoid a
            90 MB download.
        bbox_loss_weight: lambda in ``total = BCE + lambda * masked_MSE``.
        learning_rate: defaults to 1e-3 frozen, 1e-5 fine-tuning (Project 1).
    """
    height, width, channels = input_shape
    if channels != 6:
        raise ValueError(f"expected 6 input channels (RGB+ELA), got {channels}")

    inputs = Input(shape=input_shape, name="input_rgb_ela")

    rgb = ChannelSlice(0, 3, name="rgb_split")(inputs)
    ela = ChannelSlice(3, 6, name="ela_split")(inputs)
    rgb = ResNetPreprocess(name="resnet_preprocess")(rgb)

    backbone = ResNet50(include_top=False, pooling=None, weights=weights,
                        input_shape=(height, width, 3))
    backbone.trainable = False

    # ── feature extraction ────────────────────────────────────────────────────
    # The two heads need fundamentally different things from the trunk, so the
    # trunk forks rather than feeding both from one pooled vector.
    #
    # "Is this tampered?" is a whole-image question, and global average pooling
    # is the right summary: it is position-invariant, which is a feature here.
    #
    # "Where is the tampering?" is the opposite. GlobalAveragePooling2D averages
    # over every spatial position, which *discards location by construction* --
    # the same feature map with the splice in the top-left and the bottom-right
    # pools to an identical vector. A box head fed that vector cannot do better
    # than predicting the dataset-mean box, which is exactly the collapse this
    # replaced (w, h -> 0, IoU pinned at 0 while classification hit 100%).
    #
    # Project 2 used Flatten() on its 75x75 canvas, which preserves position as
    # feature index. Flatten on ResNet50's 7x7x2048 output would be 100k features
    # into Dense -- so the spatial path is 1x1-convolved down to 128 channels
    # first, then flattened. Same principle, tractable parameter count.
    backbone_out = backbone(rgb, training=False)          # (7, 7, 2048)

    rgb_pooled = GlobalAveragePooling2D(name="gap_rgb")(backbone_out)

    rgb_spatial = Conv2D(128, 1, activation="relu", name="rgb_reduce")(backbone_out)
    rgb_spatial = LayerNormalization(name="rgb_spatial_norm")(rgb_spatial)
    rgb_spatial = Flatten(name="flatten_rgb")(rgb_spatial)  # 7*7*128 = 6272

    # Small dedicated ELA branch. Feeding ELA through ResNet50 would be wrong:
    # ImageNet features describe objects and texture, while ELA carries a
    # near-binary artifact-intensity signal that a couple of conv layers read
    # more directly than a 23M-parameter object recognizer.
    ela_feat = Conv2D(32, 3, activation="relu", padding="same", name="ela_conv1")(ela)
    ela_feat = MaxPooling2D(4, name="ela_pool1")(ela_feat)     # 56x56
    ela_feat = Conv2D(64, 3, activation="relu", padding="same", name="ela_conv2")(ela_feat)
    ela_feat = MaxPooling2D(4, name="ela_pool2")(ela_feat)     # 14x14
    ela_feat = Conv2D(64, 3, activation="relu", padding="same", name="ela_conv3")(ela_feat)
    ela_feat = MaxPooling2D(2, name="ela_pool3")(ela_feat)     # 7x7

    ela_pooled = GlobalAveragePooling2D(name="gap_ela")(ela_feat)

    # The ELA branch keeps its spatial map too -- it is the branch that actually
    # carries the localization signal, since bright ELA regions ARE the splice.
    ela_spatial = Conv2D(32, 1, activation="relu", name="ela_reduce")(ela_feat)
    # Normalize before Flatten on both spatial paths. Flattening 7x7 maps yields
    # thousands of unnormalized features; feeding those straight into Dense
    # produces huge pre-activations, which is what saturated the previous
    # sigmoid box head and stalled it at the [1,1,1,1] rail.
    #
    # LayerNormalization, not BatchNormalization. BN normalizes with batch
    # statistics while training but with moving averages at inference, and those
    # averages need many hundreds of steps to converge at momentum 0.99. A short
    # run leaves them near their (mean=0, var=1) init while the real activations
    # here have variance up to ~150 -- so the box head trains fine and then
    # predicts garbage the moment training=False. Observed: bbox_raw of -0.2 in
    # training mode and -50 at inference on the same batch, decoding to a
    # zero-area box and pinning IoU at ~0 while classification read 100%.
    # LN uses per-sample statistics, so training and inference are identical by
    # construction and no warm-up is needed.
    ela_spatial = LayerNormalization(name="ela_spatial_norm")(ela_spatial)
    ela_spatial = Flatten(name="flatten_ela")(ela_spatial)     # 7*7*32 = 1568

    # ── classification head: pooled (position-invariant) features ─────────────
    cls_feat = Concatenate(name="fused_cls")([rgb_pooled, ela_pooled])
    cls_feat = Dense(256, activation="relu", name="cls_dense")(cls_feat)
    cls_feat = Dropout(0.3, name="cls_dropout")(cls_feat)
    cls_out = Dense(1, activation="sigmoid", name="label")(cls_feat)

    # ── localization head: spatial (position-preserving) features ─────────────
    box_feat = Concatenate(name="fused_box")([rgb_spatial, ela_spatial])
    box_feat = Dense(256, activation="relu", name="box_dense")(box_feat)
    box_feat = Dropout(0.3, name="box_dropout")(box_feat)

    # Raw output is (cx, cy, w, h, has_box).
    #
    # Scale anchoring matters more than activation choice here. A plain linear
    # head starts wherever the weights land -- observed predictions of -45 and
    # -116 on a [0,1] target, from which Huber's constant gradient needs a very
    # long time to walk back. So:
    #   - zero-initialized kernel: every image starts at the same box, set purely
    #     by the bias, instead of at a random large offset.
    #   - bias (0, 0, 0, 0) -> after decoding, a centred box of ~0.69 x 0.69.
    #     A middling centred box is the best possible ignorant guess, and it is
    #     within one gradient step of anything the data asks for.
    # The trunk still learns; only the starting point is pinned.
    bbox_raw = Dense(
        5, name="bbox_raw",
        kernel_initializer="zeros",
        bias_initializer="zeros",
    )(box_feat)
    # Decode to corners inside the model so every consumer -- IoU, the demo,
    # inference -- reads Project 2's [xmin, ymin, xmax, ymax] convention.
    bbox_out = CornerDecode(name="bbox")(bbox_raw)

    # Dict outputs, not a list: Keras 3 resolves dict-keyed losses and metrics by
    # output name, and a list output makes it fall back to integer indexing
    # (raises ``KeyError: 0``).
    model = Model(inputs, {"label": cls_out, "bbox": bbox_out}, name="deepfake_localizer")

    if fine_tune_from is not None:
        unfreeze_backbone(model, fine_tune_from)

    compile_model(model,
                  learning_rate=learning_rate or (1e-5 if fine_tune_from is not None else 1e-3),
                  bbox_loss_weight=bbox_loss_weight)
    return model


def compile_model(model, learning_rate=1e-3, bbox_loss_weight=1.0):
    """Compile with the dual-loss setup. Called again after unfreezing."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"label": "binary_crossentropy", "bbox": masked_bbox_loss},
        loss_weights={"label": 1.0, "bbox": bbox_loss_weight},
        metrics={"label": ["accuracy", tf.keras.metrics.AUC(name="auc")]},
    )
    return model


def get_backbone(model):
    """Find the ResNet50 sub-model by structure, not by name.

    Keras auto-uniquifies names, so building a second model in the same process
    yields ``resnet50_1``; a hardcoded ``get_layer("resnet50")`` would break.
    """
    for layer in model.layers:
        if isinstance(layer, Model) and layer.name.startswith("resnet50"):
            return layer
    raise ValueError("no ResNet50 sub-model found")


def unfreeze_backbone(model, fine_tune_from=-30):
    """Unfreeze the top of ResNet50 for phase-2 fine-tuning (Project 1 pattern)."""
    backbone = get_backbone(model)
    backbone.trainable = True
    cut = len(backbone.layers) + fine_tune_from if fine_tune_from < 0 else fine_tune_from
    for layer in backbone.layers[:cut]:
        layer.trainable = False
    # BatchNorm stays frozen even in the unfrozen region. Letting it update its
    # running statistics on small fine-tuning batches wrecks them, which shows
    # up as training accuracy climbing while validation accuracy collapses.
    for layer in backbone.layers[cut:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return model


if __name__ == "__main__":
    # weights=None keeps this self-check offline and fast.
    model = build_model(weights=None)
    out = model(tf.zeros((2, 224, 224, 6)), training=False)
    assert out["label"].shape == (2, 1), out["label"].shape
    assert out["bbox"].shape == (2, 5), out["bbox"].shape

    # The defect this parameterization exists to prevent: every predicted box
    # must have strictly positive area. A corner-predicting head can saturate to
    # xmin >= xmax, which pins IoU at 0 forever and cannot recover, because a
    # saturated sigmoid has no gradient left to recover with.
    boxes = out["bbox"].numpy()
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    assert (widths > 0).all(), f"degenerate width: {widths}"
    assert (heights > 0).all(), f"degenerate height: {heights}"

    # Authentic-only batch must produce exactly zero box loss, otherwise the
    # mask is not doing its job and real images are being taught fake boxes.
    authentic = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0]])
    pred = tf.constant([[0.3, 0.3, 0.9, 0.9, 0.5]])
    assert float(masked_bbox_loss(authentic, pred)) == 0.0

    # Tampered target must produce a positive, finite loss.
    tampered = tf.constant([[0.1, 0.1, 0.5, 0.5, 1.0]])
    loss = float(masked_bbox_loss(tampered, pred))
    assert loss > 0 and loss == loss, loss

    # Round-trip: corners -> centre/size -> corners must be identity.
    corners = tf.constant([[0.2, 0.3, 0.6, 0.8]])
    round_tripped = center_size_to_corners(
        tf.constant(corners_to_center_size(corners.numpy()))
    ).numpy()
    assert abs(round_tripped - corners.numpy()).max() < 1e-6, round_tripped

    # Train and inference mode must agree on the same batch. This is the check
    # that a normalization layer with moving statistics fails: BatchNorm on the
    # spatial paths trained fine and then predicted zero-area boxes at
    # training=False, because its moving averages had not converged. Any layer
    # whose behaviour depends on accumulated batch statistics breaks this.
    batch = tf.random.stateless_uniform((4, 224, 224, 6), seed=(1, 2))
    train_box = model(batch, training=True)["bbox"].numpy()[:, :4]
    infer_box = model(batch, training=False)["bbox"].numpy()[:, :4]
    drift = abs(train_box - infer_box).max()
    assert drift < 1e-4, f"train/inference box mismatch: {drift}"

    print(f"ok  label={tuple(out['label'].shape)}  bbox={tuple(out['bbox'].shape)}  "
          f"area>0 for all boxes  train/infer drift={drift:.2e}  "
          f"masked_loss={loss:.4f}  params={model.count_params():,}")
