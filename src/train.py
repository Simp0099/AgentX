"""Two-phase training: frozen backbone, then fine-tune the top of ResNet50.

Both phases and their callbacks come straight from Project 1 (Pet Face
Classification): train the new heads at 1e-3 with the backbone frozen, then
unfreeze the last ~30 layers and continue at 1e-5. EarlyStopping restores the
best weights; ModelCheckpoint keeps the best epoch on disk.

Class weighting is also from Project 1, but delivered differently: Keras rejects
``class_weight`` on multi-output models, so the balanced weights are applied as
per-sample weights on the label head instead.

Usage:
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/default.yaml --data-root data/synthetic
    python -m src.train --config configs/default.yaml --epochs-frozen 1 --epochs-finetune 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.utils.class_weight import compute_class_weight

from src.data.dataset import build_index, make_dataset, split_records
from src.models.localizer import build_model, compile_model, unfreeze_backbone


def load_config(path, overrides=None):
    """Load YAML config, applying ``"section.field"`` overrides from argparse."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        section, _, field = key.partition(".")
        cfg[section][field] = value
    return cfg


def class_weight_lookup(records):
    """Balanced class weights for the label head (Project 1's approach).

    CASIA v2 runs roughly 2:1 authentic:tampered. Without reweighting, predicting
    "authentic" everywhere already scores ~60% accuracy, which is a local optimum
    the model is happy to sit in.
    """
    labels = np.array([r[1] for r in records])
    classes = np.unique(labels)
    if len(classes) < 2:
        return {int(classes[0]): 1.0}
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights, strict=True)}


def apply_sample_weights(ds, weights):
    """Attach per-sample weights so the label head sees balanced classes.

    The bbox head gets weight 1.0 everywhere; its own ``has_box`` masking already
    excludes authentic images, and stacking class weights on top would double-count.
    """
    w0 = weights.get(0, 1.0)
    w1 = weights.get(1, 1.0)

    def _add(x, y):
        label = y["label"]
        return x, y, {
            "label": tf.where(label > 0.5, w1, w0),
            "bbox": tf.ones_like(label),
        }

    return ds.map(_add, num_parallel_calls=tf.data.AUTOTUNE)


def train(cfg, data_root=None, weights="imagenet"):
    dcfg, tcfg, ocfg = cfg["data"], cfg["training"], cfg["output"]
    root = data_root or dcfg["root"]
    size = tuple(dcfg["image_size"])

    records = build_index(root)
    if not records:
        raise RuntimeError(f"No usable images found under {root}. See data/README.md")

    train_recs, val_recs, test_recs = split_records(
        records, val_frac=dcfg["val_frac"], test_frac=dcfg["test_frac"], seed=dcfg["seed"]
    )
    print(f"split  train={len(train_recs)}  val={len(val_recs)}  test={len(test_recs)}")

    weight_lookup = class_weight_lookup(train_recs)
    print(f"class weights (balanced): {weight_lookup}")

    ds_kwargs = dict(size=size, batch_size=tcfg["batch_size"], ela_quality=dcfg["ela_quality"])
    train_ds = apply_sample_weights(
        make_dataset(train_recs, shuffle=True, **ds_kwargs), weight_lookup
    )
    # Validation stays unweighted: val metrics should reflect the real class
    # balance the model will face, not a reweighted view of it.
    val_ds = make_dataset(val_recs, **ds_kwargs)

    ckpt_dir = Path(ocfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / ocfg["checkpoint_name"]

    def callbacks():
        # Rebuilt per phase so EarlyStopping's patience counter restarts; a shared
        # instance can stop phase 2 immediately because phase 1 already plateaued.
        #
        # Monitor val_loss, not val_label_accuracy. Classification here is the easy
        # half of the task and saturates early -- it hit 1.000 by epoch 2 on a run
        # where val_bbox_loss went on falling from 8.7 to 2.7. Monitoring accuracy
        # made both callbacks blind to that: save_best_only never fired again, and
        # restore_best_weights rolled the model back to epoch 2, throwing away
        # every bit of localization the remaining epochs had learned. val_loss is
        # BCE + lambda*box_loss, so it still moves when only the box head is
        # improving, which is exactly when this model is getting better.
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min",
                patience=tcfg["patience"], restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(ckpt_path), monitor="val_loss", mode="min",
                save_best_only=True, verbose=1,
            ),
        ]

    model = build_model(
        input_shape=(*size, 6),
        weights=weights,
        bbox_loss_weight=tcfg["bbox_loss_weight"],
        learning_rate=tcfg["lr_frozen"],
    )
    print(f"trainable params (frozen phase): {model.count_params():,} total")

    histories = {}

    if tcfg["epochs_frozen"] > 0:
        print("\n===== phase 1: frozen backbone =====")
        h = model.fit(train_ds, validation_data=val_ds,
                      epochs=tcfg["epochs_frozen"], callbacks=callbacks())
        histories["frozen"] = {k: [float(v) for v in vs] for k, vs in h.history.items()}

    if tcfg["epochs_finetune"] > 0:
        print("\n===== phase 2: fine-tuning top of ResNet50 =====")
        unfreeze_backbone(model, tcfg["fine_tune_from"])
        compile_model(model, learning_rate=tcfg["lr_finetune"],
                      bbox_loss_weight=tcfg["bbox_loss_weight"])
        h = model.fit(train_ds, validation_data=val_ds,
                      epochs=tcfg["epochs_finetune"], callbacks=callbacks())
        histories["finetune"] = {k: [float(v) for v in vs] for k, vs in h.history.items()}

    model.save(ckpt_path)
    print(f"\nsaved {ckpt_path}")

    # Persist the test split so evaluate.py scores the exact same held-out images.
    # Without this, a re-run with a different seed would quietly leak train data
    # into the reported test metrics.
    (ckpt_dir / "test_split.json").write_text(json.dumps(
        [{"path": path, "label": int(label), "bbox": [float(v) for v in box]}
         for path, label, box in test_recs], indent=2,
    ))
    (ckpt_dir / "history.json").write_text(json.dumps(histories, indent=2))
    print(f"saved {ckpt_dir / 'test_split.json'}  ({len(test_recs)} records)")
    print(f"saved {ckpt_dir / 'history.json'}")
    return model, histories


def main():
    parser = argparse.ArgumentParser(description="Train the deepfake tampering localizer")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None, help="override data.root from the config")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs-frozen", type=int, default=None)
    parser.add_argument("--epochs-finetune", type=int, default=None)
    parser.add_argument("--bbox-loss-weight", type=float, default=None)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="skip the ImageNet weights download (smoke tests only)")
    args = parser.parse_args()

    cfg = load_config(args.config, {
        "training.batch_size": args.batch_size,
        "training.epochs_frozen": args.epochs_frozen,
        "training.epochs_finetune": args.epochs_finetune,
        "training.bbox_loss_weight": args.bbox_loss_weight,
    })
    train(cfg, data_root=args.data_root,
          weights=None if args.no_pretrained else "imagenet")


if __name__ == "__main__":
    main()
