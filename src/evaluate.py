"""Evaluate a trained checkpoint on the held-out test split.

Metrics, and where each convention comes from:
    Classification : accuracy, macro-F1, classification_report, confusion matrix
                     -- Project 1 (Pet Face Classification)
    Localization   : mean IoU + IoU distribution histogram
                     -- Project 2 (Object Detection)

IoU is computed on tampered images only. Authentic images have no tampered
region, so they have no box to score; including them would inflate or deflate
mean IoU depending on an arbitrary convention rather than on model quality.

Usage:
    python -m src.evaluate
    python -m src.evaluate --checkpoint checkpoints/best_localizer.keras \
                           --split checkpoints/test_split.json
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: works in Docker and CI without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402

from src.data.dataset import make_dataset  # noqa: E402

# This import must come before load_model: the @register_keras_serializable
# decorators on ChannelSlice, ResNetPreprocess, and masked_bbox_loss only run
# when the module is imported, and load_model needs them registered to
# deserialize the checkpoint.
from src.models.localizer import masked_bbox_loss  # noqa: E402, F401
from src.visualize import draw_boxes  # noqa: E402


def intersection_over_union(pred_box, true_box):
    """Vectorized IoU over ``[xmin, ymin, xmax, ymax]`` rows.

    Ported from Project 2 (Object_Detection_TensorFlow), including its smoothing
    factor, which keeps the ratio finite when a predicted box has zero area.
    """
    xmin_pred, ymin_pred, xmax_pred, ymax_pred = np.split(pred_box, 4, axis=1)
    xmin_true, ymin_true, xmax_true, ymax_true = np.split(true_box, 4, axis=1)
    smoothing_factor = 1e-10

    xmin_overlap = np.maximum(xmin_pred, xmin_true)
    xmax_overlap = np.minimum(xmax_pred, xmax_true)
    ymin_overlap = np.maximum(ymin_pred, ymin_true)
    ymax_overlap = np.minimum(ymax_pred, ymax_true)

    pred_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    true_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)
    overlap_area = (np.maximum(xmax_overlap - xmin_overlap, 0)
                    * np.maximum(ymax_overlap - ymin_overlap, 0))
    union_area = pred_area + true_area - overlap_area

    return (overlap_area + smoothing_factor) / (union_area + smoothing_factor)


def _plot_confusion_matrix(cm, path, class_names=("authentic", "tampered")):
    """Annotated confusion matrix. Project 1 used seaborn; matplotlib alone is
    one less dependency to install in Docker for an identical plot."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    # Annotate each cell, flipping text colour on dark cells so it stays legible.
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_iou_distribution(iou, path):
    """IoU histogram — Project 2's ``plt.hist(iou, bins=20)``, annotated."""
    mean_iou = float(iou.mean())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(iou.ravel(), bins=20, color="steelblue", edgecolor="white")
    ax.axvline(mean_iou, color="red", linestyle="--", label=f"mean = {mean_iou:.3f}")
    # 0.5 is the conventional "correct detection" cutoff in detection papers.
    ax.axvline(0.5, color="grey", linestyle=":", label="IoU = 0.5 threshold")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.set_title("IoU Distribution (tampered test images)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_qualitative(records, model, size, ela_quality, path, n=6):
    """Grid of predicted (red) vs ground-truth (green) boxes.

    Mirrors Project 2's ``display_digits_with_boxes``. Numbers alone hide failure
    modes; a model that always predicts a centred half-image box can post a
    respectable mean IoU and still be useless.
    """
    from src.data.dataset import load_example

    tampered = [r for r in records if r[1] == 1][:n]
    if not tampered:
        return
    cols = min(len(tampered), 3)
    rows = (len(tampered) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (path_str, _, true_box) in zip(axes.ravel(), tampered, strict=False):
        x = load_example(path_str, size=size, ela_quality=ela_quality)[None, ...]
        out = model(x, training=False)
        pred_box = np.asarray(out["bbox"]).ravel()[:4]
        prob = float(np.asarray(out["label"]).ravel()[0])
        iou = float(intersection_over_union(
            pred_box.reshape(1, 4), np.asarray(true_box).reshape(1, 4)
        )[0, 0])

        overlay = draw_boxes(
            path_str, [pred_box, list(true_box)],
            colors=["red", "lime"], labels=["pred", "true"],
        )
        ax.imshow(overlay)
        ax.set_title(f"p(tampered)={prob:.2f}  IoU={iou:.2f}", fontsize=10)

    fig.suptitle("Predicted (red) vs ground truth (green)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(checkpoint="checkpoints/best_localizer.keras",
             split_file="checkpoints/test_split.json",
             config_path="configs/default.yaml",
             out_dir=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]
    size = tuple(dcfg["image_size"])
    ela_quality = dcfg["ela_quality"]

    split_path = Path(split_file)
    if not split_path.exists():
        raise FileNotFoundError(
            f"No test split at {split_path}. Run training first -- it writes the "
            f"split so evaluation scores the exact same held-out images."
        )
    records = [(r["path"], r["label"], tuple(r["bbox"]))
               for r in json.loads(split_path.read_text())]

    model = tf.keras.models.load_model(checkpoint)
    ds = make_dataset(records, size=size, batch_size=32, ela_quality=ela_quality)

    y_true_cls, y_pred_cls, y_prob = [], [], []
    pred_boxes, true_boxes = [], []

    for x, y in ds:
        out = model(x, training=False)
        prob = np.asarray(out["label"]).ravel()
        box_pred = np.asarray(out["bbox"])[:, :4]

        y_prob.append(prob)
        y_pred_cls.append((prob > 0.5).astype(int))
        y_true_cls.append(np.asarray(y["label"]).ravel().astype(int))

        target = np.asarray(y["bbox"])
        mask = target[:, 4] > 0.5  # tampered only
        if mask.any():
            pred_boxes.append(box_pred[mask])
            true_boxes.append(target[mask, :4])

    y_true_cls = np.concatenate(y_true_cls)
    y_pred_cls = np.concatenate(y_pred_cls)

    out_dir = Path(out_dir or "reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n== Classification (both classes) ==")
    print(classification_report(y_true_cls, y_pred_cls,
                                target_names=["authentic", "tampered"],
                                zero_division=0))
    accuracy = float((y_true_cls == y_pred_cls).mean())
    macro_f1 = float(f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0))
    print(f"accuracy = {accuracy:.4f}    macro-F1 = {macro_f1:.4f}    n = {len(y_true_cls)}")

    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
    _plot_confusion_matrix(cm, out_dir / "confusion_matrix.png")
    print(f"saved {out_dir / 'confusion_matrix.png'}")

    mean_iou = median_iou = iou_at_50 = None
    if pred_boxes:
        iou = intersection_over_union(np.concatenate(pred_boxes),
                                      np.concatenate(true_boxes))
        mean_iou = float(iou.mean())
        median_iou = float(np.median(iou))
        iou_at_50 = float((iou > 0.5).mean())
        print("\n== Localization (tampered images only) ==")
        print(f"mean IoU = {mean_iou:.4f}    median IoU = {median_iou:.4f}    "
              f"fraction IoU>0.5 = {iou_at_50:.4f}    n = {iou.size}")
        _plot_iou_distribution(iou, out_dir / "iou_distribution.png")
        print(f"saved {out_dir / 'iou_distribution.png'}")
    else:
        print("\nNo tampered images in the test split -- IoU skipped.")

    _plot_qualitative(records, model, size, ela_quality, out_dir / "qualitative.png")
    if (out_dir / "qualitative.png").exists():
        print(f"saved {out_dir / 'qualitative.png'}")

    results = {
        "n_test": int(len(y_true_cls)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mean_iou": mean_iou,
        "median_iou": median_iou,
        "iou_above_0.5": iou_at_50,
        "confusion_matrix": cm.tolist(),
    }
    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"saved {out_dir / 'eval_results.json'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate the tampering localizer")
    parser.add_argument("--checkpoint", default="checkpoints/best_localizer.keras")
    parser.add_argument("--split", default="checkpoints/test_split.json")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.split, args.config, args.out_dir)


if __name__ == "__main__":
    main()
