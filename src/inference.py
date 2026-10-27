"""Single-image inference: real/tampered verdict plus the tampered-region box.

Usage:
    python -m src.inference --image path/to/photo.jpg
    python -m src.inference --image photo.jpg --save-overlay out.png

As a library:
    from src.inference import TamperingDetector
    detector = TamperingDetector("checkpoints/best_localizer.keras")
    result = detector.predict("photo.jpg")
"""

import argparse
import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from PIL import Image

from src.data.ela import compute_ela, ela_array

# Import for its side effect: the @register_keras_serializable decorators on
# ChannelSlice, ResNetPreprocess, and masked_bbox_loss only run on import, and
# load_model cannot deserialize the checkpoint without them registered.
from src.models.localizer import masked_bbox_loss  # noqa: F401
from src.visualize import draw_boxes

DEFAULT_CHECKPOINT = "checkpoints/best_localizer.keras"
# Config path relative to the project root (same location as app.py).
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"

# Process-wide lock: Keras predict is not thread-safe.
_inference_lock = threading.Lock()


def _load_config():
    """Load the default config YAML."""
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


class TamperingDetector:
    """Loads a trained checkpoint once and serves single-image predictions.

    The model is loaded eagerly in ``__init__`` rather than lazily, so a bad
    checkpoint path fails at startup instead of on the first user upload.
    """

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, image_size=None, ela_quality=None):
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint}. Train one first:\n"
                f"  python -m src.train --config configs/default.yaml"
            )
        # Custom layers/losses self-register via @register_keras_serializable,
        # so no custom_objects dict is needed here.
        cfg = _load_config()
        data_cfg = cfg.get("data", {})
        forensics_cfg = cfg.get("forensics", {})
        configured_size = data_cfg["image_size"]
        self.image_size = tuple(image_size or configured_size)
        self.ela_quality = int(ela_quality if ela_quality is not None else data_cfg["ela_quality"])
        self.max_image_pixels = int(forensics_cfg["max_image_pixels"])
        self.max_file_bytes = int(forensics_cfg["max_file_bytes"])
        self.min_box_area = float(forensics_cfg["min_box_area"])
        self.max_box_coverage = float(forensics_cfg["max_box_coverage"])
        self.supported_extensions = {
            str(extension).lower() for extension in forensics_cfg["supported_image_extensions"]
        }
        self.model = tf.keras.models.load_model(checkpoint)
        # Fingerprint checkpoint contents once, not the location where it lives.
        with checkpoint.open("rb") as model_file:
            self.model_version = hashlib.file_digest(model_file, "sha256").hexdigest()
        # Threshold is not stored as mutable state; it is passed per call.

    def preprocess(self, image):
        """Image -> (batched 6-channel input, original PIL image, ELA PIL image).

        Note: ELA is computed on the FULL-resolution image and only then resized.
        Computing it after downscaling would destroy the compression artifacts
        it exists to measure -- resampling rewrites exactly the high-frequency
        detail that carries the signal.
        """
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if image_path.suffix.lower() not in self.supported_extensions:
                raise ValueError(f"Unsupported image extension: {image_path.suffix or '(none)'}")
            file_size = image_path.stat().st_size
            if file_size > self.max_file_bytes:
                raise ValueError(
                    f"Image file is {file_size} bytes; byte limit is {self.max_file_bytes}"
                )
            with Image.open(image_path) as opened:
                if opened.width * opened.height > self.max_image_pixels:
                    raise ValueError(
                        f"Image has {opened.width * opened.height} pixels; "
                        f"pixel limit is {self.max_image_pixels}"
                    )
                image = opened.copy()
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        elif not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image, numpy array, or image path")
        if image.width * image.height > self.max_image_pixels:
            raise ValueError(
                f"Image has {image.width * image.height} pixels; "
                f"pixel limit is {self.max_image_pixels}"
            )
        image = image.convert("RGB")

        # ELA is computed on the FULL-resolution image and only then resized.
        # Computing it after downscaling would destroy the compression artifacts
        # it exists to measure -- resampling rewrites exactly the high-frequency
        # detail that carries the signal.
        ela_pil = compute_ela(image, quality=self.ela_quality)
        ela = ela_array(image, size=self.image_size, quality=self.ela_quality) * 255.0
        rgb = np.asarray(image.resize(self.image_size), dtype=np.float32)
        x = np.concatenate([rgb, ela], axis=-1)[None, ...]
        return x, image, ela_pil

    def _infer(self, image):
        """Preprocess once, run the locked model, and validate its outputs."""
        x, original, ela_pil = self.preprocess(image)
        with _inference_lock:
            out = self.model(x, training=False)
        if not isinstance(out, dict) or "label" not in out or "bbox" not in out:
            raise ValueError("Model output must contain label and bbox tensors")

        score_values = np.asarray(out["label"], dtype=np.float64)
        if score_values.size != 1:
            raise ValueError("Model label output must contain one score per image")
        tampered_prob = float(score_values.reshape(-1)[0])
        if not np.isfinite(tampered_prob) or not 0.0 <= tampered_prob <= 1.0:
            raise ValueError("Model tampered score must be finite and within [0, 1]")

        box_values = np.asarray(out["bbox"], dtype=np.float64)
        if box_values.size != 5:
            raise ValueError("Model bbox output must contain four corners and has_box")
        coordinates = box_values.reshape(-1)[:4]
        bbox_raw = None
        if np.all(np.isfinite(coordinates)):
            xmin, ymin, xmax, ymax = map(float, coordinates)
            if 0.0 <= xmin < xmax <= 1.0 and 0.0 <= ymin < ymax <= 1.0:
                area = (xmax - xmin) * (ymax - ymin)
                if self.min_box_area <= area <= self.max_box_coverage:
                    bbox_raw = [xmin, ymin, xmax, ymax]

        return tampered_prob, bbox_raw, original, ela_pil

    def predict_raw(self, image):
        """Raw prediction without threshold applied.

        Returns a dict with:
            tampered_prob : float, raw sigmoid score for the tampered class in [0,1]
            bbox_raw      : [xmin, ymin, xmax, ymax] normalized to [0,1], or None
                          if validation fails
            image_size    : (width, height) of the original image in pixels
            model_version : str, SHA-256 of the checkpoint file
        """
        tampered_prob, bbox_raw, original, _ela_pil = self._infer(image)

        return {
            "tampered_prob": tampered_prob,
            "bbox_raw": bbox_raw,
            "image_size": original.size,
            "model_version": self.model_version,
        }

    def predict(self, image, threshold=None):
        """Predict on one image.

        Args:
            image: PIL Image, numpy array, or path/str to an image file.
            threshold: float in [0,1] to decide tampered vs authentic.
                       Defaults to the value from ``configs/default.yaml``.

        Returns a dict with:
            label           : "tampered" | "authentic"
            confidence      : confidence in the predicted class, in [0, 1]
                            (confidence in the predicted class, NOT the raw score)
            tampered_prob   : raw sigmoid score for the tampered class
            bbox            : normalized [xmin, ymin, xmax, ymax], or None if authentic
            ela_image       : the ELA map as a PIL Image
            image           : the original image as a PIL Image
        """
        if threshold is None:
            # Load threshold from config YAML if not provided explicitly.
            cfg = _load_config()
            threshold = cfg["output"]["threshold"]
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be finite and within [0, 1]")

        tampered_prob, box, original, ela_pil = self._infer(image)

        is_tampered = tampered_prob > threshold
        # Report confidence in whichever class was predicted, not the raw sigmoid.
        # A 0.05 score is 95% confident the image is authentic, not 5% confident.
        confidence = tampered_prob if is_tampered else 1.0 - tampered_prob

        # Guard against an inverted box, which sigmoid outputs allow: nothing
        # forces xmin < xmax, and an inverted box yields a negative-area IoU.
        return {
            "label": "tampered" if is_tampered else "authentic",
            "confidence": confidence,
            "tampered_prob": tampered_prob,
            # No box for authentic images: the head was never trained on them
            # (masked loss), so whatever it emits there is meaningless.
            "bbox": box if is_tampered else None,
            "ela_image": ela_pil,
            "image": original,
        }

    def predict_with_overlay(self, image, threshold=None):
        """Same as :meth:`predict`, plus an ``overlay`` image with the box drawn."""
        result = self.predict(image, threshold=threshold)
        if result["bbox"] is not None:
            caption = f"tampered {result['confidence'] * 100:.0f}%"
            result["overlay"] = draw_boxes(result["image"], [result["bbox"]], labels=[caption])
        else:
            result["overlay"] = result["image"]
        return result


def main():
    parser = argparse.ArgumentParser(description="Detect and localize image tampering")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--save-overlay", default=None, help="write the box overlay here")
    parser.add_argument("--save-ela", default=None, help="write the ELA map here")
    args = parser.parse_args()

    detector = TamperingDetector(args.checkpoint)
    result = detector.predict_with_overlay(args.image, threshold=args.threshold)

    print(
        json.dumps(
            {
                "image": args.image,
                "label": result["label"],
                "confidence": round(result["confidence"], 4),
                "tampered_prob": round(result["tampered_prob"], 4),
                "bbox": [round(v, 4) for v in result["bbox"]] if result["bbox"] else None,
            },
            indent=2,
        )
    )

    if args.save_overlay:
        result["overlay"].save(args.save_overlay)
        print(f"saved {args.save_overlay}")
    if args.save_ela:
        result["ela_image"].save(args.save_ela)
        print(f"saved {args.save_ela}")


if __name__ == "__main__":
    main()
