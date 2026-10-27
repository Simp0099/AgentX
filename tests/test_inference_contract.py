"""CPU-only contract tests for inference boundary behavior."""

import hashlib

import numpy as np
import pytest
from PIL import Image

from src import inference
from src.inference import TamperingDetector


class StubModel:
    def __init__(self, score=0.65, bbox=(0.1, 0.2, 0.8, 0.9, 1.0)):
        self.score = score
        self.bbox = bbox
        self.input_shape = None

    def __call__(self, inputs, training=False):
        assert training is False
        self.input_shape = inputs.shape
        return {
            "label": np.asarray([[self.score]], dtype=np.float64),
            "bbox": np.asarray([self.bbox], dtype=np.float64),
        }


@pytest.fixture
def detector():
    active = object.__new__(TamperingDetector)
    config = inference._load_config()
    active.model = StubModel()
    active.image_size = tuple(config["data"]["image_size"])
    active.ela_quality = int(config["data"]["ela_quality"])
    active.max_image_pixels = int(config["forensics"]["max_image_pixels"])
    active.max_file_bytes = int(config["forensics"]["max_file_bytes"])
    active.min_box_area = float(config["forensics"]["min_box_area"])
    active.max_box_coverage = float(config["forensics"]["max_box_coverage"])
    active.supported_extensions = set(config["forensics"]["supported_image_extensions"])
    active.model_version = "a" * 64
    return active


@pytest.fixture
def valid_pil_image():
    rng = np.random.default_rng(42)
    return Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))


def test_predict_raw_returns_unthresholded_contract_and_model_input(detector, valid_pil_image):
    result = detector.predict_raw(valid_pil_image)
    assert set(result) == {"tampered_prob", "bbox_raw", "image_size", "model_version"}
    assert result["tampered_prob"] == 0.65
    assert result["bbox_raw"] == [0.1, 0.2, 0.8, 0.9]
    assert result["image_size"] == valid_pil_image.size
    assert result["model_version"] == "a" * 64
    assert detector.model.input_shape == (1, 224, 224, 6)


def test_predict_threshold_is_per_call_and_confidence_is_predicted_class(detector, valid_pil_image):
    flagged = detector.predict(valid_pil_image, threshold=0.5)
    not_flagged = detector.predict(valid_pil_image, threshold=0.8)
    assert flagged["label"] == "tampered"
    assert flagged["confidence"] == 0.65
    assert flagged["bbox"] == [0.1, 0.2, 0.8, 0.9]
    assert not_flagged["label"] == "authentic"
    assert not_flagged["confidence"] == pytest.approx(0.35)
    assert not_flagged["bbox"] is None


def test_predict_defaults_to_configured_threshold(detector, valid_pil_image):
    assert detector.predict(valid_pil_image)["label"] == "tampered"


@pytest.mark.parametrize("bad_score", [float("nan"), -0.01, 1.01])
def test_malformed_model_score_is_rejected(detector, valid_pil_image, bad_score):
    detector.model.score = bad_score
    with pytest.raises(ValueError, match="tampered score"):
        detector.predict_raw(valid_pil_image)


@pytest.mark.parametrize(
    "bbox",
    [
        (float("nan"), 0.1, 0.8, 0.9, 1.0),
        (-0.1, 0.1, 0.8, 0.9, 1.0),
        (0.8, 0.1, 0.2, 0.9, 1.0),
        (0.0, 0.0, 1.0, 1.0, 1.0),
    ],
)
def test_invalid_model_box_is_returned_as_none(detector, valid_pil_image, bbox):
    detector.model.bbox = bbox
    assert detector.predict_raw(valid_pil_image)["bbox_raw"] is None


def test_model_version_hashes_checkpoint_contents(tmp_path, monkeypatch):
    checkpoint = tmp_path / "stub.keras"
    payload = b"model bytes, not a path hash"
    checkpoint.write_bytes(payload)
    monkeypatch.setattr(inference.tf.keras.models, "load_model", lambda _path: StubModel())
    detector = TamperingDetector(checkpoint)
    assert detector.model_version == hashlib.sha256(payload).hexdigest()


def test_cli_passes_threshold_to_prediction(monkeypatch, capsys):
    received = {}

    class CliDetector:
        def __init__(self, checkpoint):
            received["checkpoint"] = checkpoint

        def predict_with_overlay(self, image, threshold):
            received["image"] = image
            received["threshold"] = threshold
            return {
                "label": "tampered",
                "confidence": 0.8,
                "tampered_prob": 0.8,
                "bbox": [0.1, 0.2, 0.7, 0.8],
                "overlay": Image.new("RGB", (8, 8)),
                "ela_image": Image.new("RGB", (8, 8)),
            }

    monkeypatch.setattr(inference, "TamperingDetector", CliDetector)
    monkeypatch.setattr(
        "sys.argv",
        ["src.inference", "--image", "example.png", "--threshold", "0.8"],
    )
    inference.main()
    assert received == {
        "checkpoint": inference.DEFAULT_CHECKPOINT,
        "image": "example.png",
        "threshold": 0.8,
    }
    assert '"tampered_prob": 0.8' in capsys.readouterr().out


def test_corrupt_image_path_returns_clean_pillow_error(detector, tmp_path):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(OSError):
        detector.predict_raw(corrupt)


def test_missing_image_returns_clean_file_error(detector, tmp_path):
    with pytest.raises(FileNotFoundError):
        detector.predict_raw(tmp_path / "missing.jpg")


def test_unsupported_extension_is_rejected(detector, tmp_path):
    image_path = tmp_path / "unsupported.xyz"
    image_path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Unsupported image extension"):
        detector.predict_raw(image_path)


def test_tiny_image_is_supported(detector):
    result = detector.predict_raw(Image.new("RGB", (1, 1), (128, 128, 128)))
    assert result["image_size"] == (1, 1)
    assert 0.0 <= result["tampered_prob"] <= 1.0


def test_image_above_configured_pixel_limit_is_rejected(detector, monkeypatch):
    config = inference._load_config()
    config["forensics"]["max_image_pixels"] = 100
    monkeypatch.setattr(inference, "_load_config", lambda: config)
    detector.max_image_pixels = config["forensics"]["max_image_pixels"]
    with pytest.raises(ValueError, match="pixel limit"):
        detector.predict_raw(Image.new("RGB", (11, 10), (128, 128, 128)))


def test_file_above_configured_byte_limit_is_rejected(detector, tmp_path):
    image_path = tmp_path / "large.jpg"
    Image.new("RGB", (10, 10)).save(image_path)
    detector.max_file_bytes = image_path.stat().st_size - 1
    with pytest.raises(ValueError, match="byte limit"):
        detector.predict_raw(image_path)


def test_predict_with_overlay_adds_overlay_only_when_box_is_valid(detector, valid_pil_image):
    result = detector.predict_with_overlay(valid_pil_image, threshold=0.5)
    assert result["overlay"].size == valid_pil_image.size

    detector.model.bbox = (float("nan"), 0.1, 0.8, 0.9, 1.0)
    invalid = detector.predict_with_overlay(valid_pil_image, threshold=0.5)
    assert invalid["bbox"] is None
    assert invalid["overlay"] is invalid["image"]


def test_ela_is_full_resolution_then_resized(detector, monkeypatch):
    events = []
    original_compute = inference.compute_ela
    original_array = inference.ela_array

    def compute(image, quality):
        events.append(("compute", image.size))
        return original_compute(image, quality=quality)

    def to_array(image, size, quality):
        events.append(("array", image.size, size))
        return original_array(image, size=size, quality=quality)

    monkeypatch.setattr(inference, "compute_ela", compute)
    monkeypatch.setattr(inference, "ela_array", to_array)
    detector.predict_raw(Image.new("RGB", (80, 40), (100, 100, 100)))
    assert events[0] == ("compute", (80, 40))
    assert events[1] == ("array", (80, 40), (224, 224))
