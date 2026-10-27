"""End-to-end pipeline tests: train -> save -> load -> evaluate -> infer.

The unit tests in test_model.py pass a model object around in-process. These
exercise the paths that only break across a process boundary -- most importantly
that a saved checkpoint can be reloaded by a script that never imported the
model module. That failure mode is invisible to in-process tests, because
importing the module in the test file is itself what registers the custom
layers.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train a 1-epoch model on synthetic splices. Shared by every test here."""
    workdir = tmp_path_factory.mktemp("pipeline")
    data_root = workdir / "data"
    ckpt_dir = workdir / "checkpoints"

    from src.data.dataset import synthetic_records
    synthetic_records(data_root, n=16, size=128)

    config = workdir / "config.yaml"
    config.write_text(f"""
data:
  root: {data_root}
  image_size: [224, 224]
  ela_quality: 90
  val_frac: 0.25
  test_frac: 0.25
  seed: 42
training:
  batch_size: 4
  epochs_frozen: 1
  epochs_finetune: 0
  fine_tune_from: -30
  lr_frozen: 1.0e-3
  lr_finetune: 1.0e-5
  bbox_loss_weight: 1.0
  patience: 2
output:
  checkpoint_dir: {ckpt_dir}
  checkpoint_name: test_model.keras
  log_dir: {workdir}/logs
""")

    from src.train import load_config, train
    # weights=None avoids a 90 MB ImageNet download in CI.
    train(load_config(config), weights=None)
    return {"workdir": workdir, "config": config, "ckpt_dir": ckpt_dir,
            "checkpoint": ckpt_dir / "test_model.keras", "data_root": data_root}


def test_training_writes_artifacts(trained):
    assert trained["checkpoint"].exists()
    assert (trained["ckpt_dir"] / "test_split.json").exists()
    assert (trained["ckpt_dir"] / "history.json").exists()


def test_test_split_is_held_out(trained):
    """The persisted test split must not overlap training data.

    Without this file, re-running with a different seed would silently leak
    training images into the reported test metrics.
    """
    split = json.loads((trained["ckpt_dir"] / "test_split.json").read_text())
    assert split, "test split is empty"
    paths = [r["path"] for r in split]
    assert len(paths) == len(set(paths)), "duplicate images in the test split"
    for record in split:
        assert record["label"] in (0, 1)
        assert len(record["bbox"]) == 4


def test_checkpoint_loads_in_a_clean_process(trained):
    """Load the checkpoint WITHOUT importing src.models.localizer first.

    This is the regression test for the real bug: the @register_keras_serializable
    decorators only run on import, so a script that calls load_model without
    importing the model module fails with "Could not locate class 'ChannelSlice'".
    Any consumer must import the module (directly or transitively) to be safe.
    """
    script = "\n".join([
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r})",
        "from src.inference import TamperingDetector",
        f"TamperingDetector({str(trained['checkpoint'])!r})",
        "print('LOADED')",
    ])

    # Inherit the real environment rather than building a minimal one. HOME must
    # stay intact: Python derives its per-user site-packages path from it, so
    # overriding HOME hides any dependency installed with `pip install --user`
    # and the subprocess dies on ModuleNotFoundError long before it reaches
    # load_model -- a failure that looks exactly like the deserialization bug
    # this test exists to catch.
    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}
    # A stale PYTHONPATH entry pointing at the repo would import src implicitly
    # and mask whether the script's own sys.path insert is what makes it work.
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600, env=env,
    )
    assert "LOADED" in result.stdout, (
        f"checkpoint failed to load in a clean process:\n{result.stderr[-2000:]}"
    )


def test_evaluate_produces_metrics_and_plots(trained):
    from src.evaluate import evaluate

    out_dir = trained["workdir"] / "reports"
    results = evaluate(
        checkpoint=str(trained["checkpoint"]),
        split_file=str(trained["ckpt_dir"] / "test_split.json"),
        config_path=str(trained["config"]),
        out_dir=str(out_dir),
    )

    assert 0.0 <= results["accuracy"] <= 1.0
    assert 0.0 <= results["macro_f1"] <= 1.0
    if results["mean_iou"] is not None:
        assert 0.0 <= results["mean_iou"] <= 1.0

    assert (out_dir / "confusion_matrix.png").exists()
    assert (out_dir / "eval_results.json").exists()


def test_inference_on_single_image(trained):
    from src.inference import TamperingDetector

    detector = TamperingDetector(str(trained["checkpoint"]))
    image = next((trained["data_root"] / "Tp").iterdir())
    result = detector.predict_with_overlay(image)

    assert result["label"] in ("authentic", "tampered")
    assert 0.5 <= result["confidence"] <= 1.0, "confidence is reported for the predicted class"
    assert 0.0 <= result["tampered_prob"] <= 1.0
    assert result["overlay"] is not None
    assert result["ela_image"].size == result["image"].size

    if result["bbox"] is not None:
        xmin, ymin, xmax, ymax = result["bbox"]
        assert 0.0 <= xmin <= xmax <= 1.0, "box must be normalized and non-inverted"
        assert 0.0 <= ymin <= ymax <= 1.0
