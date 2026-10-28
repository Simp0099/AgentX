from __future__ import annotations

import io
import zipfile

from PIL import Image

from agent.batch import BatchStatus, process_batch


def test_macos_zip_junk_ignored_and_text_skipped(tmp_path):
    image = io.BytesIO()
    Image.new("RGB", (10, 8), "blue").save(image, format="JPEG")
    path = tmp_path / "members.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(".DS_Store", b"junk")
        archive.writestr("__MACOSX/._x.jpg", b"junk")
        archive.writestr("photo.jpg", image.getvalue())
        archive.writestr("README.txt", b"notes")

    class Detector:
        def predict_raw(self, _image):
            return {"tampered_prob": 0.1, "bbox_raw": None, "image_size": (10, 8), "model_version": "fake"}

    result = process_batch(str(path), str(tmp_path), detector=Detector())
    assert result["status"] == BatchStatus.COMPLETED
    assert not any("__MACOSX" in row["filename"] or row["filename"] == ".DS_Store" for row in result["results"])
    skipped = next(row for row in result["results"] if row["filename"] == "README.txt")
    assert (skipped["status"], skipped["reason"]) == ("Skipped", "not an image")
    assert any(row["filename"] == "photo.jpg" for row in result["results"])
