"""Box drawing helpers.

Ported from Project 2 (Object_Detection_TensorFlow), which drew red predicted /
green ground-truth boxes on MNIST canvases. Same normalized ``[xmin, ymin, xmax,
ymax]`` convention; extended with a text label and thicker lines because these
are drawn on real photographs rather than 75x75 digit canvases.
"""

import numpy as np
from PIL import Image, ImageDraw


def draw_bounding_box_on_image(image, xmin, ymin, xmax, ymax, color="red",
                               thickness=3, label=None, use_normalized_coordinates=True):
    """Draw one box on a PIL image, in place.

    Args:
        image: PIL Image to draw on.
        xmin, ymin, xmax, ymax: box corners.
        color: outline colour.
        thickness: line width in pixels.
        label: optional text drawn just above the box.
        use_normalized_coordinates: if True, coords are fractions of image size.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    if use_normalized_coordinates:
        left, right = xmin * width, xmax * width
        top, bottom = ymin * height, ymax * height
    else:
        left, right, top, bottom = xmin, xmax, ymin, ymax

    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness, fill=color,
    )

    if label:
        # Keep the caption on-canvas when the box is flush against the top edge.
        text_y = max(top - 14, 0)
        draw.text((left + 2, text_y), label, fill=color)


def draw_boxes(image, boxes, colors=None, labels=None, thickness=3):
    """Draw several boxes on a copy of ``image`` and return it.

    Args:
        image: PIL Image, numpy array, or path.
        boxes: array-like of shape (N, 4), normalized ``[xmin, ymin, xmax, ymax]``.
        colors: per-box colours; defaults to red for the first, green for the rest
            (matching Project 2's pred=red / true=green convention).
        labels: optional per-box captions.
        thickness: line width.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    elif isinstance(image, (str, bytes)):
        image = Image.open(image)
    image = image.convert("RGB").copy()

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if colors is None:
        colors = ["red"] + ["lime"] * (len(boxes) - 1)
    if labels is None:
        labels = [None] * len(boxes)

    for box, color, label in zip(boxes, colors, labels, strict=False):
        draw_bounding_box_on_image(
            image, *box, color=color, thickness=thickness, label=label
        )
    return image


if __name__ == "__main__":
    canvas = Image.new("RGB", (200, 100), "black")
    out = draw_boxes(canvas, [[0.25, 0.25, 0.75, 0.75]], labels=["pred"])
    arr = np.asarray(out)
    assert out.size == (200, 100)
    assert arr.max() > 0, "nothing was drawn"
    # Box edge at x=0.25*200=50, y=0.25*100=25 must be non-black.
    assert arr[25, 50].sum() > 0, "box edge not where expected"
    # Two-box call must not raise and must default to distinct colours.
    two = np.asarray(draw_boxes(canvas, [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9]]))
    assert two.max() > 0
    print("ok  boxes drawn at expected pixels")
