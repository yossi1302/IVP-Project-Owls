"""Coordinate checks for locate_clock on real training images.

Skipped when the dataset is not in data/raw/. Uses the first images of
train.csv in file order; nothing is chosen by filename.
"""

import csv
import sys
from pathlib import Path

import cv2
import pytest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.localization import (  # noqa: E402
    locate_clock,
    locate_clock_details,
    resize_for_detection,
    to_original_coordinates,
)

data_dir = root / "data" / "raw"
N_IMAGES = 10

pytestmark = pytest.mark.skipif(not (data_dir / "train.csv").exists(),
                                reason="dataset not in data/raw")


def first_training_images():
    with (data_dir / "train.csv").open(encoding="utf-8-sig", newline="") as file:
        names = [row["ID"] for row in csv.DictReader(file)][:N_IMAGES]
    return [cv2.imread(str(data_dir / "train" / "train" / name)) for name in names]


@pytest.fixture(scope="module")
def images():
    return first_training_images()


def test_result_is_in_original_pixels(images):
    """Detecting on the full image must equal detecting on the shrunk copy
    and mapping back, because detection runs on exactly that copy."""
    resized_count = 0
    for image in images:
        small, scale = resize_for_detection(image)
        resized_count += scale < 1.0
        full = locate_clock(image)
        shrunk = locate_clock(small)
        assert (full is None) == (shrunk is None)
        if full is not None:
            assert full == pytest.approx(to_original_coordinates(*shrunk, scale))
    assert resized_count > 0, "no image was large enough to test the resize path"


def test_circles_lie_on_the_image(images):
    for image in images:
        details = locate_clock_details(image)
        if details is None:
            continue
        height, width = image.shape[:2]
        assert 0 <= details["cx"] < width and 0 <= details["cy"] < height
        assert 0 < details["radius"] <= max(width, height)
        assert 0 < details["scale"] <= 1.0


def test_details_match_locate_clock(images):
    for image in images:
        details = locate_clock_details(image)
        result = locate_clock(image)
        if details is None:
            assert result is None
        else:
            assert result == (details["cx"], details["cy"], details["radius"])
