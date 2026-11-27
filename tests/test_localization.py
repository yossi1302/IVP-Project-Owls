"""Checks for src/localization.py using synthetic drawn clocks.

These images are generated here, not taken from the dataset. They check the
coordinate handling and failure behaviour, not real-world accuracy.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.localization import (  # noqa: E402
    PARAMS,
    detect_circle,
    locate_clock,
    locate_clock_details,
    resize_for_detection,
    to_original_coordinates,
)


def draw_clock(width, height, cx, cy, radius, seed=0):
    """Light noisy background with a dark rim, tick marks and two hands."""
    rng = np.random.default_rng(seed)
    image = rng.integers(170, 210, size=(height, width, 3), dtype=np.uint8)
    rim = max(2, radius // 40)
    cv2.circle(image, (cx, cy), radius, (240, 240, 240), -1)
    cv2.circle(image, (cx, cy), radius, (30, 30, 30), rim)
    for k in range(12):
        a = k * np.pi / 6
        p1 = (int(cx + 0.85 * radius * np.cos(a)), int(cy + 0.85 * radius * np.sin(a)))
        p2 = (int(cx + 0.95 * radius * np.cos(a)), int(cy + 0.95 * radius * np.sin(a)))
        cv2.line(image, p1, p2, (20, 20, 20), rim)
    cv2.line(image, (cx, cy), (cx, cy - int(0.5 * radius)), (10, 10, 10), rim)
    cv2.line(image, (cx, cy), (cx + int(0.7 * radius), cy), (10, 10, 10), rim // 2 + 1)
    return image


def assert_close(found, cx, cy, radius, rel_tol=0.03):
    assert found is not None, "clock was not detected"
    fx, fy, fr = found
    assert abs(fx - cx) <= rel_tol * radius
    assert abs(fy - cy) <= rel_tol * radius
    assert abs(fr - radius) <= rel_tol * radius


def test_large_image_is_mapped_back_to_original_pixels():
    # 2400 px wide -> detection runs at 512 px, so scale is about 0.21.
    # Without mapping back, the answer would be roughly 5x too small.
    cx, cy, r = 1300, 900, 600
    image = draw_clock(2400, 1800, cx, cy, r)
    assert_close(locate_clock(image), cx, cy, r)


def test_small_image_is_not_resized():
    cx, cy, r = 200, 160, 110
    image = draw_clock(400, 320, cx, cy, r)
    small, scale = resize_for_detection(image)
    assert scale == 1.0 and small is image
    assert_close(locate_clock(image), cx, cy, r)


def test_portrait_image():
    # Height is the long side here; checks that width/height are not swapped.
    cx, cy, r = 500, 1400, 420
    image = draw_clock(1000, 2000, cx, cy, r)
    assert_close(locate_clock(image), cx, cy, r)


def test_resize_scale_and_round_trip():
    image = np.zeros((1000, 2000, 3), np.uint8)
    small, scale = resize_for_detection(image, work_size=500)
    assert small.shape[:2] == (250, 500)
    assert scale == pytest.approx(0.25)
    assert to_original_coordinates(100, 50, 20, scale) == pytest.approx((400, 200, 80))


def test_grayscale_input_works():
    cx, cy, r = 300, 300, 200
    gray = cv2.cvtColor(draw_clock(600, 600, cx, cy, r), cv2.COLOR_BGR2GRAY)
    assert_close(locate_clock(gray), cx, cy, r)


def test_blank_image_returns_none():
    assert locate_clock(np.full((600, 800, 3), 128, np.uint8)) is None


def test_noise_only_returns_none():
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 256, size=(600, 800, 3), dtype=np.uint8)
    assert locate_clock(noise) is None


def test_circle_smaller_than_min_radius_is_ignored():
    # The clock radius is below min_radius_frac of the short side.
    r = int(0.5 * PARAMS["min_radius_frac"] * 600)
    image = draw_clock(800, 600, 400, 300, r)
    assert locate_clock(image) is None


def faint_rim_image(delta=8):
    """Grey noise with a thin rim only `delta` grey levels brighter."""
    rng = np.random.default_rng(0)
    image = rng.normal(128, 3, (600, 800)).clip(0, 255).astype(np.uint8)
    cv2.circle(image, (420, 300), 200, 128 + delta, 6)
    return image


def test_faint_rim_needs_fallback():
    image = faint_rim_image()
    small, _scale = resize_for_detection(image)
    assert detect_circle(small) is None, "first pass alone should miss this rim"
    details = locate_clock_details(image)
    assert details is not None and details["method"] == "fallback"
    assert_close((details["cx"], details["cy"], details["radius"]), 420, 300, 200)


def test_clear_clock_uses_first_pass():
    details = locate_clock_details(draw_clock(1200, 900, 600, 450, 300))
    assert details is not None and details["method"] == "primary"
    assert details["score"] >= PARAMS["weak_score"]


@pytest.mark.parametrize("bad", [None, np.zeros((0, 0, 3), np.uint8)])
def test_invalid_input_raises(bad):
    with pytest.raises(ValueError):
        locate_clock(bad)
