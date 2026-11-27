"""Unit and synthetic-clock tests for hand detection and angle conversion."""

import math

import cv2
import numpy as np
import pytest

from src.evaluation import parse_time
from src.time_reading import (
    angular_difference,
    angles_to_time,
    detect_hand_candidates,
    extract_clock_face,
    point_to_clock_angle,
)


def make_synthetic_clock(
    time_str: str | None,
    *,
    image_size: tuple[int, int] = (500, 500),
    circle: tuple[float, float, float] = (250.0, 250.0, 180.0),
    dial_color: tuple[int, int, int] = (245, 245, 245),
    bg_color: tuple[int, int, int] = (180, 180, 180),
    hand_color: tuple[int, int, int] = (20, 20, 20),
    hour_len_ratio: float = 0.50,
    minute_len_ratio: float = 0.78,
    hour_width_ratio: float = 0.045,
    minute_width_ratio: float = 0.028,
    draw_ticks: bool = True,
    draw_bezel: bool = True,
    draw_center_pin: bool = True,
    draw_counterweight: bool = False,
    draw_brand_bar: bool = False,
    second_hand_angle_deg: float | None = None,
    second_hand_color: tuple[int, int, int] = (20, 20, 220),
) -> np.ndarray:
    """Render a synthetic BGR analogue clock image for geometry/detection tests."""
    height, width = image_size
    cx, cy, radius = circle
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    cv2.circle(img, (int(round(cx)), int(round(cy))), int(round(radius)), dial_color, -1)
    if draw_bezel:
        cv2.circle(
            img,
            (int(round(cx)), int(round(cy))),
            int(round(radius)),
            hand_color,
            max(2, int(round(radius * 0.03))),
        )

    if draw_ticks:
        for tick in range(60):
            angle = math.radians(tick * 6.0)
            ux, uy = math.sin(angle), -math.cos(angle)
            is_hour = tick % 5 == 0
            r_in = radius * (0.82 if is_hour else 0.89)
            r_out = radius * 0.95
            thickness = max(1, int(round(radius * (0.022 if is_hour else 0.01))))
            p1 = (int(round(cx + r_in * ux)), int(round(cy + r_in * uy)))
            p2 = (int(round(cx + r_out * ux)), int(round(cy + r_out * uy)))
            cv2.line(img, p1, p2, hand_color, thickness, lineType=cv2.LINE_AA)

    if draw_brand_bar:
        y_bar = int(round(cy - radius * 0.42))
        x_left = int(round(cx - radius * 0.18))
        x_right = int(round(cx + radius * 0.18))
        cv2.rectangle(img, (x_left, y_bar - 3), (x_right, y_bar + 3), hand_color, -1)

    def draw_hand(angle_deg: float, length_ratio: float, width_ratio: float, color, tail_ratio: float = 0.0):
        angle = math.radians(angle_deg)
        ux, uy = math.sin(angle), -math.cos(angle)
        p_start = (
            int(round(cx - radius * tail_ratio * ux)),
            int(round(cy - radius * tail_ratio * uy)),
        )
        p_end = (
            int(round(cx + radius * length_ratio * ux)),
            int(round(cy + radius * length_ratio * uy)),
        )
        thickness = max(1, int(round(radius * width_ratio)))
        cv2.line(img, p_start, p_end, color, thickness, lineType=cv2.LINE_AA)

    if time_str is not None:
        hour, minute = parse_time(time_str)
        minute_angle = minute * 6.0
        hour_angle = ((hour % 12) + minute / 60.0) * 30.0
        tail = 0.16 if draw_counterweight else 0.0
        draw_hand(hour_angle, hour_len_ratio, hour_width_ratio, hand_color, tail_ratio=tail)
        draw_hand(minute_angle, minute_len_ratio, minute_width_ratio, hand_color, tail_ratio=tail)

    if second_hand_angle_deg is not None:
        draw_hand(
            second_hand_angle_deg,
            0.82,
            0.012,
            second_hand_color,
            tail_ratio=0.18 if draw_counterweight else 0.0,
        )

    if draw_center_pin:
        cv2.circle(
            img,
            (int(round(cx)), int(round(cy))),
            max(2, int(round(radius * 0.04))),
            hand_color,
            -1,
        )

    return img


class TestClockGeometry:
    @pytest.mark.parametrize(
        ("dx", "dy", "expected_deg"),
        [
            (0.0, -10.0, 0.0),
            (10.0, -10.0, 45.0),
            (10.0, 0.0, 90.0),
            (10.0, 10.0, 135.0),
            (0.0, 10.0, 180.0),
            (-10.0, 10.0, 225.0),
            (-10.0, 0.0, 270.0),
            (-10.0, -10.0, 315.0),
        ],
    )
    def test_point_to_clock_angle(self, dx, dy, expected_deg):
        cx, cy = 120.0, 80.0
        angle = point_to_clock_angle(cx, cy, cx + dx, cy + dy)
        assert angle == pytest.approx(expected_deg, abs=1e-6)

    def test_point_to_clock_angle_zero_vector_raises(self):
        with pytest.raises(ValueError):
            point_to_clock_angle(50.0, 50.0, 50.0, 50.0)

    def test_angular_difference(self):
        assert angular_difference(355.0, 5.0) == pytest.approx(10.0)
        assert angular_difference(5.0, 355.0) == pytest.approx(10.0)
        assert angular_difference(0.0, 180.0) == pytest.approx(180.0)
        assert angular_difference(90.0, 270.0) == pytest.approx(180.0)

    @pytest.mark.parametrize(
        ("hour_angle", "minute_angle", "expected_time"),
        [
            (0.0, 0.0, "12:00"),
            (90.0, 0.0, "03:00"),
            (180.0, 0.0, "06:00"),
            (270.0, 0.0, "09:00"),
            (89.0, 348.0, "02:58"),
            (91.5, 348.0, "02:58"),
            (0.5, 354.0, "11:59"),
            (89.5, 359.0, "03:00"),
            (15.0, 180.0, "12:30"),
            (29.5, 354.0, "12:59"),
        ],
    )
    def test_angles_to_time_continuous_hour_hand(self, hour_angle, minute_angle, expected_time):
        assert angles_to_time(hour_angle, minute_angle) == expected_time


class TestCandidateHandDetection:
    def test_detects_both_hands_on_synthetic_clock(self):
        circle = (250.0, 250.0, 180.0)
        img = make_synthetic_clock("03:00", circle=circle)
        face = extract_clock_face(img, circle)
        candidates = detect_hand_candidates(face)
        assert len(candidates) == 2
        angles = sorted(c.angle_deg for c in candidates)
        assert angular_difference(angles[0], 0.0) <= 3.0
        assert angular_difference(angles[1], 90.0) <= 3.0

    def test_detects_no_candidates_on_blank_dial(self):
        circle = (200.0, 200.0, 150.0)
        blank = make_synthetic_clock(None, image_size=(400, 400), circle=circle)
        face = extract_clock_face(blank, circle)
        assert detect_hand_candidates(face) == []
