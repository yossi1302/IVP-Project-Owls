"""Classical image-processing hand detection and time reading for analogue clocks.

Geometry conventions
--------------------
- OpenCV images use pixel coordinates ``(x, y)`` where ``x`` increases to the
  right and ``y`` increases **downward**.
- Clock angles are measured in degrees **clockwise from 12 o'clock** in
  ``[0, 360)``:
    - 12 o'clock -> ``0 deg``   (direction ``dx = 0, dy = -1``)
    -  3 o'clock -> ``90 deg``  (direction ``dx = +1, dy = 0``)
    -  6 o'clock -> ``180 deg`` (direction ``dx = 0, dy = +1``)
    -  9 o'clock -> ``270 deg`` (direction ``dx = -1, dy = 0``)
- Because ``y`` points downward, the upward vector is ``-dy``, giving:
      ``angle_deg = degrees(atan2(dx, -dy)) % 360``
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np

from src.evaluation import format_time

WORK_RADIUS = 200
INNER_MASK_RATIO = 0.86
MIN_CLOCK_RADIUS = 15.0


@dataclass(frozen=True)
class CandidateHand:
    """Geometric and appearance summary of a detected radial hand candidate."""

    angle_deg: float
    tip_radius: float  # Normalized to [0, 1] relative to clock radius
    width: float  # Median thickness normalized to clock radius
    solidity: float  # Fraction of radial samples belonging to foreground
    mean_saturation: float  # Mean HSV saturation in [0, 255]
    redness: float  # Mean (R - max(G, B)) along hand pixels
    mean_contrast: float = 0.0  # Mean local contrast along hand pixels
    segment_count: int = 1


def point_to_clock_angle(cx: float, cy: float, x: float, y: float) -> float:
    """Convert vector ``(cx, cy) -> (x, y)`` to clockwise degrees from 12 o'clock.

    Accounts for OpenCV's image coordinate system where ``y`` increases downward.
    """
    dx = x - cx
    dy = y - cy
    if dx == 0.0 and dy == 0.0:
        raise ValueError("Cannot compute angle for a zero-length vector")
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def angular_difference(angle_a: float, angle_b: float) -> float:
    """Return the shortest circular difference in degrees in ``[0, 180]``."""
    diff = abs((angle_a - angle_b) % 360.0)
    return min(diff, 360.0 - diff)


def hour_minute_consistency_error(
    hour_angle_deg: float, minute_angle_deg: float
) -> float:
    """Measure geometric inconsistency (in ``[0, 15]`` deg) of a hand pair.

    On a mechanical 12-hour clock, the fractional position of the hour hand
    between consecutive hour marks (``hour_angle % 30``) must match
    ``(minute_angle % 360) / 12``. A true ``(hour, minute)`` pair has a small
    error (typically 0-4 deg), whereas swapped hands or unrelated distractors
    often violate this relation by up to 15 degrees.
    """
    expected_offset = (minute_angle_deg % 360.0) / 12.0
    observed_offset = hour_angle_deg % 30.0
    diff = abs(observed_offset - expected_offset)
    return min(diff, 30.0 - diff)


def _circular_mean_deg(angles_deg: Sequence[float], weights: Sequence[float]) -> float:
    sin_sum = sum(w * math.sin(math.radians(a)) for a, w in zip(angles_deg, weights))
    cos_sum = sum(w * math.cos(math.radians(a)) for a, w in zip(angles_deg, weights))
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0


def angles_to_time(hour_angle_deg: float, minute_angle_deg: float) -> str:
    """Convert clockwise hand angles (degrees from 12 o'clock) to ``HH:MM``.

    The minute hand determines the minute (6 degrees per minute). Because the
    hour hand advances continuously between hour marks as minutes progress
    (``(hour + minute / 60) * 30`` degrees), we select the hour ``0..11`` whose
    expected hour-hand angle best matches ``hour_angle_deg`` on the circle.
    This avoids off-by-one-hour errors near the top of the hour (e.g. 02:58).
    """
    minute_norm = minute_angle_deg % 360.0
    hour_norm = hour_angle_deg % 360.0

    minute = int(round(minute_norm / 6.0)) % 60

    best_hour = 0
    best_error = 360.0
    for hour_idx in range(12):
        expected_hour_angle = ((hour_idx + minute / 60.0) * 30.0) % 360.0
        error = angular_difference(hour_norm, expected_hour_angle)
        if error < best_error:
            best_error = error
            best_hour = hour_idx

    display_hour = 12 if best_hour == 0 else best_hour
    return format_time(display_hour, minute)


def _validate_inputs(
    image: np.ndarray, circle: Sequence[float] | None
) -> tuple[float, float, float] | None:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None
    if image.ndim not in (2, 3):
        return None
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        return None
    if circle is None or len(circle) != 3:
        return None

    cx, cy, radius = float(circle[0]), float(circle[1]), float(circle[2])
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(radius)):
        return None
    if radius < MIN_CLOCK_RADIUS:
        return None

    height, width = image.shape[:2]
    if not (0 <= cx < width and 0 <= cy < height):
        return None
    return cx, cy, radius


def extract_clock_face(
    image: np.ndarray,
    circle: Sequence[float],
    work_radius: int = WORK_RADIUS,
) -> np.ndarray:
    """Crop the square region around ``circle`` and resize to ``2 * work_radius``.

    Uses border replication if the circle extends slightly past the image edge
    so the detected circle center always lands at ``(work_radius, work_radius)``.
    """
    cx, cy, radius = float(circle[0]), float(circle[1]), float(circle[2])
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image

    size = 2 * work_radius
    scale = work_radius / radius
    matrix = np.array(
        [
            [scale, 0.0, work_radius - cx * scale],
            [0.0, scale, work_radius - cy * scale],
        ],
        dtype=np.float32,
    )
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.warpAffine(
        bgr,
        matrix,
        (size, size),
        flags=interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _create_circle_mask(
    size: int, radius: float, center: tuple[float, float] | None = None
) -> np.ndarray:
    if center is None:
        cx, cy = size // 2, size // 2
    else:
        cx, cy = int(round(center[0])), int(round(center[1]))
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(round(radius)), 255, thickness=-1)
    return mask


def compute_contrast_map(
    clock_bgr: np.ndarray, work_radius: int = WORK_RADIUS
) -> np.ndarray:
    """Compute a polarity- and color-invariant contrast map of the clock face.

    Estimates the local dial background with a large median filter (at half
    resolution for speed) in CIELAB space. Unlike Gaussian blurring, a median
    filter completely removes narrow clock hands without creating bright halos
    around them, while preserving gradual lighting gradients and capturing both
    luminance and chromatic contrast (e.g. colored hands on dark or tinted dials).
    """
    size = clock_bgr.shape[0]
    lab = cv2.cvtColor(clock_bgr, cv2.COLOR_BGR2LAB)
    blurred_lab = cv2.GaussianBlur(lab, (3, 3), 0)

    # Half-resolution median background: hands (<12px at half-res) cover <50%
    # of a 27x27 window, so the median tracks the clean dial face with zero halo.
    half_size = max(64, size // 2)
    small_lab = cv2.resize(blurred_lab, (half_size, half_size), interpolation=cv2.INTER_AREA)
    med_k = max(15, int(half_size * 0.14) | 1)
    small_bg = cv2.medianBlur(small_lab, med_k)
    local_bg = cv2.resize(small_bg, (size, size), interpolation=cv2.INTER_LINEAR)

    diff = blurred_lab.astype(np.float32) - local_bg.astype(np.float32)
    lab_dist = np.sqrt(
        diff[:, :, 0] ** 2 + 1.15 * (diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2)
    )

    # Blend modestly with global contrast against the core dial median so very
    # wide hands still keep solid interior response.
    core_mask = _create_circle_mask(size, work_radius * 0.65)
    core_pixels = blurred_lab[core_mask > 0].astype(np.float32)
    median_lab = np.median(core_pixels, axis=0)
    gdiff = blurred_lab.astype(np.float32) - median_lab
    global_dist = np.sqrt(
        gdiff[:, :, 0] ** 2 + 1.0 * (gdiff[:, :, 1] ** 2 + gdiff[:, :, 2] ** 2)
    )

    combined = np.maximum(lab_dist, 0.55 * global_dist)
    return np.clip(combined, 0, 255).astype(np.uint8)


def build_foreground_mask(
    clock_bgr: np.ndarray,
    work_radius: int = WORK_RADIUS,
    contrast_map: np.ndarray | None = None,
) -> np.ndarray:
    """Build a polarity-invariant binary mask of hand/marking foreground pixels."""
    size = clock_bgr.shape[0]
    inner_mask = _create_circle_mask(size, work_radius * INNER_MASK_RATIO)
    core_mask = _create_circle_mask(size, work_radius * 0.65)

    if contrast_map is None:
        contrast_map = compute_contrast_map(clock_bgr, work_radius)

    # Compute Otsu threshold on the core dial (0.65 R), clipped at the 98th
    # percentile so tiny specular highlights cannot inflate the threshold.
    core_values = contrast_map[core_mask > 0]
    p98 = float(np.percentile(core_values, 98))
    clipped = np.clip(core_values, 0, max(1.0, p98)).astype(np.uint8)
    otsu_thresh, _ = cv2.threshold(
        clipped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    thresh = max(18.0, float(otsu_thresh) * 0.85)

    fg_mask = np.zeros((size, size), dtype=np.uint8)
    fg_mask[(contrast_map >= thresh) & (inner_mask > 0)] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    return fg_mask


def estimate_hand_pivot(
    lines: np.ndarray | None,
    fg_mask: np.ndarray,
    work_radius: int = WORK_RADIUS,
) -> tuple[float, float]:
    """Estimate the true hand pivot ``(px, py)`` inside the normalized clock face.

    When clock localization fits the outer bezel or casing on a tilted photo,
    the hand pivot can be shifted by 5-20% of the radius from the circle center.
    We locate the pivot by accumulating votes from extended Hough line segments:
    real clock hands all intersect at the pivot, creating a strong multi-angle
    peak near the center that also lies on foreground pixels.
    """
    size = fg_mask.shape[0]
    center = float(size // 2)
    if lines is None or len(lines) < 2:
        return center, center

    max_shift = work_radius * 0.28
    # Keep segments whose infinite lines pass within max_shift of the circle center.
    kept_segments: list[tuple[float, float, float, float, float, float]] = []
    for x1, y1, x2, y2 in np.asarray(lines, dtype=np.float64).reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len < work_radius * 0.14:
            continue
        perp = abs(dx * (y1 - center) - dy * (x1 - center)) / seg_len
        if perp > max_shift:
            continue
        # Reject segments located solely on the very outer rim.
        mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if math.hypot(mx - center, my - center) > work_radius * 0.78:
            continue
        angle_mod180 = math.degrees(math.atan2(dy, dx)) % 180.0
        kept_segments.append((x1, y1, x2, y2, seg_len, angle_mod180))

    if len(kept_segments) < 2:
        return center, center

    # Pairwise intersections of non-parallel segments vote into a smoothed map.
    vote_map = np.zeros((size, size), dtype=np.float32)
    n = len(kept_segments)
    pair_count = 0
    for i in range(n):
        x1, y1, x2, y2, len1, ang1 = kept_segments[i]
        dx1, dy1 = x2 - x1, y2 - y1
        for j in range(i + 1, n):
            x3, y3, x4, y4, len2, ang2 = kept_segments[j]
            ang_diff = abs(ang1 - ang2)
            ang_diff = min(ang_diff, 180.0 - ang_diff)
            if ang_diff < 18.0:
                continue
            dx2, dy2 = x4 - x3, y4 - y3
            denom = dx1 * dy2 - dy1 * dx2
            if abs(denom) < 1e-6:
                continue
            t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / denom
            ix = x1 + t * dx1
            iy = y1 + t * dy1
            dist_c = math.hypot(ix - center, iy - center)
            if dist_c > max_shift:
                continue
            # Prefer intersections close to the endpoints (not far-off extensions).
            d_seg1 = min(math.hypot(ix - x1, iy - y1), math.hypot(ix - x2, iy - y2))
            d_seg2 = min(math.hypot(ix - x3, iy - y3), math.hypot(ix - x4, iy - y4))
            if d_seg1 > work_radius * 0.45 or d_seg2 > work_radius * 0.45:
                continue
            weight = math.sqrt(len1 * len2) * math.sin(math.radians(ang_diff))
            xi, yi = int(round(ix)), int(round(iy))
            if 0 <= xi < size and 0 <= yi < size:
                vote_map[yi, xi] += float(weight)
                pair_count += 1

    if pair_count == 0:
        return center, center

    # Smooth vote map so intersections within a ~6px pivot pin reinforce each other.
    smoothed = cv2.GaussianBlur(vote_map, (17, 17), 4.0)

    # Gentle prior favoring points near the circle center + bonus for foreground
    # pixels near the pivot (hands/hub physically pass over the pivot).
    yy, xx = np.ogrid[:size, :size]
    dist_sq = (xx - center) ** 2 + (yy - center) ** 2
    prior = np.exp(-dist_sq / (2.0 * (work_radius * 0.18) ** 2)).astype(np.float32)

    fg_dilated = cv2.dilate(
        fg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    fg_weight = np.where(fg_dilated > 0, 1.0, 0.35).astype(np.float32)

    score_map = smoothed * prior * fg_weight
    search_mask = dist_sq <= (max_shift**2)
    score_map = np.where(search_mask, score_map, 0.0)

    best_idx = int(np.argmax(score_map))
    best_y, best_x = divmod(best_idx, size)
    best_score = float(score_map[best_y, best_x])
    center_score = float(score_map[int(center), int(center)])

    # Only shift away from the circle center if the intersection peak is clearly
    # stronger than the center itself.
    if best_score <= 0.0 or (best_score < center_score * 1.25 and math.hypot(best_x - center, best_y - center) > 6.0):
        return center, center

    return float(best_x), float(best_y)


def _segment_to_outward_rays(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    pivot_x: float,
    pivot_y: float,
    work_radius: int,
) -> list[tuple[float, float]]:
    """Convert a Hough segment near the pivot into ``(angle_deg, weight)`` ray(s).

    Returns 0, 1, or 2 rays:
    - 0 rays if the line misses the pivot or is too short/far from pivot.
    - 1 ray for a normal hand segment on one side of the pivot.
    - 2 rays if a single segment passes straight through the pivot (e.g. 06:00).
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_len = math.hypot(dx, dy)
    if seg_len < work_radius * 0.11:
        return []

    vx1, vy1 = x1 - pivot_x, y1 - pivot_y
    vx2, vy2 = x2 - pivot_x, y2 - pivot_y
    perp_dist = abs(dx * vy1 - dy * vx1) / seg_len
    if perp_dist > work_radius * 0.11:
        return []

    t_proj = -(vx1 * dx + vy1 * dy) / (seg_len * seg_len)
    r1 = math.hypot(vx1, vy1)
    r2 = math.hypot(vx2, vy2)

    rays: list[tuple[float, float]] = []
    if 0.0 < t_proj < 1.0 and r1 >= work_radius * 0.20 and r2 >= work_radius * 0.20:
        px = x1 + t_proj * dx
        py = y1 + t_proj * dy
        if math.hypot(x1 - px, y1 - py) >= work_radius * 0.14:
            rays.append((point_to_clock_angle(px, py, x1, y1), r1))
        if math.hypot(x2 - px, y2 - py) >= work_radius * 0.14:
            rays.append((point_to_clock_angle(px, py, x2, y2), r2))
        return rays

    if r1 <= r2:
        near_x, near_y, r_near = x1, y1, r1
        far_x, far_y, r_far = x2, y2, r2
    else:
        near_x, near_y, r_near = x2, y2, r2
        far_x, far_y, r_far = x1, y1, r1

    if r_near > work_radius * 0.35 or r_far < work_radius * 0.25:
        return []

    dir_angle = point_to_clock_angle(near_x, near_y, far_x, far_y)
    rad_angle = point_to_clock_angle(pivot_x, pivot_y, far_x, far_y)
    if angular_difference(dir_angle, rad_angle) > 20.0:
        return []

    angle = _circular_mean_deg([dir_angle, rad_angle], [0.85, 0.15])
    rays.append((angle, seg_len))
    return rays


def _polar_ray_proposals(
    fg_mask: np.ndarray,
    contrast_map: np.ndarray,
    pivot_x: float,
    pivot_y: float,
    work_radius: int,
) -> list[tuple[float, float]]:
    """Propose candidate hand angles by scanning radial foreground profiles.

    Complements Hough line detection for thin, slightly soft, or ornate hands
    whose straight edges may fragment under Canny/Hough.
    """
    size = fg_mask.shape[0]
    angles_deg = np.arange(360, dtype=np.float32)
    rads = np.deg2rad(angles_deg)
    # Focus on inner-to-mid radii (0.12..0.62 R) where hands live and numerals
    # / outer tick marks rarely appear.
    radii = np.arange(
        int(work_radius * 0.12), int(work_radius * 0.62) + 1, 2, dtype=np.float32
    )
    xs = np.round(pivot_x + radii[None, :] * np.sin(rads)[:, None]).astype(int)
    ys = np.round(pivot_y - radii[None, :] * np.cos(rads)[:, None]).astype(int)
    xs = np.clip(xs, 0, size - 1)
    ys = np.clip(ys, 0, size - 1)

    fg_polar = (fg_mask[ys, xs] > 0).astype(np.float32)
    contrast_polar = contrast_map[ys, xs].astype(np.float32) / 255.0

    # Measure continuous radial run length from the inner end of each ray.
    n_ang, n_rad = fg_polar.shape
    run_scores = np.zeros(n_ang, dtype=np.float32)
    for a in range(n_ang):
        row = fg_polar[a]
        c_row = contrast_polar[a]
        gap = 0
        hits = 0
        c_sum = 0.0
        for k in range(n_rad):
            if row[k] > 0:
                hits += 1
                c_sum += float(c_row[k])
                gap = 0
            else:
                gap += 1
                if gap > 2:
                    break
        run_scores[a] = hits * 0.6 + c_sum * 1.4

    # Smooth circularly by 3 degrees to merge adjacent pixels of one hand.
    padded = np.concatenate([run_scores[-2:], run_scores, run_scores[:2]])
    smooth = np.convolve(padded, np.array([0.1, 0.2, 0.4, 0.2, 0.1]), mode="valid")

    # Reject broad angular plateaus (shadows / large dark regions > ~24 deg wide).
    bg_padded = np.concatenate([smooth[-15:], smooth, smooth[:15]])
    local_base = np.array(
        [np.percentile(bg_padded[i : i + 31], 35) for i in range(360)],
        dtype=np.float32,
    )
    prominence = smooth - local_base

    min_score = n_rad * 0.22
    proposals: list[tuple[float, float]] = []
    for a in range(360):
        val = float(prominence[a])
        if val < min_score:
            continue
        # Local maximum over +-4 degrees circularly.
        is_peak = True
        for d in range(-4, 5):
            if d == 0:
                continue
            if prominence[(a + d) % 360] > val:
                is_peak = False
                break
        if is_peak:
            proposals.append((float(a), val))

    return proposals


def _cluster_rays(
    rays: Sequence[tuple[float, float]], cluster_tol_deg: float = 5.5
) -> list[tuple[float, int]]:
    """Merge nearby ray angles into circular clusters ``(mean_angle_deg, count)``."""
    if not rays:
        return []

    sorted_rays = sorted(rays, key=lambda item: item[0])
    clusters: list[list[tuple[float, float]]] = [[sorted_rays[0]]]

    for angle, weight in sorted_rays[1:]:
        prev_angles = [a for a, _ in clusters[-1]]
        prev_weights = [w for _, w in clusters[-1]]
        cluster_center = _circular_mean_deg(prev_angles, prev_weights)
        if angular_difference(angle, cluster_center) <= cluster_tol_deg:
            clusters[-1].append((angle, weight))
        else:
            clusters.append([(angle, weight)])

    if len(clusters) > 1:
        first_center = _circular_mean_deg(
            [a for a, _ in clusters[0]], [w for _, w in clusters[0]]
        )
        last_center = _circular_mean_deg(
            [a for a, _ in clusters[-1]], [w for _, w in clusters[-1]]
        )
        if angular_difference(first_center, last_center) <= cluster_tol_deg:
            clusters[0].extend(clusters.pop())

    merged: list[tuple[float, int]] = []
    for group in clusters:
        mean_angle = _circular_mean_deg(
            [a for a, _ in group], [w for _, w in group]
        )
        merged.append((mean_angle, len(group)))
    return merged


def _measure_hand_along_ray(
    angle_deg: float,
    segment_count: int,
    fg_mask: np.ndarray,
    contrast_map: np.ndarray,
    hsv_image: np.ndarray,
    bgr_image: np.ndarray,
    pivot_x: float,
    pivot_y: float,
    work_radius: int,
) -> CandidateHand | None:
    """Measure hand length, width, solidity, contrast, color, and refined angle."""
    size = fg_mask.shape[0]
    dial_center = float(size // 2)
    max_r = int(work_radius * (INNER_MASK_RATIO - 0.01))
    min_r = max(8, int(work_radius * 0.07))
    max_half_width = max(14, int(work_radius * 0.11))

    rad = math.radians(angle_deg)
    ux, uy = math.sin(rad), -math.cos(rad)
    px, py = math.cos(rad), math.sin(rad)

    radii = np.arange(min_r, max_r + 1, dtype=np.float32)
    offsets = np.arange(-max_half_width, max_half_width + 1, dtype=np.float32)

    xs_f = pivot_x + radii[:, None] * ux + offsets[None, :] * px
    ys_f = pivot_y + radii[:, None] * uy + offsets[None, :] * py
    # Stop if the ray exits the inner dial circle around the original dial center.
    ray_cx = pivot_x + radii * ux
    ray_cy = pivot_y + radii * uy
    inside_dial = (
        np.hypot(ray_cx - dial_center, ray_cy - dial_center)
        <= work_radius * (INNER_MASK_RATIO - 0.01)
    )

    xs = np.clip(np.round(xs_f).astype(int), 0, size - 1)
    ys = np.clip(np.round(ys_f).astype(int), 0, size - 1)

    strip = fg_mask[ys, xs] > 0
    mid_idx = max_half_width

    search_window = 4
    present = np.zeros(len(radii), dtype=bool)
    row_widths = np.zeros(len(radii), dtype=np.float32)
    lateral_offsets: list[tuple[float, float]] = []

    for i, r in enumerate(radii):
        if not inside_dial[i]:
            break
        row = strip[i]
        lo = max(0, mid_idx - search_window)
        hi = min(len(offsets), mid_idx + search_window + 1)
        near_hits = np.flatnonzero(row[lo:hi])
        if near_hits.size == 0:
            continue

        seed = lo + int(near_hits[np.argmin(np.abs(lo + near_hits - mid_idx))])
        left = seed
        while left > 0 and row[left - 1]:
            left -= 1
        right = seed
        while right + 1 < len(offsets) and row[right + 1]:
            right += 1

        run_width = float(right - left + 1)
        present[i] = True
        row_widths[i] = run_width
        if run_width < max_half_width * 1.4:
            run_center = 0.5 * (left + right) - mid_idx
            lateral_offsets.append((float(r), run_center))

    # Trace outward from near the pivot with a tight gap limit so the hand does
    # not jump across empty dial space onto an outer numeral or tick mark.
    max_gap = max(4, int(work_radius * 0.025))
    start_limit = int(work_radius * 0.25)

    first_idx = None
    for i, r in enumerate(radii):
        if r > start_limit:
            break
        if present[i]:
            first_idx = i
            break
    if first_idx is None:
        return None

    gap = 0
    tip_idx = first_idx
    seen_widths: list[float] = []
    blob_run = 0
    for i in range(first_idx, len(radii)):
        if not inside_dial[i]:
            break
        if present[i]:
            w = float(row_widths[i])
            # Detect sudden blow-up into a wide bezel/shadow/numeral blob past
            # the inner hub (> 0.35 R).
            if radii[i] > work_radius * 0.35 and len(seen_widths) >= 8:
                base_w = float(np.percentile(seen_widths, 40))
                if w >= max_half_width * 1.85 and w > base_w * 2.6:
                    blob_run += 1
                    if blob_run >= 5:
                        break
                    continue
                blob_run = 0
            seen_widths.append(w)
            tip_idx = i
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break

    tip_r = float(radii[tip_idx])
    if tip_r < work_radius * 0.26:
        return None

    active_slice = present[first_idx : tip_idx + 1]
    solidity = float(np.mean(active_slice))
    if solidity < 0.65:
        return None

    outer_widths = [
        float(row_widths[i])
        for i in range(first_idx, tip_idx + 1)
        if present[i] and 0.40 * tip_r <= radii[i] <= 0.90 * tip_r
    ]
    if not outer_widths:
        outer_widths = [
            float(row_widths[i])
            for i in range(first_idx, tip_idx + 1)
            if present[i]
        ]
    hand_width_px = float(np.percentile(outer_widths, 40))
    norm_width = hand_width_px / float(work_radius)

    # Reject overly wide blobs (shadows / dark dial sectors, not clock hands).
    if norm_width > 0.135:
        return None

    # Refine angle using the foreground centerline offsets along the hand.
    refined_angle = angle_deg
    valid_offsets = [
        (r, off) for r, off in lateral_offsets if 0.22 * tip_r <= r <= 0.95 * tip_r
    ]
    if len(valid_offsets) >= 5:
        r_arr = np.array([r for r, _ in valid_offsets], dtype=np.float64)
        off_arr = np.array([off for _, off in valid_offsets], dtype=np.float64)
        delta_rad = math.atan2(
            float(np.sum(r_arr * off_arr)), float(np.sum(r_arr * r_arr))
        )
        delta_deg = math.degrees(delta_rad)
        if abs(delta_deg) <= 8.0:
            refined_angle = (angle_deg + delta_deg) % 360.0

    # Sample color and contrast properties along the active portion of the hand.
    active_mask = np.zeros_like(strip, dtype=bool)
    half_w = max(1, int(round(hand_width_px / 2.0)))
    c_lo = max(0, mid_idx - half_w)
    c_hi = min(len(offsets), mid_idx + half_w + 1)
    for i in range(first_idx, tip_idx + 1):
        if present[i]:
            active_mask[i, c_lo:c_hi] = strip[i, c_lo:c_hi]

    hand_ys = ys[active_mask]
    hand_xs = xs[active_mask]
    if hand_ys.size > 0:
        mean_sat = float(np.mean(hsv_image[hand_ys, hand_xs, 1]))
        bgr_pixels = bgr_image[hand_ys, hand_xs].astype(np.float32)
        redness = float(
            np.mean(bgr_pixels[:, 2] - np.maximum(bgr_pixels[:, 1], bgr_pixels[:, 0]))
        )
        mean_contrast = float(np.mean(contrast_map[hand_ys, hand_xs]))
    else:
        mean_sat = 0.0
        redness = 0.0
        mean_contrast = 0.0

    return CandidateHand(
        angle_deg=refined_angle,
        tip_radius=tip_r / float(work_radius),
        width=norm_width,
        solidity=solidity,
        mean_saturation=mean_sat,
        redness=redness,
        mean_contrast=mean_contrast,
        segment_count=segment_count,
    )


def _deduplicate_candidates(
    candidates: Sequence[CandidateHand], angle_tol_deg: float = 5.5
) -> list[CandidateHand]:
    """Keep the strongest candidate when two refined rays converge to the same hand."""
    ordered = sorted(
        candidates,
        key=lambda c: (c.tip_radius * c.solidity, c.mean_contrast, c.width),
        reverse=True,
    )
    kept: list[CandidateHand] = []
    for cand in ordered:
        if all(
            angular_difference(cand.angle_deg, k.angle_deg) > angle_tol_deg
            for k in kept
        ):
            kept.append(cand)
    return kept


def _filter_counterweight_tails(
    candidates: Sequence[CandidateHand],
) -> list[CandidateHand]:
    """Remove short counterweight tails lying ~180 deg opposite a much longer hand."""
    kept: list[CandidateHand] = []
    for idx, cand in enumerate(candidates):
        is_tail = False
        if cand.tip_radius < 0.36:
            for other_idx, other in enumerate(candidates):
                if other_idx == idx:
                    continue
                opp_diff = abs(
                    angular_difference(cand.angle_deg, other.angle_deg) - 180.0
                )
                if opp_diff <= 10.0 and other.tip_radius >= cand.tip_radius + 0.22:
                    is_tail = True
                    break
        if not is_tail:
            kept.append(cand)
    return kept


def detect_hand_candidates(
    clock_bgr: np.ndarray, work_radius: int = WORK_RADIUS
) -> list[CandidateHand]:
    """Detect candidate hands on a cropped, normalized BGR clock face."""
    size = clock_bgr.shape[0]
    inner_mask = _create_circle_mask(size, work_radius * INNER_MASK_RATIO)

    contrast_map = compute_contrast_map(clock_bgr, work_radius)
    fg_mask = build_foreground_mask(
        clock_bgr, work_radius, contrast_map=contrast_map
    )

    gray = cv2.cvtColor(clock_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    inner_vals = blurred[inner_mask > 0]
    sigma = float(np.std(inner_vals))
    high_thresh = max(45.0, min(180.0, 0.7 * sigma + 30.0))
    low_thresh = max(15.0, 0.4 * high_thresh)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    edges = cv2.bitwise_and(edges, inner_mask)

    fg_edges = cv2.Canny(fg_mask, 50, 150)
    combined_edges = cv2.bitwise_or(edges, fg_edges)

    lines = cv2.HoughLinesP(
        combined_edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=max(18, int(work_radius * 0.12)),
        minLineLength=max(16, int(work_radius * 0.13)),
        maxLineGap=max(6, int(work_radius * 0.05)),
    )

    # Estimate the true hand pivot (accounts for tilted clocks or thick bezels
    # where the hand pin is shifted from the outer circle center).
    center = float(size // 2)
    pivot_x, pivot_y = estimate_hand_pivot(lines, fg_mask, work_radius)
    hsv = cv2.cvtColor(clock_bgr, cv2.COLOR_BGR2HSV)

    def _extract_from_pivot(px: float, py: float) -> list[CandidateHand]:
        rays: list[tuple[float, float]] = []
        if lines is not None:
            for x1, y1, x2, y2 in np.asarray(lines, dtype=np.float64).reshape(-1, 4):
                rays.extend(
                    _segment_to_outward_rays(x1, y1, x2, y2, px, py, work_radius)
                )
        rays.extend(
            _polar_ray_proposals(fg_mask, contrast_map, px, py, work_radius)
        )
        clusters = _cluster_rays(rays, cluster_tol_deg=5.5)
        found: list[CandidateHand] = []
        for angle_deg, count in clusters:
            measured = _measure_hand_along_ray(
                angle_deg,
                count,
                fg_mask,
                contrast_map,
                hsv,
                clock_bgr,
                px,
                py,
                work_radius,
            )
            if measured is not None:
                found.append(measured)
        found = _deduplicate_candidates(found, angle_tol_deg=6.5)
        found = _filter_counterweight_tails(found)
        return found

    candidates = _extract_from_pivot(pivot_x, pivot_y)
    # If shifting the pivot yielded fewer than 2 hands, also check the nominal
    # circle center and keep whichever pivot finds more hands.
    if len(candidates) < 2 and math.hypot(pivot_x - center, pivot_y - center) > 3.0:
        center_candidates = _extract_from_pivot(center, center)
        if len(center_candidates) > len(candidates):
            candidates = center_candidates

    return candidates


def _is_likely_second_hand(
    cand: CandidateHand, others: Sequence[CandidateHand]
) -> bool:
    """Identify a thin or distinctively colored second hand when >= 3 hands exist."""
    if len(others) < 2:
        return False

    # Use substantial, realistic-width hands as the reference pair so short
    # stubs or wide shadow blobs cannot make a real minute hand look thin.
    ref_pool = [
        o for o in others if o.tip_radius >= 0.42 and o.width <= 0.085
    ]
    if len(ref_pool) < 2:
        return False
    other_widths = sorted((o.width for o in ref_pool), reverse=True)
    ref_width = other_widths[1]
    other_sats = [o.mean_saturation for o in ref_pool]
    median_other_sat = float(np.median(other_sats))

    is_vividly_colored = (
        cand.mean_saturation >= 100.0
        and cand.mean_saturation > median_other_sat + 55.0
    ) or (cand.redness >= 38.0 and all(o.redness < 15.0 for o in ref_pool))

    if is_vividly_colored and cand.width <= ref_width * 1.05:
        return True

    is_much_thinner = cand.width <= 0.017 and cand.width < 0.55 * ref_width
    if is_much_thinner:
        return True
    return False


def _overlap_consistency_error(angle_deg: float) -> float:
    """Return angular error (deg) if both hands were overlapping at ``angle_deg``."""
    return hour_minute_consistency_error(angle_deg, angle_deg)


def _pair_score(
    hour_cand: CandidateHand,
    minute_cand: CandidateHand,
    all_candidates: Sequence[CandidateHand],
) -> float:
    """Score an ordered ``(hour_hand, minute_hand)`` hypothesis.

    Combines:
    - Hand quality (solidity, contrast, length, realistic hand width).
    - Smooth relative length preference (minute hand longer than hour hand).
    - Gentle relative width preference (hour hand typically thicker).
    - Mechanical geometric consistency: ``(hour_angle % 30)`` vs ``minute_angle / 12``.
    """
    max_contrast = max((c.mean_contrast for c in all_candidates), default=1.0)
    max_contrast = max(max_contrast, 1.0)

    def hand_quality(c: CandidateHand) -> float:
        contrast_norm = c.mean_contrast / max_contrast
        width_bonus = min(c.width, 0.055) * 9.0
        if c.width > 0.085:
            width_bonus -= (c.width - 0.085) * 18.0
        return (
            1.2 * c.solidity
            + 0.9 * contrast_norm
            + width_bonus
            + 1.15 * min(c.tip_radius, 0.85)
        )

    score = hand_quality(hour_cand) + hand_quality(minute_cand)

    # Smooth continuous preference for minute hand being longer than hour hand.
    len_diff = minute_cand.tip_radius - hour_cand.tip_radius
    score += 1.35 * math.tanh(len_diff / 0.10)

    # Penalize unrealistically short stubs for either hand.
    if hour_cand.tip_radius < 0.36:
        score -= (0.36 - hour_cand.tip_radius) * 9.0
    if minute_cand.tip_radius < 0.45:
        score -= (0.45 - minute_cand.tip_radius) * 9.0

    # Gentle smooth preference for hour hand being at least as thick as minute hand.
    width_diff = hour_cand.width - minute_cand.width
    score += 0.45 * math.tanh(width_diff / 0.025)

    # Moderate mechanical geometric consistency (0..15 deg): breaks ties and
    # rejects impossible pairs without overriding clear length cues on tilted dials.
    geo_err = hour_minute_consistency_error(
        hour_cand.angle_deg, minute_cand.angle_deg
    )
    score -= (geo_err / 15.0) * 1.15

    return score


def select_hour_and_minute_hands(
    candidates: Sequence[CandidateHand],
) -> tuple[CandidateHand, CandidateHand] | None:
    """Choose ``(hour_hand, minute_hand)`` from detected candidate hands.

    Handles:
    - 0 candidates -> returns ``None``.
    - 1 candidate -> accepts as overlapping hands only if the hand is long enough
      and its angle is geometrically consistent with an hour/minute overlap.
    - 3+ candidates -> removes likely second hands (very thin or vividly colored),
      then scores all ordered ``(hour, minute)`` pairs jointly using length,
      thickness, contrast, and mechanical hour/minute angle consistency.
    - 2 candidates -> scores both role assignments jointly using length,
      thickness, and mechanical geometric consistency.
    """
    if not candidates:
        return None

    pool = list(candidates)

    # Filter out likely second hands when 3 or more hands are present.
    while len(pool) >= 3:
        second_hand_idx = None
        best_second_score = -1.0
        for idx, cand in enumerate(pool):
            others = [c for j, c in enumerate(pool) if j != idx]
            if _is_likely_second_hand(cand, others):
                score = (
                    cand.mean_saturation / 255.0
                    + max(0.0, cand.redness) / 100.0
                    - cand.width * 10.0
                )
                if score > best_second_score:
                    best_second_score = score
                    second_hand_idx = idx
        if second_hand_idx is None:
            break
        pool.pop(second_hand_idx)

    # Keep the top candidates by quality if a noisy clock produced many rays.
    if len(pool) > 6:
        pool.sort(
            key=lambda c: (c.width * 2.0 + c.tip_radius) * c.solidity,
            reverse=True,
        )
        pool = pool[:6]

    if len(pool) == 1:
        sole = pool[0]
        if sole.tip_radius >= 0.48 and _overlap_consistency_error(sole.angle_deg) <= 8.0:
            return sole, sole
        return None

    best_pair: tuple[CandidateHand, CandidateHand] | None = None
    best_score = -1e9
    for i, hour_cand in enumerate(pool):
        for j, minute_cand in enumerate(pool):
            if i == j:
                continue
            s = _pair_score(hour_cand, minute_cand, pool)
            if s > best_score:
                best_score = s
                best_pair = (hour_cand, minute_cand)

    return best_pair


def read_time(image: np.ndarray, circle: Sequence[float] | None) -> str | None:
    """Read the time shown on an analogue clock face.

    Args:
        image: OpenCV BGR image (original resolution).
        circle: ``(cx, cy, radius)`` in original image coordinates (matching
            ``src.localization.locate_clock``), or ``None``.

    Returns:
        Predicted time formatted as ``"HH:MM"`` (hours ``01``–``12``), or
        ``None`` when hand detection fails.
    """
    validated = _validate_inputs(image, circle)
    if validated is None:
        return None

    clock_bgr = extract_clock_face(image, validated, work_radius=WORK_RADIUS)
    candidates = detect_hand_candidates(clock_bgr, work_radius=WORK_RADIUS)
    selected = select_hour_and_minute_hands(candidates)
    if selected is None:
        return None

    hour_hand, minute_hand = selected
    return angles_to_time(hour_hand.angle_deg, minute_hand.angle_deg)
