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
    image: np.ndarray, circle: Sequence[float]
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
    # Ensure the circle center lies inside the image bounds.
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
    so the clock center always lands cleanly at ``(work_radius, work_radius)``.
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
    # Affine transform maps original (cx, cy) -> (work_radius, work_radius).
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


def _create_circle_mask(size: int, radius: float) -> np.ndarray:
    center = size // 2
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (center, center), int(round(radius)), 255, thickness=-1)
    return mask


def build_foreground_mask(
    clock_bgr: np.ndarray, work_radius: int = WORK_RADIUS
) -> np.ndarray:
    """Build a polarity-invariant binary mask of hand/marking foreground pixels.

    Hands stand out from the dial face by intensity contrast (either darker than
    a light dial or lighter than a dark dial) or by color saturation (e.g. a red
    hand on a gray dial). Measuring contrast against a smoothed background model
    handles uneven illumination as well as both light and dark dials.
    """
    size = clock_bgr.shape[0]
    inner_mask = _create_circle_mask(size, work_radius * INNER_MASK_RATIO)
    gray = cv2.cvtColor(clock_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Estimate local dial background with a large median/Gaussian filter so
    # narrow hands are smoothed away while gradual shading is preserved.
    bg_kernel = max(31, int(work_radius * 0.35) | 1)
    local_bg = cv2.GaussianBlur(blurred, (bg_kernel, bg_kernel), 0)
    local_diff = cv2.absdiff(blurred, local_bg)

    # Also compute contrast relative to the global median of the inner dial.
    dial_pixels = blurred[inner_mask > 0]
    dial_median = int(np.median(dial_pixels))
    global_diff = cv2.absdiff(
        blurred, np.full_like(blurred, dial_median, dtype=np.uint8)
    )

    contrast = np.maximum(local_diff, global_diff)
    masked_values = contrast[inner_mask > 0]

    # Compute an Otsu threshold on contrast values inside the clock face,
    # floored to prevent noise on a blank dial from being treated as hands.
    otsu_thresh, _ = cv2.threshold(
        masked_values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    thresh = max(float(otsu_thresh), 22.0)

    fg_mask = np.zeros((size, size), dtype=np.uint8)
    fg_mask[(contrast >= thresh) & (inner_mask > 0)] = 255

    # Clean up isolated 1px speckles without erasing thin hands.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    return fg_mask


def _segment_to_outward_rays(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    center: float,
    work_radius: int,
) -> list[tuple[float, float]]:
    """Convert a Hough segment near the center into ``(angle_deg, weight)`` ray(s).

    Returns 0, 1, or 2 rays:
    - 0 rays if the line misses the center or is too short/far from center.
    - 1 ray for a normal hand segment on one side of the center.
    - 2 rays if a single segment passes straight through the center (e.g. 06:00).
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_len = math.hypot(dx, dy)
    if seg_len < work_radius * 0.12:
        return []

    # Perpendicular distance from clock center to the infinite line of the segment.
    vx1, vy1 = x1 - center, y1 - center
    vx2, vy2 = x2 - center, y2 - center
    perp_dist = abs(dx * vy1 - dy * vx1) / seg_len
    if perp_dist > work_radius * 0.12:
        return []

    # Project center onto the segment parameter t in [0, 1].
    t_proj = -(vx1 * dx + vy1 * dy) / (seg_len * seg_len)
    r1 = math.hypot(vx1, vy1)
    r2 = math.hypot(vx2, vy2)

    rays: list[tuple[float, float]] = []
    # Case 1: Segment spans across the center (both endpoints extend outward).
    if 0.0 < t_proj < 1.0 and r1 >= work_radius * 0.22 and r2 >= work_radius * 0.22:
        px = x1 + t_proj * dx
        py = y1 + t_proj * dy
        if math.hypot(x1 - px, y1 - py) >= work_radius * 0.15:
            rays.append((point_to_clock_angle(px, py, x1, y1), r1))
        if math.hypot(x2 - px, y2 - py) >= work_radius * 0.15:
            rays.append((point_to_clock_angle(px, py, x2, y2), r2))
        return rays

    # Case 2: Segment lies predominantly on one side of the center.
    if r1 <= r2:
        near_x, near_y, r_near = x1, y1, r1
        far_x, far_y, r_far = x2, y2, r2
    else:
        near_x, near_y, r_near = x2, y2, r2
        far_x, far_y, r_far = x1, y1, r1

    # Reject segments that start too far from center (e.g. outer tick marks or
    # numerals) or do not extend far enough to be a clock hand.
    if r_near > work_radius * 0.35 or r_far < work_radius * 0.28:
        return []

    # Orient the segment direction outward from near_pt to far_pt.
    # Using segment direction keeps left/right edges of a parallel-sided hand
    # closely aligned while a small radial blend stabilizes tapered hands.
    dir_angle = point_to_clock_angle(near_x, near_y, far_x, far_y)
    rad_angle = point_to_clock_angle(center, center, far_x, far_y)
    if angular_difference(dir_angle, rad_angle) > 20.0:
        return []

    angle = _circular_mean_deg([dir_angle, rad_angle], [0.85, 0.15])
    rays.append((angle, seg_len))
    return rays


def _cluster_rays(
    rays: Sequence[tuple[float, float]], cluster_tol_deg: float = 9.0
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

    # Check wrap-around between the last cluster (near 360 deg) and first (near 0 deg).
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
    hsv_image: np.ndarray,
    bgr_image: np.ndarray,
    work_radius: int,
) -> CandidateHand | None:
    """Measure hand length, width, solidity, color, and refined angle along a ray."""
    size = fg_mask.shape[0]
    center = float(size // 2)
    max_r = int(work_radius * (INNER_MASK_RATIO - 0.01))
    min_r = max(8, int(work_radius * 0.08))
    max_half_width = max(12, int(work_radius * 0.10))

    rad = math.radians(angle_deg)
    # Unit vector along the ray (clockwise from 12 o'clock, y downward).
    ux, uy = math.sin(rad), -math.cos(rad)
    # Unit vector perpendicular to the ray (pointing clockwise).
    px, py = math.cos(rad), math.sin(rad)

    radii = np.arange(min_r, max_r + 1, dtype=np.float32)
    offsets = np.arange(-max_half_width, max_half_width + 1, dtype=np.float32)

    # Sample coordinates for the rectangular strip along the ray: shape (R, W).
    xs = np.round(center + radii[:, None] * ux + offsets[None, :] * px).astype(int)
    ys = np.round(center + radii[:, None] * uy + offsets[None, :] * py).astype(int)
    xs = np.clip(xs, 0, size - 1)
    ys = np.clip(ys, 0, size - 1)

    strip = fg_mask[ys, xs] > 0  # shape (R, W)
    mid_idx = max_half_width

    # For each radius, check whether a foreground run passes near the ray center.
    search_window = 4
    present = np.zeros(len(radii), dtype=bool)
    widths: list[tuple[float, float]] = []  # (radius, width_px)
    lateral_offsets: list[tuple[float, float]] = []  # (radius, offset_px)

    for i, r in enumerate(radii):
        row = strip[i]
        lo = max(0, mid_idx - search_window)
        hi = min(len(offsets), mid_idx + search_window + 1)
        near_hits = np.flatnonzero(row[lo:hi])
        if near_hits.size == 0:
            continue

        # Pick the foreground pixel closest to the centerline and expand to the
        # contiguous foreground run containing it.
        seed = lo + int(near_hits[np.argmin(np.abs(lo + near_hits - mid_idx))])
        left = seed
        while left > 0 and row[left - 1]:
            left -= 1
        right = seed
        while right + 1 < len(offsets) and row[right + 1]:
            right += 1

        run_width = float(right - left + 1)
        present[i] = True
        widths.append((float(r), run_width))
        # Only use clean, non-merged cross-sections for centerline refinement.
        if run_width < max_half_width * 1.4:
            run_center = 0.5 * (left + right) - mid_idx
            lateral_offsets.append((float(r), run_center))

    # Trace outward from near the center, allowing small gaps (e.g. thin gap or
    # highlight across the hand) to find the hand tip.
    max_gap = max(6, int(work_radius * 0.05))
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
    for i in range(first_idx, len(radii)):
        if present[i]:
            tip_idx = i
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break

    tip_r = float(radii[tip_idx])
    if tip_r < work_radius * 0.28:
        return None

    active_slice = present[first_idx : tip_idx + 1]
    solidity = float(np.mean(active_slice))
    if solidity < 0.65:
        return None

    # Measure width in the outer half of the hand (away from the center hub and
    # away from where hands overlap near the center).
    outer_widths = [
        w for r, w in widths if 0.45 * tip_r <= r <= 0.90 * tip_r
    ]
    if not outer_widths:
        outer_widths = [w for r, w in widths if r <= tip_r]
    # Use 40th percentile width to resist localized crossings or decorative tips.
    hand_width_px = float(np.percentile(outer_widths, 40))

    # Refine angle using the foreground centerline offsets along the hand.
    refined_angle = angle_deg
    valid_offsets = [
        (r, off) for r, off in lateral_offsets if 0.25 * tip_r <= r <= 0.95 * tip_r
    ]
    if len(valid_offsets) >= 5:
        r_arr = np.array([r for r, _ in valid_offsets], dtype=np.float64)
        off_arr = np.array([off for _, off in valid_offsets], dtype=np.float64)
        # Small-angle correction: tan(delta) ~= sum(r * off) / sum(r^2).
        delta_rad = math.atan2(float(np.sum(r_arr * off_arr)), float(np.sum(r_arr * r_arr)))
        delta_deg = math.degrees(delta_rad)
        if abs(delta_deg) <= 8.0:
            refined_angle = (angle_deg + delta_deg) % 360.0

    # Sample color properties (saturation and redness) on foreground pixels along
    # the active portion of the hand to help identify colored second hands.
    active_mask = np.zeros_like(strip, dtype=bool)
    for i in range(first_idx, tip_idx + 1):
        if present[i]:
            half_w = max(1, int(round(hand_width_px / 2.0)))
            c_lo = max(0, mid_idx - half_w)
            c_hi = min(len(offsets), mid_idx + half_w + 1)
            active_mask[i, c_lo:c_hi] = strip[i, c_lo:c_hi]

    hand_ys = ys[active_mask]
    hand_xs = xs[active_mask]
    if hand_ys.size > 0:
        mean_sat = float(np.mean(hsv_image[hand_ys, hand_xs, 1]))
        bgr_pixels = bgr_image[hand_ys, hand_xs].astype(np.float32)
        redness = float(
            np.mean(bgr_pixels[:, 2] - np.maximum(bgr_pixels[:, 1], bgr_pixels[:, 0]))
        )
    else:
        mean_sat = 0.0
        redness = 0.0

    return CandidateHand(
        angle_deg=refined_angle,
        tip_radius=tip_r / float(work_radius),
        width=hand_width_px / float(work_radius),
        solidity=solidity,
        mean_saturation=mean_sat,
        redness=redness,
        segment_count=segment_count,
    )


def _deduplicate_candidates(
    candidates: Sequence[CandidateHand], angle_tol_deg: float = 8.0
) -> list[CandidateHand]:
    """Keep the strongest candidate when two refined rays converge to the same hand."""
    ordered = sorted(
        candidates,
        key=lambda c: (c.tip_radius * c.solidity, c.width),
        reverse=True,
    )
    kept: list[CandidateHand] = []
    for cand in ordered:
        if all(angular_difference(cand.angle_deg, k.angle_deg) > angle_tol_deg for k in kept):
            kept.append(cand)
    return kept


def _filter_counterweight_tails(
    candidates: Sequence[CandidateHand],
) -> list[CandidateHand]:
    """Remove short counterweight tails lying ~180 deg opposite a much longer hand."""
    kept: list[CandidateHand] = []
    for idx, cand in enumerate(candidates):
        is_tail = False
        if cand.tip_radius < 0.34:
            for other_idx, other in enumerate(candidates):
                if other_idx == idx:
                    continue
                opp_diff = abs(angular_difference(cand.angle_deg, other.angle_deg) - 180.0)
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
    center = float(size // 2)
    inner_mask = _create_circle_mask(size, work_radius * INNER_MASK_RATIO)

    gray = cv2.cvtColor(clock_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Canny edges inside the inner dial (excluding the circular bezel).
    inner_vals = blurred[inner_mask > 0]
    sigma = float(np.std(inner_vals))
    high_thresh = max(45.0, min(180.0, 0.7 * sigma + 30.0))
    low_thresh = max(15.0, 0.4 * high_thresh)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    edges = cv2.bitwise_and(edges, inner_mask)

    # Also include edges of the polarity-invariant foreground mask so low-contrast
    # or colored hands still generate clean straight line segments.
    fg_mask = build_foreground_mask(clock_bgr, work_radius)
    fg_edges = cv2.Canny(fg_mask, 50, 150)
    combined_edges = cv2.bitwise_or(edges, fg_edges)

    lines = cv2.HoughLinesP(
        combined_edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=max(20, int(work_radius * 0.14)),
        minLineLength=max(18, int(work_radius * 0.15)),
        maxLineGap=max(8, int(work_radius * 0.06)),
    )

    rays: list[tuple[float, float]] = []
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines, dtype=np.float64).reshape(-1, 4):
            rays.extend(_segment_to_outward_rays(x1, y1, x2, y2, center, work_radius))

    clusters = _cluster_rays(rays, cluster_tol_deg=5.5)
    hsv = cv2.cvtColor(clock_bgr, cv2.COLOR_BGR2HSV)

    candidates: list[CandidateHand] = []
    for angle_deg, count in clusters:
        measured = _measure_hand_along_ray(
            angle_deg, count, fg_mask, hsv, clock_bgr, work_radius
        )
        if measured is not None:
            candidates.append(measured)

    candidates = _deduplicate_candidates(candidates, angle_tol_deg=5.5)
    candidates = _filter_counterweight_tails(candidates)
    return candidates


def _is_likely_second_hand(
    cand: CandidateHand, others: Sequence[CandidateHand]
) -> bool:
    """Identify a thin or distinctively colored second hand when >= 3 hands exist."""
    if len(others) < 2:
        return False

    other_widths = sorted((o.width for o in others), reverse=True)
    ref_width = other_widths[1]  # Width of the second-thickest other hand
    other_sats = [o.mean_saturation for o in others]
    median_other_sat = float(np.median(other_sats))

    is_much_thinner = cand.width < 0.62 * ref_width
    is_vividly_colored = (
        cand.mean_saturation >= 85.0
        and cand.mean_saturation > median_other_sat + 45.0
    ) or (cand.redness >= 40.0 and all(o.redness < 15.0 for o in others))

    if is_vividly_colored and cand.width <= ref_width * 1.05:
        return True
    if is_much_thinner:
        return True
    return False


def _overlap_consistency_error(angle_deg: float) -> float:
    """Return angular error (deg) if both hands were overlapping at ``angle_deg``."""
    minute = int(round((angle_deg % 360.0) / 6.0)) % 60
    return min(
        angular_difference(angle_deg, ((h + minute / 60.0) * 30.0) % 360.0)
        for h in range(12)
    )


def select_hour_and_minute_hands(
    candidates: Sequence[CandidateHand],
) -> tuple[CandidateHand, CandidateHand] | None:
    """Choose ``(hour_hand, minute_hand)`` from detected candidate hands.

    Handles:
    - 0 candidates -> returns ``None``.
    - 1 candidate -> accepts as overlapping hands only if the hand is long enough
      and its angle is geometrically consistent with an hour/minute overlap.
    - 3+ candidates -> removes likely second hands (very thin or vividly colored)
      and ranks remaining hands by thickness, length, and solidity.
    - 2 candidates -> distinguishes minute hand (longer, or thinner on tie) from
      hour hand (shorter, or thicker on tie).
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
                # Prefer removing the thinnest / most saturated second hand first.
                score = cand.mean_saturation / 255.0 + max(0.0, cand.redness) / 100.0 - cand.width * 10.0
                if score > best_second_score:
                    best_second_score = score
                    second_hand_idx = idx
        if second_hand_idx is None:
            break
        pool.pop(second_hand_idx)

    # If more than 2 candidates still remain, keep the two strongest hands.
    if len(pool) > 2:
        pool.sort(
            key=lambda c: (c.width * 2.0 + c.tip_radius) * c.solidity,
            reverse=True,
        )
        pool = pool[:2]

    # Single detected hand: accept only if consistent with overlapping hands.
    if len(pool) == 1:
        sole = pool[0]
        if sole.tip_radius >= 0.50 and _overlap_consistency_error(sole.angle_deg) <= 8.0:
            return sole, sole
        return None

    hand_a, hand_b = pool[0], pool[1]
    # Primary cue: minute hand is longer than hour hand.
    if abs(hand_a.tip_radius - hand_b.tip_radius) >= 0.04:
        if hand_a.tip_radius > hand_b.tip_radius:
            return hand_b, hand_a
        return hand_a, hand_b

    # Tie-breaker when lengths are very close: hour hand is typically thicker.
    if hand_a.width >= hand_b.width:
        return hand_a, hand_b
    return hand_b, hand_a


def read_time(image: np.ndarray, circle: Sequence[float]) -> str | None:
    """Read the time shown on an analogue clock face.

    Args:
        image: OpenCV BGR image (original resolution).
        circle: ``(cx, cy, radius)`` in original image coordinates.

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
