"""Automatic clock face localization.

locate_clock(image) finds the round clock face in a BGR image and returns
(cx, cy, radius) in the ORIGINAL image's pixel coordinates, or None.

Procedure (identical for every image, no per-file settings):
1. Shrink the image so its longest side is at most work_size pixels.
   Detection runs on this smaller copy, which makes it faster and makes the
   pixel-based parameters below mean the same thing for every image.
2. Convert to grayscale and smooth with a median blur to suppress texture
   and noise while keeping the clock rim sharp.
3. Run the Hough circle transform to get candidate circles.
4. Score every candidate by its "radial edge support": the fraction of points
   on the circle that have a Canny edge nearby whose gradient points along
   the radius (towards or away from the centre). A clock rim gives radial
   edges all the way round; background texture gives edges in random
   directions. The rate expected by chance (from how many edge pixels the
   image has) is subtracted, so busy images don't get high scores for free.
5. Among candidates scoring close to the best, take the largest. This picks
   the outer rim rather than an inner ring of the dial (both are circles
   with the same centre). Reject the result if its score is too low.
6. Fallback for weak boundaries: if step 5 found nothing, or its score is
   below weak_score, steps 3-5 are repeated on a contrast-enhanced copy
   (CLAHE, i.e. histogram equalisation in local tiles) with more permissive
   Hough and Canny thresholds. This brings out faint or low-contrast rims.
   The result with the higher score is kept. The trigger is the same fixed
   score rule for every image.
7. Divide the coordinates by the resize scale to return to original pixels.
"""

import cv2
import numpy as np


# All tunable parameters live here. Pixel values refer to the resized image.
PARAMS = {
    # Longest image side used for detection. Larger images are shrunk to this.
    "work_size": 512,
    # Median blur kernel (odd). Removes fine texture so the rim dominates.
    # 1 or less switches the blur off.
    "blur_kernel": 5,
    # Hough accumulator resolution: 1 = same resolution as the image.
    "hough_dp": 1.2,
    # Upper Canny threshold, used by HoughCircles and for scoring
    # (the lower Canny threshold is half of this).
    "canny_high": 100,
    # Accumulator votes needed for a Hough candidate. Lower = more candidates.
    "hough_votes": 30,
    # Allowed radius range as a fraction of the shorter image side.
    # Some clocks are small in the photo (clock towers, a watch on a table).
    "min_radius_frac": 0.05,
    "max_radius_frac": 0.60,
    # Maximum number of Hough candidates that are scored.
    "max_candidates": 60,
    # Points sampled on a candidate circle when scoring it.
    "support_samples": 360,
    # How far (pixels, along the radius) to look for an edge from each point.
    "support_tolerance": 2,
    # An edge counts as radial if |cos(angle to radius)| >= this (~37 deg).
    "radial_cos_min": 0.8,
    # Candidates scoring at least this fraction of the best score compete,
    # and the largest of them wins.
    "prefer_larger_ratio": 0.8,
    # Minimum score (support minus chance level) to accept a circle.
    "min_score": 0.25,
    # Fallback (step 6). Results scoring below weak_score trigger it; on the
    # training images 7 of the 8 detections scoring <= 0.40 were wrong.
    "weak_score": 0.45,
    # CLAHE contrast limit and tile grid (tiles per image side).
    "fallback_clahe_clip": 2.0,
    "fallback_clahe_tiles": 8,
    # More permissive thresholds than the first pass, because the rim edges
    # are weaker. Chance correction in the score still applies.
    "fallback_hough_votes": 20,
    "fallback_canny_high": 60,
}


def resize_for_detection(image, work_size=PARAMS["work_size"]):
    """Shrink image so its longest side is at most work_size.

    Returns (resized_image, scale) where resized = original * scale.
    Images that are already small enough are not enlarged (scale = 1.0).
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= work_size:
        return image, 1.0
    scale = work_size / longest
    new_size = (round(width * scale), round(height * scale))  # cv2 wants (w, h)
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def to_original_coordinates(cx, cy, radius, scale):
    """Map a circle found on the resized image back to original pixels."""
    return cx / scale, cy / scale, radius / scale


def to_gray(image):
    """Return a single-channel uint8 image (accepts BGR or grayscale)."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def radial_support(edges, gx, gy, cx, cy, radius, params=PARAMS):
    """Fraction of points on the circle with a radial edge close by.

    For each sampled angle we look at the pixels from radius - tolerance to
    radius + tolerance along that direction. The point is supported if one
    of them is a Canny edge whose gradient is roughly parallel to the radius.
    Points outside the image count as unsupported.
    """
    height, width = edges.shape
    angles = np.linspace(0, 2 * np.pi, params["support_samples"], endpoint=False)
    ux, uy = np.cos(angles), np.sin(angles)
    supported = np.zeros(len(angles), dtype=bool)

    tol = params["support_tolerance"]
    for offset in range(-tol, tol + 1):
        xs = np.round(cx + (radius + offset) * ux).astype(int)
        ys = np.round(cy + (radius + offset) * uy).astype(int)
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        x, y = xs[inside], ys[inside]
        gxs, gys = gx[y, x], gy[y, x]
        magnitude = np.hypot(gxs, gys) + 1e-9
        cos_angle = np.abs(gxs * ux[inside] + gys * uy[inside]) / magnitude
        radial_edge = (edges[y, x] > 0) & (cos_angle >= params["radial_cos_min"])
        supported[inside] |= radial_edge
    return float(supported.mean())


def chance_support(edges, params=PARAMS):
    """Support a random circle would get if edges had random directions.

    p = fraction of edge pixels; a random direction is radial with
    probability cone (the share of angles with |cos| >= radial_cos_min);
    each sample checks 2 * tolerance + 1 pixels.
    """
    p = float((edges > 0).mean())
    cone = 4 * np.arccos(params["radial_cos_min"]) / (2 * np.pi)
    checked = 2 * params["support_tolerance"] + 1
    return 1 - (1 - p * cone) ** checked


def detect_circle(gray, params=PARAMS):
    """Find the clock circle in an (already resized) grayscale image.

    Returns (cx, cy, radius, score) in this image's coordinates, or None.
    """
    if params["blur_kernel"] > 1:
        blurred = cv2.medianBlur(gray, params["blur_kernel"])
    else:  # no smoothing; used when comparing preprocessing options
        blurred = gray
    short_side = min(gray.shape)
    min_r = max(1, int(params["min_radius_frac"] * short_side))
    max_r = int(params["max_radius_frac"] * short_side)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=params["hough_dp"],
        minDist=min_r,  # candidate centres closer than this are merged
        param1=params["canny_high"],
        param2=params["hough_votes"],
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None

    edges = cv2.Canny(blurred, params["canny_high"] // 2, params["canny_high"])
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1)
    chance = chance_support(edges, params)

    scored = []
    for cx, cy, r in circles[0][: params["max_candidates"]]:
        score = radial_support(edges, gx, gy, cx, cy, r, params) - chance
        scored.append((score, float(cx), float(cy), float(r)))

    best_score = max(s[0] for s in scored)
    if best_score < params["min_score"]:
        return None
    competitors = [s for s in scored if s[0] >= params["prefer_larger_ratio"] * best_score]
    score, cx, cy, r = max(competitors, key=lambda s: s[3])
    return cx, cy, r, score


def enhance_contrast(gray, params=PARAMS):
    """Local histogram equalisation (CLAHE) to strengthen faint edges."""
    tiles = params["fallback_clahe_tiles"]
    clahe = cv2.createCLAHE(clipLimit=params["fallback_clahe_clip"], tileGridSize=(tiles, tiles))
    return clahe.apply(gray)


def find_circle(gray, params=PARAMS):
    """First pass plus the weak-boundary fallback (steps 3-6).

    Returns (cx, cy, radius, score, method) in this image's coordinates,
    where method is "primary" or "fallback", or None.
    """
    primary = detect_circle(gray, params)
    if primary is not None and primary[3] >= params["weak_score"]:
        return (*primary, "primary")

    fallback_params = dict(params, hough_votes=params["fallback_hough_votes"],
                           canny_high=params["fallback_canny_high"])
    fallback = detect_circle(enhance_contrast(gray, params), fallback_params)

    if fallback is None and primary is None:
        return None
    if primary is None or (fallback is not None and fallback[3] > primary[3]):
        return (*fallback, "fallback")
    return (*primary, "primary")


def locate_clock_details(image, params=PARAMS):
    """Like locate_clock, but also returns diagnostic values.

    Returns a dict with cx, cy, radius (original pixels), score (radial
    support above chance, roughly 0..1), method ("primary" or "fallback")
    and scale (resized / original), or None if no convincing circle is found.
    """
    if image is None or image.size == 0:
        raise ValueError("locate_clock needs a non-empty image")

    small, scale = resize_for_detection(image, params["work_size"])
    found = find_circle(to_gray(small), params)
    if found is None:
        return None
    cx, cy, radius, score, method = found
    cx, cy, radius = to_original_coordinates(cx, cy, radius, scale)
    return {"cx": cx, "cy": cy, "radius": radius, "score": score,
            "method": method, "scale": scale}


def locate_clock(image, params=PARAMS):
    """Locate the clock face in a BGR image.

    Returns (cx, cy, radius) as floats in the original image's pixel
    coordinates, or None if no convincing circle is found.
    Raises ValueError if image is None or empty (e.g. cv2.imread failed).
    """
    details = locate_clock_details(image, params)
    if details is None:
        return None
    return details["cx"], details["cy"], details["radius"]
