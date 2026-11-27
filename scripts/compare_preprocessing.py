"""Compare preprocessing options for clock localization on training images.

Usage:
    .venv/bin/python scripts/compare_preprocessing.py [--limit N]

Every variant runs the same circle detector (Hough candidates + radial edge
score, src/localization.py) and only changes what happens before it:
colour channel, smoothing filter, or working resolution. Variants run the
first pass only (no fallback), so each preprocessing step is judged on its
own. The full current method (first pass + CLAHE fallback) is included as a
reference row, and so is the current method with a stronger median blur.

There are no ground-truth clock positions, so the measures are:
- found:     images where a circle is returned
- weak:      found circles with score below PARAMS["weak_score"]
- half/pad:  stability under shrinking to half size / adding a border,
             same test as scripts/check_localization_stability.py
             (centre error <= 5% and radius error <= 10% of the radius)
- ms/image:  resizing + detection time per image (excludes file loading)
- differ:    images where the circle differs from the current method; these
             are written as side-by-side sheets for visual inspection.

Results go to outputs/localization/preprocessing/.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "scripts"))

from src.localization import (  # noqa: E402
    PARAMS,
    detect_circle,
    find_circle,
    resize_for_detection,
    to_gray,
)
from check_localization_stability import (  # noqa: E402
    CENTRE_OK,
    PAD_FRACTION,
    RADIUS_OK,
    relative_errors,
)
from show_localization import draw_overlay, thumbnail  # noqa: E402

data_dir = root / "data" / "raw"
out_dir = root / "outputs" / "localization" / "preprocessing"
REFERENCE = "current method"


def gray_no_blur(small):
    return to_gray(small)


def gaussian(small):
    return cv2.GaussianBlur(to_gray(small), (5, 5), 0)


def bilateral(small):
    # Edge-preserving smoothing: averages similar neighbouring intensities.
    return cv2.bilateralFilter(to_gray(small), 9, 50, 50)


def clahe(small):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(to_gray(small))


def saturation(small):
    return cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]


def hsv_value(small):
    # V = max(B, G, R): keeps coloured rims bright where gray can lose them.
    return cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 2]


# name: (channel function, PARAMS overrides, use full method with fallback)
VARIANTS = {
    REFERENCE: (to_gray, {}, True),
    "gray + median 5 (first pass)": (to_gray, {}, False),
    "gray, no blur": (gray_no_blur, {"blur_kernel": 1}, False),
    "gray + median 9": (to_gray, {"blur_kernel": 9}, False),
    "gray + median 9 + fallback": (to_gray, {"blur_kernel": 9}, True),
    "gray + gaussian 5": (gaussian, {"blur_kernel": 1}, False),
    "gray + bilateral": (bilateral, {"blur_kernel": 1}, False),
    "CLAHE + median 5": (clahe, {}, False),
    "HSV saturation + median 5": (saturation, {}, False),
    "HSV value + median 5": (hsv_value, {}, False),
    "work size 384": (to_gray, {"work_size": 384}, False),
    "work size 768": (to_gray, {"work_size": 768}, False),
}


def run_variant(image, variant):
    """Detect on image with one variant; circle in image pixels or None."""
    channel_fn, overrides, full = VARIANTS[variant]
    params = dict(PARAMS, **overrides)
    small, scale = resize_for_detection(image, params["work_size"])
    channel = channel_fn(small)
    found = find_circle(channel, params) if full else detect_circle(channel, params)
    if found is None:
        return None, None
    cx, cy, r, score = found[:4]
    return (cx / scale, cy / scale, r / scale), score


def stable(errors):
    return errors[0] <= CENTRE_OK and errors[1] <= RADIUS_OK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with (data_dir / "train.csv").open(encoding="utf-8-sig", newline="") as file:
        ids = [row["ID"] for row in csv.DictReader(file)]
    if args.limit:
        ids = ids[: args.limit]

    stats = {v: {"found": 0, "weak": 0, "half": 0, "pad": 0, "seconds": 0.0, "differ": []}
             for v in VARIANTS}
    per_image = []
    tiles = {v: [] for v in VARIANTS}

    for number, name in enumerate(ids):
        image = cv2.imread(str(data_dir / "train" / "train" / name))
        height, width = image.shape[:2]
        half_w, half_h = width // 2, height // 2
        half_image = cv2.resize(image, (half_w, half_h), interpolation=cv2.INTER_AREA)
        pad = round(PAD_FRACTION * height)
        pad_image = cv2.copyMakeBorder(image, pad, 0, pad, 0, cv2.BORDER_CONSTANT,
                                       value=(0, 0, 0))

        results = {}
        for variant in VARIANTS:
            s = stats[variant]
            start = time.perf_counter()
            circle, score = run_variant(image, variant)
            s["seconds"] += time.perf_counter() - start
            results[variant] = circle
            per_image.append([number, name, variant,
                              *(["", "", ""] if circle is None else [f"{v:.1f}" for v in circle]),
                              "" if score is None else f"{score:.3f}"])
            if circle is None:
                continue
            s["found"] += 1
            s["weak"] += score < PARAMS["weak_score"]

            half, _ = run_variant(half_image, variant)
            if half is not None:
                half = (half[0] * width / half_w, half[1] * height / half_h,
                        half[2] * width / half_w)
            s["half"] += stable(relative_errors(circle, half))

            moved, _ = run_variant(pad_image, variant)
            if moved is not None:
                moved = (moved[0] - pad, moved[1] - pad, moved[2])
            s["pad"] += stable(relative_errors(circle, moved))

        reference = results[REFERENCE]
        for variant, circle in results.items():
            if variant == REFERENCE:
                continue
            same = (reference is None and circle is None) or (
                reference is not None and circle is not None
                and stable(relative_errors(reference, circle)))
            if not same:
                stats[variant]["differ"].append(number)
                ref_tile = thumbnail(draw_overlay(image, None if reference is None else
                                     dict(zip(("cx", "cy", "radius"), reference),
                                          score=0, method="current")), number)
                var_tile = thumbnail(draw_overlay(image, None if circle is None else
                                     dict(zip(("cx", "cy", "radius"), circle),
                                          score=0, method="variant")), f"{number}v")
                tiles[variant].append(np.hstack([ref_tile, var_tile]))
        print(f"{number + 1}/{len(ids)}", end="\r", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(ids)
    with (out_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["variant", "found", "weak", "half_stable", "pad_stable",
                         "ms_per_image", "differ_from_current"])
        print()
        print(f"{'variant':30} found  weak  half  pad   ms   differ (of {n})")
        for variant, s in stats.items():
            ms = 1000 * s["seconds"] / n
            differ = "" if variant == REFERENCE else " ".join(map(str, s["differ"]))
            writer.writerow([variant, s["found"], s["weak"], s["half"], s["pad"],
                             f"{ms:.0f}", differ])
            count = "-" if variant == REFERENCE else len(s["differ"])
            print(f"{variant:30} {s['found']:5} {s['weak']:5} {s['half']:5} {s['pad']:4}"
                  f" {ms:5.0f}  {count}")

    with (out_dir / "per_image.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["number", "ID", "variant", "cx", "cy", "radius", "score"])
        writer.writerows(per_image)

    # Side-by-side sheets: left = current method, right = variant (suffix v).
    for k, variant in enumerate(VARIANTS):
        pairs = tiles[variant]
        if not pairs:
            continue
        blank = np.full_like(pairs[0], 255)
        pairs = pairs + [blank] * (-len(pairs) % 3)
        grid = np.vstack([np.hstack(pairs[i : i + 3]) for i in range(0, len(pairs), 3)])
        cv2.imwrite(str(out_dir / f"differ_{k:02d}.jpg"), grid)
    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
