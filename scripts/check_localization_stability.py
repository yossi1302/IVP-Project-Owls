"""Check that detections move correctly when a training image is changed.

Usage:
    .venv/bin/python scripts/check_localization_stability.py [--limit N]

For every training image where a clock is found, the image is changed in
two ways whose effect on the true clock position is known exactly:
- half:  shrunk to half size, so the circle should be the same after
         multiplying by 2 (checks resizing + mapping back to pixels)
- pad:   a black border is added on the top and left, so the centre should
         move by the border width and the radius should not change

The error is reported relative to the radius found on the original image.
Large errors mean the detector picked a different circle, which marks the
image as difficult. This is a diagnostic only; it doesn't change detection.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.localization import locate_clock  # noqa: E402

data_dir = root / "data" / "raw"
out_file = root / "outputs" / "localization" / "train_stability.csv"

CENTRE_OK = 0.05  # centre error at most 5% of the radius
RADIUS_OK = 0.10  # radius error at most 10%
PAD_FRACTION = 0.10  # border width as a fraction of the image height


def relative_errors(reference, moved_back):
    """(centre error, radius error) relative to the reference radius."""
    if moved_back is None:
        return float("inf"), float("inf")
    cx, cy, r = reference
    mx, my, mr = moved_back
    return float(np.hypot(mx - cx, my - cy) / r), abs(mr - r) / r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with (data_dir / "train.csv").open(encoding="utf-8-sig", newline="") as file:
        ids = [row["ID"] for row in csv.DictReader(file)]
    if args.limit:
        ids = ids[: args.limit]

    rows = []
    for number, name in enumerate(ids):
        image = cv2.imread(str(data_dir / "train" / "train" / name))
        reference = locate_clock(image)
        if reference is None:
            continue
        height, width = image.shape[:2]

        half_w, half_h = width // 2, height // 2
        half = locate_clock(cv2.resize(image, (half_w, half_h), interpolation=cv2.INTER_AREA))
        if half is not None:
            # x and y are scaled separately because // 2 may round odd sizes.
            half = (half[0] * width / half_w, half[1] * height / half_h, half[2] * width / half_w)

        pad = round(PAD_FRACTION * height)
        padded = cv2.copyMakeBorder(image, pad, 0, pad, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        moved = locate_clock(padded)
        if moved is not None:
            moved = (moved[0] - pad, moved[1] - pad, moved[2])

        rows.append([number, name, *relative_errors(reference, half),
                     *relative_errors(reference, moved)])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["number", "ID", "half_centre", "half_radius", "pad_centre", "pad_radius"])
        writer.writerows([r[:2] + [f"{v:.3f}" for v in r[2:]] for r in rows])

    print(f"{len(rows)} images with a detection")
    for label, c, r in (("half", 2, 3), ("pad", 4, 5)):
        stable = [row[0] for row in rows if row[c] <= CENTRE_OK and row[r] <= RADIUS_OK]
        jumped = [row[0] for row in rows if row[c] > 0.15]
        print(f"{label}: {len(stable)} stable (centre <= {CENTRE_OK:.0%}, radius <= {RADIUS_OK:.0%});"
              f" centre moved > 15% of radius on images {jumped}")
    print(f"Details in {out_file}")


if __name__ == "__main__":
    main()
