"""Draw locate_clock results on dataset images for visual inspection.

Usage:
    .venv/bin/python scripts/show_localization.py [--split train] [--limit N]

Writes, under outputs/localization/:
- <split>/<image>.jpg   one overlay per image (green circle, red centre,
                        radius, score and method bottom-left)
- <split>_results.csv   detected circle per image, in original pixels
- <split>_sheet_K.jpg   contact sheets of 25 numbered thumbnails, where the
                        number is the image's row in the CSV (0-based)

The overlays are only for looking at the detector's behaviour; nothing here
feeds corrections back into the detection.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.localization import locate_clock_details  # noqa: E402

data_dir = root / "data" / "raw"
out_dir = root / "outputs" / "localization"

SHEET_TILE = 300  # thumbnail size in pixels
SHEET_COLUMNS = 5
SHEET_ROWS = 5


def draw_overlay(image, details):
    """Return a copy of image with the detected circle and centre drawn."""
    overlay = image.copy()
    height, width = image.shape[:2]
    thickness = max(2, round(max(height, width) / 300))
    font_scale = thickness / 2
    text_pos = (10 * thickness, height - 10 * thickness)  # bottom-left corner
    if details is None:
        cv2.putText(overlay, "NOT FOUND", text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 255), thickness)
        return overlay
    cx, cy, radius = round(details["cx"]), round(details["cy"]), round(details["radius"])
    cv2.circle(overlay, (cx, cy), radius, (0, 255, 0), thickness)
    cv2.drawMarker(overlay, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS,
                   markerSize=6 * thickness, thickness=thickness)
    label = f"r={radius} score={details['score']:.2f} {details['method']}"
    cv2.putText(overlay, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 255, 255), thickness)
    return overlay


def thumbnail(image, number):
    """Fit image into a square white tile and write its number on it."""
    height, width = image.shape[:2]
    scale = SHEET_TILE / max(height, width)
    small = cv2.resize(image, (round(width * scale), round(height * scale)),
                       interpolation=cv2.INTER_AREA)
    tile = np.full((SHEET_TILE, SHEET_TILE, 3), 255, np.uint8)
    tile[: small.shape[0], : small.shape[1]] = small
    cv2.putText(tile, str(number), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    return tile


def write_sheets(tiles, split):
    per_sheet = SHEET_COLUMNS * SHEET_ROWS
    blank = np.full((SHEET_TILE, SHEET_TILE, 3), 255, np.uint8)
    for k, start in enumerate(range(0, len(tiles), per_sheet)):
        page = tiles[start : start + per_sheet]
        page += [blank] * (per_sheet - len(page))
        rows = [np.hstack(page[r * SHEET_COLUMNS : (r + 1) * SHEET_COLUMNS])
                for r in range(SHEET_ROWS)]
        cv2.imwrite(str(out_dir / f"{split}_sheet_{k}.jpg"), np.vstack(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None, help="only the first N images")
    args = parser.parse_args()

    with (data_dir / f"{args.split}.csv").open(encoding="utf-8-sig", newline="") as file:
        ids = [row["ID"] for row in csv.DictReader(file)]
    if args.limit:
        ids = ids[: args.limit]

    image_dir = data_dir / args.split / args.split
    overlay_dir = out_dir / args.split
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows, tiles = [], []
    for number, name in enumerate(ids):
        image = cv2.imread(str(image_dir / name))
        if image is None:
            print(f"could not read {name}")
            rows.append([name, "", "", "", "", "", "", "", "unreadable"])
            continue
        details = locate_clock_details(image)
        height, width = image.shape[:2]
        if details is None:
            rows.append([name, width, height, "", "", "", "", "", "not_found"])
        else:
            values = [f"{details[k]:.1f}" for k in ("cx", "cy", "radius")]
            rows.append([name, width, height, *values, f"{details['score']:.3f}",
                         details["method"], "found"])
        overlay = draw_overlay(image, details)
        cv2.imwrite(str(overlay_dir / (Path(name).stem + ".jpg")), overlay)
        tiles.append(thumbnail(overlay, number))

    with (out_dir / f"{args.split}_results.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "width", "height", "cx", "cy", "radius", "score", "method",
                         "status"])
        writer.writerows(rows)
    write_sheets(tiles, args.split)

    found = sum(row[-1] == "found" for row in rows)
    print(f"{found}/{len(rows)} clocks found. Overlays in {overlay_dir}")


if __name__ == "__main__":
    main()
