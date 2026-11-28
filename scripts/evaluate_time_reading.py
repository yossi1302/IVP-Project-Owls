from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    circular_time_difference,
    evaluate_predictions,
    score_prediction,
)
from src.localization import locate_clock
from src.time_reading import read_time


def load_train_rows(data_dir: Path) -> list[dict[str, str]]:
    with (data_dir / "train.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_split_membership(split_csv: Path) -> dict[str, str]:
    with split_csv.open(encoding="utf-8-sig", newline="") as f:
        return {row["ID"]: row["subset"] for row in csv.DictReader(f)}


def evaluate_rows(
    rows: list[dict[str, str]],
    image_dir: Path,
    label: str,
    show_worst: int = 0,
) -> dict[str, object]:
    all_preds: list[str | None] = []
    all_targets: list[str] = []
    loc_preds: list[str | None] = []
    loc_targets: list[str] = []
    localized_count = 0
    details: list[tuple[int, str, str, str | None, float, bool]] = []

    for row in rows:
        image_id = row["ID"]
        truth = row["time"]
        image = cv2.imread(str(image_dir / image_id), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_dir / image_id}")

        circle = locate_clock(image)
        if circle is None:
            all_preds.append(None)
            all_targets.append(truth)
            details.append((999, image_id, truth, None, 0.0, False))
            continue

        localized_count += 1
        pred = read_time(image, circle)
        all_preds.append(pred)
        all_targets.append(truth)
        loc_preds.append(pred)
        loc_targets.append(truth)

        err = 999 if pred is None else circular_time_difference(pred, truth)
        pts = score_prediction(pred, truth)
        details.append((err, image_id, truth, pred, pts, True))

    overall = evaluate_predictions(all_preds, all_targets)
    localized = evaluate_predictions(loc_preds, loc_targets) if loc_targets else None

    print(f"=== {label} ({len(rows)} images) ===")
    print(f"Localized: {localized_count}/{len(rows)}")
    print(
        f"End-to-end score: {overall['accuracy_with_tolerance']:.4f} "
        f"| tiers={overall['tier_counts']} "
        f"| MAE(pred)={overall['mae_predicted']:.1f} min "
        f"| median(pred)={overall['median_predicted']} min"
    )
    if localized is not None:
        print(
            f"Localized-only score: {localized['accuracy_with_tolerance']:.4f} "
            f"| tiers={localized['tier_counts']}"
        )

    if show_worst > 0:
        print(f"Worst {show_worst} predictions:")
        for err, image_id, truth, pred, pts, was_loc in sorted(
            details, key=lambda x: x[0], reverse=True
        )[:show_worst]:
            loc_tag = "ok" if was_loc else "no_circle"
            err_str = "N/A" if err == 999 else f"{err:3d}m"
            print(
                f"  err={err_str:>4s} pts={pts:.1f} loc={loc_tag:9s} "
                f"truth={truth} pred={str(pred):5s} {image_id}"
            )

    return overall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "train_validation.csv",
    )
    parser.add_argument("--show-worst", type=int, default=0)
    args = parser.parse_args()

    rows = load_train_rows(args.data_dir)
    image_dir = args.data_dir / "train" / "train"

    if args.split_csv.is_file():
        membership = load_split_membership(args.split_csv)
        dev_rows = [r for r in rows if membership.get(r["ID"]) == "development"]
        val_rows = [r for r in rows if membership.get(r["ID"]) == "validation"]
        evaluate_rows(dev_rows, image_dir, "development", show_worst=args.show_worst)
        print()
        evaluate_rows(val_rows, image_dir, "validation", show_worst=args.show_worst)
        print()

    evaluate_rows(rows, image_dir, "all_train", show_worst=args.show_worst)


if __name__ == "__main__":
    main()
