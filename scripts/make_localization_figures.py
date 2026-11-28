"""Make the report figures for clock localization.

Usage:
    .venv/bin/python scripts/make_localization_figures.py

Writes PNG files to outputs/report_figures/ (git-ignored):
- localization_pipeline.png   the steps on one training image
- localization_fallback.png   first pass vs CLAHE fallback on weak images
- localization_examples.png   typical successes and failure cases

The example images are picked by their row number in train.csv to show
typical behaviour, including failures. They only illustrate the method;
the detector itself treats every image the same way.
"""

import csv
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.localization import (  # noqa: E402
    PARAMS,
    enhance_contrast,
    find_circle,
    resize_for_detection,
    score_candidates,
    select_circle,
    to_gray,
)

data_dir = root / "data" / "raw"
out_dir = root / "outputs" / "report_figures"

PIPELINE_IMAGE = 13            # alarm clock in front of tree bark
FALLBACK_IMAGES = [37, 31]     # weak first-pass results
SUCCESS_IMAGES = [3, 26, 46, 61, 70, 72]
FAILURE_IMAGES = [55, 20, 59, 58]  # square, rimless, small/tower, inner ring
FAILURE_NOTES = {55: "square clock", 20: "no rim", 59: "small, wrong circle", 58: "inner ring"}

GREEN = "#2ca02c"
RED = "#d62728"


def training_ids():
    with (data_dir / "train.csv").open(encoding="utf-8-sig", newline="") as file:
        return [row["ID"] for row in csv.DictReader(file)]


def load_small(number, ids):
    """Training image as resized RGB (for display) and resized gray."""
    image = cv2.imread(str(data_dir / "train" / "train" / ids[number]))
    small, _scale = resize_for_detection(image)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB), to_gray(small)


def show(ax, picture, title, cmap=None):
    ax.imshow(picture, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def draw_circle(ax, circle, color=GREEN, width=2.0):
    if circle is None:
        ax.text(0.5, 0.5, "not found", transform=ax.transAxes, ha="center",
                color=RED, fontsize=10, weight="bold")
        return
    cx, cy, r = circle[:3]
    ax.add_patch(Circle((cx, cy), r, fill=False, color=color, linewidth=width))
    ax.plot(cx, cy, "+", color=RED, markersize=8, markeredgewidth=2)


def pipeline_figure(ids):
    rgb, gray = load_small(PIPELINE_IMAGE, ids)
    scored, blurred, edges = score_candidates(gray)
    chosen = select_circle(scored)

    fig, axes = plt.subplots(1, 5, figsize=(15, 2.8))
    show(axes[0], rgb, "(a) resized input")
    show(axes[1], blurred, "(b) gray + median blur", cmap="gray")
    show(axes[2], edges, "(c) Canny edges", cmap="gray")

    show(axes[3], rgb, f"(d) {len(scored)} Hough candidates")
    norm = plt.Normalize(min(s[0] for s in scored), max(s[0] for s in scored))
    cmap = plt.get_cmap("viridis")
    for score, cx, cy, r in sorted(scored):  # best drawn last, on top
        axes[3].add_patch(Circle((cx, cy), r, fill=False, color=cmap(norm(score)),
                                 linewidth=1.0))
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[3],
                       fraction=0.046, pad=0.02)
    bar.set_label("radial score", fontsize=8)

    show(axes[4], rgb, f"(e) selected, score {chosen[3]:.2f}")
    draw_circle(axes[4], chosen)
    fig.tight_layout()
    fig.savefig(out_dir / "localization_pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fallback_figure(ids):
    fallback_params = dict(PARAMS, hough_votes=PARAMS["fallback_hough_votes"],
                           canny_high=PARAMS["fallback_canny_high"])
    fig, axes = plt.subplots(len(FALLBACK_IMAGES), 3, figsize=(10, 3.4 * len(FALLBACK_IMAGES)))
    for row, number in zip(axes, FALLBACK_IMAGES):
        rgb, gray = load_small(number, ids)
        primary = select_circle(score_candidates(gray)[0])
        enhanced = enhance_contrast(gray)
        fallback = select_circle(score_candidates(enhanced, fallback_params)[0])
        final = find_circle(gray)

        title = "first pass" + ("" if primary is None else f", score {primary[3]:.3f}")
        show(row[0], rgb, title)
        draw_circle(row[0], primary)
        title = "CLAHE fallback" + ("" if fallback is None else f", score {fallback[3]:.3f}")
        show(row[1], enhanced, title, cmap="gray")
        draw_circle(row[1], fallback)
        show(row[2], rgb, f"kept: {final[4]}")
        draw_circle(row[2], final)
    fig.tight_layout()
    fig.savefig(out_dir / "localization_fallback.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def examples_figure(ids):
    numbers = SUCCESS_IMAGES + FAILURE_IMAGES
    columns = 5
    rows = (len(numbers) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
    for ax, number in zip(axes.flat, numbers):
        rgb, gray = load_small(number, ids)
        found = find_circle(gray)
        note = FAILURE_NOTES.get(number, "")
        color = RED if number in FAILURE_NOTES else GREEN
        show(ax, rgb, f"#{number}" + (f": {note}" if note else ""))
        draw_circle(ax, found, color=color)
    for ax in list(axes.flat)[len(numbers):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "localization_examples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = training_ids()
    pipeline_figure(ids)
    fallback_figure(ids)
    examples_figure(ids)
    print("Image credits (from the dataset file names):")
    for number in [PIPELINE_IMAGE, *FALLBACK_IMAGES, *SUCCESS_IMAGES, *FAILURE_IMAGES]:
        print(f"  #{number}: {ids[number]}")
    print(f"Figures in {out_dir}")


if __name__ == "__main__":
    main()
