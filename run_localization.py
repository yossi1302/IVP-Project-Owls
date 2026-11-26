import csv
from pathlib import Path

import cv2

from src.data import load_image, load_training_split
from src.localization import locate_clock


def localize_images(rows):
    results = []
    for row in rows:
        name = row["ID"]
        result = {"ID": name, "cx": "", "cy": "", "radius": "",
                  "status": "not_found", "error": ""}

        try:
            image = load_image(name)
            circle = locate_clock(image)
            if circle is not None:
                result["cx"], result["cy"], result["radius"] = circle
                result["status"] = "found"
        except (OSError, ValueError, cv2.error) as error:
            result["status"] = "error"
            result["error"] = str(error)

        results.append(result)
        print(f"{name}: {result['status']}")
    return results


def main():
    development, _ = load_training_split()
    results = localize_images(development)

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "development_localization.csv"
    with output_file.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["ID", "cx", "cy", "radius", "status", "error"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)

    found = sum(row["status"] == "found" for row in results)
    errors = sum(row["status"] == "error" for row in results)
    print(f"Detected circles in {found}/{len(results)} images. Processing errors: {errors}.")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
