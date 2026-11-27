import csv
from pathlib import Path

import cv2

from src.data import load_image, load_training_split
from src.localization import locate_clock_details


def localize_images(rows):
    results = []
    for row in rows:
        name = row["ID"]
        result = {"ID": name, "cx": "", "cy": "", "radius": "",
                  "score": "", "method": "", "status": "not_found", "error": ""}

        try:
            image = load_image(name)
            details = locate_clock_details(image)
            if details is not None:
                for key in ("cx", "cy", "radius", "score", "method"):
                    result[key] = details[key]
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
        writer = csv.DictWriter(file, fieldnames=["ID", "cx", "cy", "radius", "score", "method", "status", "error"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)

    found = sum(row["status"] == "found" for row in results)
    errors = sum(row["status"] == "error" for row in results)
    fallback = sum(row["method"] == "fallback" for row in results)
    print(f"Detected circles in {found}/{len(results)} images. Processing errors: {errors}.")
    print(f"Used the fallback result for {fallback} images.")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
