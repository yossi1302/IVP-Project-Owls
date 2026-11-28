import argparse
import csv
import re
from pathlib import Path

from src.data import read_metadata


output_file = Path(__file__).resolve().parent / "outputs" / "submission.csv"


def write_submission(predictions, output_path=output_file):
    test_ids = [row["ID"] for row in read_metadata("test")]
    if len(test_ids) != 150:
        raise ValueError(f"Expected 150 test IDs, found {len(test_ids)}")

    times = {}
    for row in predictions:
        if None in row:
            raise ValueError("Prediction row has extra CSV fields")
        name = row.get("ID")
        time = row.get("time")
        if name not in test_ids:
            raise ValueError(f"Unknown test ID: {name}")
        if name in times:
            raise ValueError(f"Duplicate prediction: {name}")
        if not isinstance(time, str) or not re.fullmatch(r"(0[1-9]|1[0-2]):[0-5][0-9]", time):
            raise ValueError(f"Invalid time for {name}: {time}. Use HH:MM with hours 01-12.")
        times[name] = time

    missing = [name for name in test_ids if name not in times]
    if missing:
        raise ValueError(f"Missing predictions: {', '.join(missing)}")

    # Validate everything before opening the output file.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["ID", "time"])
        for name in test_ids:
            writer.writerow([name, times[name]])
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check predictions and write the submission CSV.")
    parser.add_argument("predictions", type=Path, help="CSV with ID,time columns")
    parser.add_argument("--output", type=Path, default=output_file)
    args = parser.parse_args()

    with args.predictions.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["ID", "time"]:
            raise ValueError("Predictions CSV must have ID,time columns")
        predictions = list(reader)

    path = write_submission(predictions, args.output)
    print(f"Saved 150 predictions to {path}")
