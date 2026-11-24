import csv
from pathlib import Path


data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"


def read_ids(filename, columns):
    with (data_dir / filename).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != columns:
            raise ValueError(f"Wrong columns in {filename}: {reader.fieldnames}")
        rows = list(reader)

    ids = []
    for row in rows:
        if None in row or None in row.values():
            raise ValueError(f"Incomplete or extra fields in {filename}")

        name = row["ID"]
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError(f"Invalid filename in {filename}: {name}")
        ids.append(name)

        if "time" in columns:
            time = row["time"]
            digits = time[:2] + time[3:]
            if len(time) != 5 or time[2] != ":" or not digits.isascii() or not digits.isdigit():
                raise ValueError(f"Invalid time for {name}: {time}")
            hour = int(time[:2])
            minute = int(time[3:])
            if not (1 <= hour <= 12 and 0 <= minute <= 59):
                raise ValueError(f"Invalid time for {name}: {time}")

    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate filenames in {filename}")
    return ids


def check_images(split, ids, expected_count):
    if len(ids) != expected_count:
        raise ValueError(f"Expected {expected_count} {split} images, found {len(ids)}")

    # The download has nested folders: train/train and test/test.
    folder = data_dir / split / split
    files = set()
    for file in folder.iterdir():
        if file.is_file() and not file.name.startswith("."):
            files.add(file.name)

    missing = set(ids) - files
    extra = files - set(ids)
    if missing or extra:
        raise ValueError(f"{split}: missing files {sorted(missing)}, extra files {sorted(extra)}")

    for name in ids:
        if (folder / name).stat().st_size == 0:
            raise ValueError(f"Empty image file: {name}")
    print(f"{split}: checked {len(ids)} image files")


if __name__ == "__main__":
    train_ids = read_ids("train.csv", ["ID", "time"])
    test_ids = read_ids("test.csv", ["ID"])
    check_images("train", train_ids, 100)
    check_images("test", test_ids, 150)

    submission_ids = read_ids("submission_example.csv", ["ID", "time"])
    if submission_ids != test_ids:
        raise ValueError("Submission IDs do not match test.csv in the same order")

    print("CSV and file checks passed. Image contents are not checked here.")
