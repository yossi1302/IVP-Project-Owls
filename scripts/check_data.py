"""Check the local competition CSVs and image inventory (standard library only)."""

import argparse
import csv
from pathlib import Path
import re
import sys


TIME_PATTERN = re.compile(r"(?:0[1-9]|1[0-2]):[0-5][0-9]")
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


def read_rows(path, columns):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError(f"{path.name}: expected columns {columns}")
        rows = list(reader)
    ids = []
    for number, row in enumerate(rows, start=2):
        name = row.get("ID")
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError(f"{path.name}:{number}: invalid image ID")
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path.name}:{number}: incorrect number of fields")
        if "time" in columns and not TIME_PATTERN.fullmatch(row["time"]):
            raise ValueError(f"{path.name}:{number}: expected time 01:00 through 12:59")
        ids.append(name)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: duplicate image IDs")
    return ids


def check_dataset(root):
    test_ids = None
    for split, count, columns in (
        ("train", 100, ["ID", "time"]),
        ("test", 150, ["ID"]),
    ):
        ids = read_rows(root / f"{split}.csv", columns)
        if len(ids) != count:
            raise ValueError(f"{split}: expected {count} rows, found {len(ids)}")
        folder = root / split / split
        images = {p.name for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")}
        missing = set(ids) - images
        extra = images - set(ids)
        if missing or extra:
            raise ValueError(f"{split}: missing images {sorted(missing)}; extra files {sorted(extra)}")
        empty = [name for name in ids if (folder / name).stat().st_size == 0]
        if empty:
            raise ValueError(f"{split}: empty image files {empty}")
        print(f"PASS {split}: {count} unique IDs match {count} nonempty image files")
        if split == "test":
            test_ids = ids
    submission_ids = read_rows(root / "submission_example.csv", ["ID", "time"])
    if submission_ids != test_ids:
        raise ValueError("Submission example IDs must match test.csv in the same order")
    print("PASS submission example: valid time format and all 150 test IDs in order")
    print("Dataset inventory checks passed. Image decoding is not checked.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    try:
        check_dataset(args.data_dir)
    except (OSError, ValueError, csv.Error) as error:
        print(f"Dataset check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
