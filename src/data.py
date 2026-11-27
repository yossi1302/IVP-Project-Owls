import csv
import random
from pathlib import Path

import cv2


project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data" / "raw"
split_file = project_dir / "data" / "splits" / "train_validation.csv"


def check_filename(name):
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"Invalid image filename: {name}")


def read_metadata(split="train", root=data_dir):
    """Read the CSV in its original order. Times stay as strings."""
    if split not in ("train", "test"):
        raise ValueError("Split must be 'train' or 'test'")

    columns = ["ID", "time"] if split == "train" else ["ID"]
    with (Path(root) / f"{split}.csv").open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != columns:
            raise ValueError(f"Expected columns {columns} in {split}.csv")
        rows = list(reader)

    ids = set()
    for row in rows:
        if None in row or any(value in (None, "") for value in row.values()):
            raise ValueError(f"Missing or extra fields in {split}.csv")
        check_filename(row["ID"])
        if row["ID"] in ids:
            raise ValueError(f"Duplicate image ID: {row['ID']}")
        ids.add(row["ID"])
    return rows


def load_image(image_id, split="train", root=data_dir):
    """Load an image in OpenCV BGR format, without resizing or cropping."""
    if split not in ("train", "test"):
        raise ValueError("Split must be 'train' or 'test'")
    check_filename(image_id)
    path = Path(root) / split / split / image_id
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def load_training_split(root=data_dir, path=split_file):
    """Create the fixed 80/20 split once, then reuse the saved membership."""
    rows = read_metadata("train", root)
    if len(rows) < 2:
        raise ValueError("Need at least two labelled images to make a split")

    ids = sorted(row["ID"] for row in rows)
    validation_count = max(1, len(ids) // 5)
    path = Path(path)

    if not path.exists():
        # Sorting first makes membership independent of the CSV row order.
        validation_ids = set(random.Random(42).sample(ids, validation_count))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(["ID", "subset"])
            for image_id in ids:
                subset = "validation" if image_id in validation_ids else "development"
                writer.writerow([image_id, subset])

    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["ID", "subset"]:
            raise ValueError("Split CSV must have ID and subset columns")
        assignments = list(reader)

    membership = {}
    for row in assignments:
        if None in row or row["subset"] not in ("development", "validation"):
            raise ValueError("Invalid row in the saved split")
        if row["ID"] in membership:
            raise ValueError(f"Repeated ID in the saved split: {row['ID']}")
        membership[row["ID"]] = row["subset"]

    if set(membership) != set(ids):
        raise ValueError("Saved split does not match train.csv; check the dataset")
    if list(membership.values()).count("validation") != validation_count:
        raise ValueError("Saved split does not have the expected validation size")

    development = [row for row in rows if membership[row["ID"]] == "development"]
    validation = [row for row in rows if membership[row["ID"]] == "validation"]
    return development, validation


if __name__ == "__main__":
    development, validation = load_training_split()
    test = read_metadata("test")
    print(f"Development: {len(development)} labelled images")
    print(f"Validation: {len(validation)} labelled images")
    print(f"Test: {len(test)} unlabelled images")
    print(f"Split saved in {split_file.relative_to(project_dir)}")
