# IVP Project Owls

Team project for IVP-2026: reading analogue clocks from images.

## Local dataset

Download `ivp-2026.zip` from the competition. From the repository root, extract it with:

```sh
mkdir -p data/raw
unzip /path/to/ivp-2026.zip -d data/raw
```

The archive has the following layout (the repeated folder names are intentional):

```text
data/raw/
├── train.csv                 # 100 rows: ID,time
├── test.csv                  # 150 rows: ID
├── submission_example.csv    # Example format: ID,time
├── train/train/              # 100 training images
└── test/test/                # 150 test images
```

`ID` is the image filename; `time` uses `HH:MM` text. The submission example is a format reference, not test ground truth. Keep times as strings to preserve leading zeroes.

The dataset and ZIP archives are excluded from Git. Each teammate should download and extract their own local copy.

## Check your setup

With Python 3 installed, run from the repository root:

```sh
python3 scripts/check_data.py
```

This checks CSV columns, unique IDs, time formatting, image inventory, and submission order using only the Python standard library. It does not decode images or predict times. To check another extracted dataset location, pass `--data-dir /path/to/data`.

## Team workflow

Each teammate should use their own Git identity, work on a separate branch, and commit meaningful changes as they complete them. Use pull requests to review and merge contributions while retaining individual authorship. Record experiments and results as work proceeds so the report reflects the team's actual work.
