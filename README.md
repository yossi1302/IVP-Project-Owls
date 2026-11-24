# IVP Project Owls

Read analogue clocks from images using classical image processing.

## Setup

Use Python 3.12. A local `.venv` is already set up; on a fresh clone, create it with `python3.12 -m venv .venv`.

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Extract the Kaggle download into `data/raw/`, preserving the archive's `train/train/` and `test/test/` folders:

```sh
unzip /path/to/ivp-2026.zip -d data/raw
python3 scripts/check_data.py
```

Run `jupyter lab` for notebooks. The dataset and `.venv` are excluded from Git. The clock-reading pipeline is not implemented yet.
