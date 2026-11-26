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
python3 src/data.py
python3 run_localization.py
```

The loader creates or reuses a fixed 80/20 development/validation split in `data/splits/train_validation.csv` (seed 42). Tune on development images and reserve validation images for evaluation.

`run_localization.py` loads the 80 development images and runs clock detection. It saves circles in original image coordinates and any failures to `outputs/development_localization.csv`. Finding a circle does not guarantee it is the correct clock face.

Run `jupyter lab` for notebooks. The dataset, `.venv`, and generated outputs are excluded from Git. Time prediction is not connected yet.
