# AdverSec

Adversarial robustness of ML-based intrusion detection for in-vehicle CAN bus
networks, using the CICIoV2024 dataset.

## Environment
- Python 3.12, venv at `~/envs/adversec`
- Key libs: PyTorch (CUDA), scikit-learn, adversarial-robustness-toolbox 1.20.1
- Install: `pip install -r requirements.txt`
- NOTE: do NOT install the PyPI package `art` (ASCII-art lib) — it shadows the
  Adversarial Robustness Toolbox and breaks imports.

## Data
CICIoV2024 decimal CSVs go in `datasets/raw/decimal/` (six per-class files).
Not tracked in git (regenerable, large).

## Run order
1. `notebooks/01_data_cleaning.ipynb` — load, audit (99.75% duplication),
   de-duplicate, split, light duplication. Produces processed datasets.
2. `notebooks/02_preprocessing.ipynb` — label encoding + feature scaling.
3. `notebooks/03_baselines.ipynb` — RF + 1D-CNN baselines, cross-validation,
   ablation, ANOVA RF variant.
4. `notebooks/04_steering_investigation.ipynb` — STEERING divergence from
   Le & Alsmadi.
5. `notebooks/05_adversarial_attacks.ipynb` — FGSM/PGD attacks, transfer to RF,
   adversarial training (defence), CV of the defended result, latency benchmark.

## Source modules (`src/`)
config, cleaning, preprocessing, models, evaluation, crossval, attacks, defense

## Key findings
- Accuracy trap quantified: leaky CV ~1.00 vs honest CV ~0.68–0.77 macro-F1.
- On honestly de-duplicated data, RF and CNN are comparable (not RF-superior).
- Adversarial training gives the CNN a robustness advantage the RF cannot match,
  confirmed across cross-validation folds.