"""
config.py
=========
Shared pipeline settings for the AdverSec package.

NOTHING dataset-specific lives here. Per-dataset settings (raw paths, class maps,
benign sourcing, id_max) live in configs/<name>.yaml and are read by that
dataset's adapter via load_dataset_config(). Keeping this file dataset-agnostic
is half of what un-mixes the codebase.
"""
from __future__ import annotations

from pathlib import Path

import yaml


# Project root = the folder that contains this package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "datasets"
PROCESSED_DIR = DATA_DIR / "processed"     # prep artifacts: <name>_strict.csv, _stage2_arrays.npz, ... (tracked in git)
RESULTS_DIR = PROJECT_ROOT / "results"     # citable JSON reports (tracked in git)
CONFIGS_DIR = PROJECT_ROOT / "configs"


# --- Reproducibility ---
RANDOM_SEED = 42

# --- Signature-level split rule (dataset-agnostic) ---
TEST_FRACTION = 0.20
SMALL_CLASS_THRESHOLD = 6      # classes below this send a single-signature test floor

# --- Light duplication (a convergence crutch; only fires on scarce classes) ---
DUP_TARGET = 200

# --- Adversarial attack hyper-parameters ---
FGSM_EPSILONS = [0.01, 0.05, 0.10, 0.20, 0.30]
PGD_STEP_SIZE = 0.01
PGD_MAX_ITER = 40

# --- Madry-style (true min-max) adversarial training ---
# Inner-loop PGD steps used WHILE TRAINING, crafted fresh against the model's
# current weights every batch. Deliberately lower than PGD_MAX_ITER (used to
# evaluate robustness afterwards): Madry et al. (2018) use fewer steps at
# train time than test time to keep the inner loop tractable (7 for CIFAR-10),
# then verify with a stronger attack budget at evaluation. The eval-time PGD
# attack used to measure white-box robustness is unchanged (PGD_MAX_ITER, 40).
MADRY_TRAIN_EPSILON = 0.10
MADRY_TRAIN_MAX_ITER = 7


def load_dataset_config(name: str) -> dict:
    """Read configs/<name>.yaml into a plain dict."""
    path = CONFIGS_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
