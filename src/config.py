"""
config.py
=========
Every script and notebook in the Adversec project imports configurations from this file.
Chaning a value here once can help the whole pipeline follow behind it.
"""


# Library import
from pathlib import Path


# Anchoring the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Folder path definitions
DATA_DIR = PROJECT_ROOT / "datasets"            # Dataset: raw and processed
RAW_DIR = DATA_DIR / "raw" / "decimal"          # The original dataset files
PROCESSED_DIR = DATA_DIR / "processed"          # Cleaned outputs
MODELS_DIR = PROJECT_ROOT / "models"            # Trained models
ATTACKS_DIR = PROJECT_ROOT / "attacks"          # Adversarial samples
RESULTS_DIR = PROJECT_ROOT / "results"          # Reports


# Raw filenames
RAW_FILES = {
    "benign": "decimal_benign.csv",
    "DoS": "decimal_DoS.csv",
    "spoofing-GAS": "decimal_spoofing-GAS.csv",
    "spoofing-RPM": "decimal_spoofing-RPM.csv",
    "spoofing-SPEED": "decimal_spoofing-SPEED.csv",
    "spoofing-STEERING_WHEEL": "decimal_spoofing-STEERING_WHEEL.csv",
}


# Feature columns
ID_COLUMN = "ID"                                # CAN frame in decimal form: one arbitrary ID plus 8 data types
DATA_COLUMNS = [f"DATA_{i}" for i in range(8)]  # DATA_0 through DATA_7
FEATURE_COLUMNS = [ID_COLUMN] + DATA_COLUMNS    # 9 total columns


# Valid value range
FEATURE_MIN = 0
FEATURE_MAX = 255


# Reproductability and de-duplication settings
RANDOM_SEED = 42                                # The conventional arbitrary choice
RELAXED_MIN_ROWS_PER_CLASS = 1000               # For the relaxed fallback dataset to make sure the 1D-CNN has enough to learn from


# Adversarial attacks hyper parameters
FGSM_EPSILONS = [0.01, 0.05, 0.10, 0.20, 0.30]
PGD_EPSILON = 0.10
PGD_STEP_SIZE = 0.01
PGD_MAX_ITER = 40