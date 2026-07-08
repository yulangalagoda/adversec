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








# ---------------------------------------------------------------------------
# ROAD dataset (Oak Ridge) settings
# ---------------------------------------------------------------------------
ROAD_RAW_DIR = DATA_DIR / "raw" / "road"                # ROAD raw captures
ROAD_ATTACKS_DIR = ROAD_RAW_DIR / "attacks"             # attack .log files + metadata
ROAD_AMBIENT_DIR = ROAD_RAW_DIR / "ambient"             # benign .log files

# Map clean class names to the capture files that supply them.
# Non-masquerade fabrication captures only.
#
# correlated-signal is EXCLUDED from modelling: it collapses to a single
# unique signature under strict de-duplication (fully-specified fixed payload),
# so it is not statistically viable to train or test. It is retained only as a
# reported de-duplication statistic (the floor of the diversity gradient).
#
# Masquerade variants are EXCLUDED: they are built by deleting the target-ID
# legitimate frames from the fabrication captures, so their injected frames
# duplicate the fabrication signatures rather than adding diversity. Masquerade
# poses a frequency-detection challenge out of scope for a payload-level model.
ROAD_ATTACK_CAPTURES = {
    "max-speedometer": [
        "max_speedometer_attack_1",
        "max_speedometer_attack_2",
        "max_speedometer_attack_3",
    ],
    "reverse-light-on": [
        "reverse_light_on_attack_1",
        "reverse_light_on_attack_2",
        "reverse_light_on_attack_3",
    ],
    "reverse-light-off": [
        "reverse_light_off_attack_1",
        "reverse_light_off_attack_2",
        "reverse_light_off_attack_3",
    ],
    "fuzzing": [
        "fuzzing_attack_1",
        "fuzzing_attack_2",
        "fuzzing_attack_3",
    ],
}

# Captures that require fuzzing-style labelling (all-FF payload filter)
# rather than the standard ID + interval + byte-mask labelling.
ROAD_FUZZING_CLASSES = ["fuzzing"]

# Ambient captures chosen for benign diversity (dyno + real road).
ROAD_AMBIENT_CAPTURES = [
    "ambient_dyno_drive_basic_long",
    "ambient_highway_street_driving_long",
]

# Cap on benign frames sampled per ambient capture, so benign does not
# swamp the attack classes. Seeded for reproducibility.
ROAD_AMBIENT_SAMPLE_PER_CAPTURE = 20000


# Adversarial-training epsilon ranges for the ROAD defence comparison.
# MEANINGFUL: the band where classes transition robust -> collapsed (defence can learn).
# FULL: the complete sweep, matching CICIoV methodology (includes high-eps examples).
ROAD_DEFENCE_EPS_MEANINGFUL = [0.01, 0.05, 0.10]
ROAD_DEFENCE_EPS_FULL = [0.01, 0.05, 0.10, 0.20, 0.30]