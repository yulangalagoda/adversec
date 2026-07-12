"""
ROAD pipeline verification (step-3 migration gate, ROAD side).

Mirrors test_pipeline.py for ROAD: runs the dataset-agnostic stages on the ROAD
canonical table and checks every derived number against the committed ROAD
artifacts (road_strict.csv / road_test.csv / road_train_dup.csv /
road_stage2_arrays.npz + road_*.joblib). Skips gracefully when the ROAD raw
captures or committed artifacts are absent (i.e. in the Windows working copy);
runs on the lab machine.

Run:  python tests/test_pipeline_road.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adversec.contract import CANONICAL_COLUMNS, FEATURES, LABEL_COLUMN   # noqa: E402
from adversec.datasets.registry import get_dataset                        # noqa: E402
from adversec.pipeline import (                                           # noqa: E402
    duplicate_train_classes,
    encode_labels,
    scale_features,
    split_train_test,
    strict_dedup,
)

PROC = ROOT / "datasets" / "processed"

# Expected strict per-class unique signatures (the diversity gradient, section 43).
EXPECTED_STRICT = {
    "benign": 21188,
    "max-speedometer": 10559,
    "reverse-light-on": 5994,
    "reverse-light-off": 1525,
    "fuzzing": 592,
}
EXPECTED_TEST_SIZE = 7972
EXPECTED_TRAIN_SIZE = 31886   # duplication does not fire (all classes above target)


def _counts(df):
    return {str(k): int(v) for k, v in df[LABEL_COLUMN].value_counts().items()}


def _canon_sorted(df):
    return df[CANONICAL_COLUMNS].sort_values(CANONICAL_COLUMNS).reset_index(drop=True)


def _sort_rows(X):
    return X[np.lexsort(X.T[::-1])]


def test_road_pipeline():
    raw_ok = (ROOT / "datasets" / "raw" / "road" / "attacks" / "capture_metadata.json").exists()
    art_ok = (PROC / "road_strict.csv").exists()
    if not (raw_ok and art_ok):
        print("SKIP road pipeline: raw captures or committed artifacts absent (run on the lab machine)")
        return

    ds = get_dataset("road")
    meta = ds.meta()
    raw = ds.load()

    strict = strict_dedup(raw, FEATURES)
    assert _counts(strict) == EXPECTED_STRICT, _counts(strict)
    assert _canon_sorted(strict).equals(_canon_sorted(pd.read_csv(PROC / "road_strict.csv")))
    print(f"strict ok: {len(strict)} unique, per-class == diversity gradient, content == road_strict.csv")

    train, test = split_train_test(strict)
    assert len(test) == EXPECTED_TEST_SIZE and len(train) == EXPECTED_TRAIN_SIZE
    assert _canon_sorted(test).equals(_canon_sorted(pd.read_csv(PROC / "road_test.csv")))
    print(f"split ok: {len(train)} train / {len(test)} test, content == road_test.csv")

    train_dup = duplicate_train_classes(train, benign_class=meta.benign_label)
    assert len(train_dup) == EXPECTED_TRAIN_SIZE   # dup does not fire on signature-rich ROAD
    assert _canon_sorted(train_dup).equals(_canon_sorted(pd.read_csv(PROC / "road_train_dup.csv")))
    print(f"augment ok: {len(train_dup)} rows (duplication did not fire), content == road_train_dup.csv")

    y_train, y_test, enc = encode_labels(train_dup, test)
    X_train, X_test, scaler = scale_features(train_dup, test, FEATURES)
    arrays = np.load(PROC / "road_stage2_arrays.npz")
    assert X_train.shape == arrays["X_train"].shape == (len(train_dup), len(FEATURES))
    assert X_test.shape == arrays["X_test"].shape == (len(test), len(FEATURES))

    import joblib
    comm_scaler = joblib.load(PROC / "road_feature_scaler.joblib")
    comm_enc = joblib.load(PROC / "road_label_encoder.joblib")
    assert np.allclose(scaler.data_min_, comm_scaler.data_min_)
    assert np.allclose(scaler.data_max_, comm_scaler.data_max_)
    assert list(enc.classes_) == list(comm_enc.classes_)
    assert np.allclose(_sort_rows(X_test), _sort_rows(arrays["X_test"]), atol=1e-6)
    assert np.allclose(_sort_rows(X_train), _sort_rows(arrays["X_train"]), atol=1e-6)
    print(f"encode+scale ok: shapes {X_train.shape}/{X_test.shape}, scaler+encoder+content match")
    print(f"  label mapping: {list(enc.classes_)}")


if __name__ == "__main__":
    test_road_pipeline()
    print("\nROAD STEP-3 GATE PASSED (or skipped if data absent)")
