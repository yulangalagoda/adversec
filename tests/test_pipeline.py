"""
Pipeline verification (step-3 migration gate).

Runs the dataset-agnostic stages on CICIoV2024 and checks every derived number
against the committed artifacts:
    results/stage1_audit_report.json
    datasets/processed/ciciov2024_strict.csv / _test.csv / _train_dup.csv
    datasets/processed/stage2_arrays.npz  (+ feature_scaler / label_encoder joblib)

Content comparisons are order-independent (rows sorted), so they verify the
numbers reproduce regardless of pandas/numpy row-ordering across machines.
Skips gracefully if the CIC raw data or committed artifacts are absent.

Run:  python tests/test_pipeline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adversec import config                                             # noqa: E402
from adversec.contract import CANONICAL_COLUMNS, FEATURES, LABEL_COLUMN  # noqa: E402
from adversec.datasets.registry import get_dataset                      # noqa: E402
from adversec.pipeline import (                                         # noqa: E402
    audit_duplication,
    duplicate_train_classes,
    encode_labels,
    scale_features,
    split_train_test,
    strict_dedup,
)

PROC = ROOT / "datasets" / "processed"
RES = ROOT / "results"


def _counts(df) -> dict:
    return {str(k): int(v) for k, v in df[LABEL_COLUMN].value_counts().items()}


def _canon_sorted(df) -> pd.DataFrame:
    return df[CANONICAL_COLUMNS].sort_values(CANONICAL_COLUMNS).reset_index(drop=True)


def _sort_rows(X: np.ndarray) -> np.ndarray:
    return X[np.lexsort(X.T[::-1])]


def test_ciciov_pipeline():
    if not (ROOT / "datasets" / "raw" / "decimal" / "decimal_benign.csv").exists():
        print("SKIP: CIC raw not present")
        return
    if not (RES / "stage1_audit_report.json").exists():
        print("SKIP: committed artifacts not present")
        return

    report = json.load(open(RES / "stage1_audit_report.json"))
    ds = get_dataset("ciciov2024")
    meta = ds.meta()
    raw = ds.load()

    # --- audit ---
    audit = audit_duplication(raw, FEATURES + [LABEL_COLUMN])
    assert audit["total_rows"] == report["raw"]["total_rows"]
    assert audit["unique_signatures"] == report["raw"]["unique_signatures"]
    assert audit["duplicate_rows"] == report["raw"]["duplicate_rows"]
    assert audit["duplication_rate_pct"] == report["raw"]["duplication_rate_pct"]
    print(f"audit ok: {audit['unique_signatures']} unique / {audit['total_rows']} rows "
          f"({audit['duplication_rate_pct']}% duplicated)")

    # --- strict de-dup ---
    strict = strict_dedup(raw, FEATURES)
    assert _counts(strict) == report["strict_per_class"]
    strict_csv = pd.read_csv(PROC / "ciciov2024_strict.csv")
    assert _canon_sorted(strict).equals(_canon_sorted(strict_csv))
    print(f"strict ok: {len(strict)} unique, per-class matches, content == ciciov2024_strict.csv")

    # --- split ---
    train, test = split_train_test(strict)
    assert _counts(test) == report["test_per_class"]
    test_csv = pd.read_csv(PROC / "ciciov2024_test.csv")
    assert _canon_sorted(test).equals(_canon_sorted(test_csv))
    print(f"split ok: {len(train)} train / {len(test)} test, content == ciciov2024_test.csv")

    # --- augment (light duplication) ---
    train_dup = duplicate_train_classes(train, benign_class=meta.benign_label)
    assert _counts(train_dup) == report["train_per_class_after_dup"]
    dup_csv = pd.read_csv(PROC / "ciciov2024_train_dup.csv")
    assert len(train_dup) == len(dup_csv)
    assert _canon_sorted(train_dup).equals(_canon_sorted(dup_csv))
    print(f"augment ok: {len(train_dup)} rows, per-class matches, content == ciciov2024_train_dup.csv")

    # --- encode + scale ---
    y_train, y_test, enc = encode_labels(train_dup, test)
    X_train, X_test, scaler = scale_features(train_dup, test, FEATURES)
    arrays = np.load(PROC / "stage2_arrays.npz")
    assert X_train.shape == arrays["X_train"].shape == (len(train_dup), len(FEATURES))
    assert X_test.shape == arrays["X_test"].shape == (len(test), len(FEATURES))

    import joblib
    comm_scaler = joblib.load(PROC / "feature_scaler.joblib")
    comm_enc = joblib.load(PROC / "label_encoder.joblib")
    assert np.allclose(scaler.data_min_, comm_scaler.data_min_)
    assert np.allclose(scaler.data_max_, comm_scaler.data_max_)
    assert list(enc.classes_) == list(comm_enc.classes_)

    # order-independent content check on the scaled arrays
    assert np.allclose(_sort_rows(X_test), _sort_rows(arrays["X_test"]), atol=1e-6)
    assert np.allclose(_sort_rows(X_train), _sort_rows(arrays["X_train"]), atol=1e-6)
    print(f"encode+scale ok: shapes {X_train.shape}/{X_test.shape}, scaler+encoder+content match")
    print(f"  label mapping: {list(enc.classes_)}")


if __name__ == "__main__":
    test_ciciov_pipeline()
    print("\nSTEP-3 GATE PASSED")
