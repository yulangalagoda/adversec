"""
encode.py
=========
Label encoding and feature scaling, both fit on TRAIN only.

The scaler is a per-column MinMaxScaler, so each feature is scaled by its own
observed range -- the arbitration ID (0..2047) by its range, each payload byte
(0..255) by its. Fitting on train only keeps test min/max from leaking into
training. The [0,1] scaled space is also the coordinate system for adversarial
perturbations later.
"""
from __future__ import annotations

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from ..contract import FEATURES, LABEL_COLUMN


def encode_labels(train_df, test_df, target_column=LABEL_COLUMN):
    """Map class strings to integers, fitting on TRAIN only. Returns (y_train, y_test, encoder)."""
    encoder = LabelEncoder()
    encoder.fit(train_df[target_column])
    y_train = encoder.transform(train_df[target_column])
    y_test = encoder.transform(test_df[target_column])
    return y_train, y_test, encoder


def scale_features(train_df, test_df, feature_columns=FEATURES):
    """Scale features to [0,1] per column, fitting on TRAIN only. Returns (X_train, X_test, scaler)."""
    X_train_raw = train_df[list(feature_columns)].values
    X_test_raw = test_df[list(feature_columns)].values
    scaler = MinMaxScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    return X_train, X_test, scaler
