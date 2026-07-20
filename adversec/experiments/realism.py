"""
realism.py
==========
Physical-plausibility checks for adversarial CAN frames.

Adversarial examples are crafted in scaled [0,1] space and are continuous, so they
do not correspond to legal CAN frames as-is. These functions test how much of the
measured model degradation survives when the examples are constrained to plausible
frames:

  Check 1 (nearest-integer): round each feature to the nearest integer within its
      OWN legal range. Per-feature is essential -- the ID spans up to 2047, the
      payload bytes 0-255 -- so a single global [0,255] clip would wrongly crush
      the ID and read as an "unknown ID" downstream.

  Check 2 (observed-range per ID): additionally require each byte to fall within
      the range that byte actually takes for that arbitration ID in real traffic.
      A proxy for protocol legality, not a DBC-level guarantee.
"""
from __future__ import annotations

import numpy as np


def round_to_integer_frames(X_scaled, scaler):
    """
    Invert scaling, round each feature to the nearest integer, clip each feature to
    its own legal range (from the fitted scaler), then re-scale.

    Returns (X_rounded_scaled, X_int): the frames in scaled space (ready for the
    model) and in raw integer feature space (for the observed-range check).
    """
    X_raw = scaler.inverse_transform(X_scaled)
    X_int = np.round(X_raw)
    # Per-feature legal range from the fitted scaler. floor(min)/ceil(max) absorb
    # the sub-1 overshoot that rounding can cause at the range boundaries. This is
    # what stops the arbitration ID being clamped to a byte range.
    lo = np.floor(scaler.data_min_)
    hi = np.ceil(scaler.data_max_)
    X_int = np.clip(X_int, lo, hi)
    X_rounded_scaled = scaler.transform(X_int).astype(np.float32)
    return X_rounded_scaled, X_int


def learn_observed_ranges(df_raw, id_column, data_columns):
    """Per arbitration ID, learn the observed [min, max] of each payload byte."""
    ranges = {}
    for id_val, group in df_raw.groupby(id_column):
        ranges[id_val] = {col: (group[col].min(), group[col].max()) for col in data_columns}
    return ranges


def observed_range_mask(X_int, ranges, id_column_index, data_column_indices):
    """
    True where a rounded frame is plausible: its ID is known and every payload byte
    is within that ID's observed range.
    """
    plausible = np.ones(len(X_int), dtype=bool)
    for row_i in range(len(X_int)):
        id_val = int(round(X_int[row_i, id_column_index]))
        if id_val not in ranges:
            plausible[row_i] = False   # unknown ID -> detectable injection
            continue
        id_ranges = ranges[id_val]
        for col_name, col_idx in zip(id_ranges.keys(), data_column_indices):
            lo, hi = id_ranges[col_name]
            val = X_int[row_i, col_idx]
            if val < lo or val > hi:
                plausible[row_i] = False
                break
    return plausible


def clip_to_id_envelope(X_int, ranges, id_column_index, data_column_indices):
    """
    Clip each frame's payload bytes into its OWN arbitration ID's observed [min,max]
    range. Frames on an unknown ID are left unchanged -- there is no known range to
    clip to, and an unknown ID alone is already a detectable injection regardless of
    byte content.

    Models an adaptive attacker who already knows the per-ID envelope and deliberately
    stays inside it (the limitation threat_sizing() itself names but does not test).
    """
    X_clipped = X_int.copy()
    for row_i in range(len(X_int)):
        id_val = int(round(X_int[row_i, id_column_index]))
        if id_val not in ranges:
            continue
        id_ranges = ranges[id_val]
        for col_name, col_idx in zip(id_ranges.keys(), data_column_indices):
            lo, hi = id_ranges[col_name]
            X_clipped[row_i, col_idx] = np.clip(X_clipped[row_i, col_idx], lo, hi)
    return X_clipped
