"""
realism.py
==========
Physical-plausibility checks for adversarial CAN frames.

Adversarial examples are crafted in scaled [0,1] space and are continuous, so
they do not correspond to legal CAN frames as-is. These functions test how much
of the measured model degradation survives when the adversarial examples are
constrained to physically plausible frames:

  Check 1 (nearest-integer): round each feature to the nearest integer in
      [0,255]. The minimal 'must live on the integer grid' constraint.

  Check 2 (observed-range per ID): additionally require that each byte falls
      within the range that byte actually takes for that arbitration ID in real
      traffic. A proxy for protocol legality (not a DBC-level guarantee).
"""

import numpy as np


def round_to_integer_frames(X_scaled, scaler, feature_min=0, feature_max=255):
    """
    Invert scaling, round to nearest integer, clip to [feature_min, feature_max],
    then re-scale. Returns adversarial examples snapped to the integer grid, in
    scaled space (ready to feed back to the model).

    Args:
        X_scaled: adversarial examples in scaled [0,1] space.
        scaler: the fitted MinMaxScaler used for the features.
        feature_min, feature_max: valid raw integer range.

    Returns:
        X_rounded_scaled: same shape, snapped to legal integers, in scaled space.
    """
    # Back to raw feature space
    X_raw = scaler.inverse_transform(X_scaled)

    # Snap to nearest integer and clip to the legal byte/ID range
    X_int = np.clip(np.round(X_raw), feature_min, feature_max)

    # Re-scale so the model sees it in the space it was trained on
    X_rounded_scaled = scaler.transform(X_int).astype(np.float32)
    return X_rounded_scaled, X_int


def learn_observed_ranges(df_raw, id_column, data_columns):
    """
    Learn, per arbitration ID, the observed [min, max] of each feature byte from
    real traffic. Used by the observed-range plausibility check.

    Args:
        df_raw: DataFrame of real frames (raw integer features, not scaled).
        id_column: name of the ID column.
        data_columns: list of the 8 payload byte columns.

    Returns:
        A dict: id_value -> {column -> (min, max)}.
    """
    ranges = {}
    for id_val, group in df_raw.groupby(id_column):
        ranges[id_val] = {
            col: (group[col].min(), group[col].max()) for col in data_columns
        }
    return ranges


def observed_range_mask(X_int, ranges, id_column_index, data_column_indices):
    """
    For each rounded frame, return True if it is physically plausible:
    its ID is known, and every payload byte is within that ID's observed range.

    Args:
        X_int: rounded adversarial frames in RAW integer space (n, n_features).
        ranges: output of learn_observed_ranges.
        id_column_index: index of the ID feature within a row.
        data_column_indices: list of indices of the 8 payload bytes within a row,
            paired in order with the data_columns used to build 'ranges'.

    Returns:
        Boolean array (n,): True where the frame is plausible.
    """
    plausible = np.ones(len(X_int), dtype=bool)

    for row_i in range(len(X_int)):
        id_val = int(round(X_int[row_i, id_column_index]))

        # Unknown ID -> implausible (an attacker injecting a novel ID is detectable)
        if id_val not in ranges:
            plausible[row_i] = False
            continue

        # Every payload byte must sit within that ID's observed range
        id_ranges = ranges[id_val]
        for col_name, col_idx in zip(id_ranges.keys(), data_column_indices):
            lo, hi = id_ranges[col_name]
            val = X_int[row_i, col_idx]
            if val < lo or val > hi:
                plausible[row_i] = False
                break

    return plausible