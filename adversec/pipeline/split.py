"""
split.py
========
Signature-level train/test split.

Splitting happens on the strict unique signatures, BEFORE any augmentation, so no
duplicated or synthetic copy can straddle train and test. A class with fewer than
config.SMALL_CLASS_THRESHOLD signatures sends a single-signature test floor; a
larger class sends config.TEST_FRACTION of its signatures to test. Stratified and
seeded.
"""
from __future__ import annotations

import pandas as pd

from .. import config
from ..contract import LABEL_COLUMN


def split_train_test(strict_df, test_fraction=None, small_class_threshold=None, random_seed=None):
    """Split strict unique signatures into disjoint (train_df, test_df)."""
    test_fraction = config.TEST_FRACTION if test_fraction is None else test_fraction
    small_class_threshold = (
        config.SMALL_CLASS_THRESHOLD if small_class_threshold is None else small_class_threshold
    )
    seed = config.RANDOM_SEED if random_seed is None else random_seed

    train_pieces, test_pieces = [], []
    for class_name in sorted(strict_df[LABEL_COLUMN].unique()):
        class_rows = strict_df[strict_df[LABEL_COLUMN] == class_name]
        n = len(class_rows)
        shuffled = class_rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_test = 1 if n < small_class_threshold else round(n * test_fraction)
        test_pieces.append(shuffled.iloc[:n_test])
        train_pieces.append(shuffled.iloc[n_test:])

    train_df = pd.concat(train_pieces, ignore_index=True)
    test_df = pd.concat(test_pieces, ignore_index=True)
    return train_df, test_df
