"""
augment.py
==========
Light duplication of attack classes in the TRAIN split, for convergence only.

Each attack class is sampled up to config.DUP_TARGET rows by repeating its real
signatures; benign is left untouched. This gives the optimiser enough gradient
signal per class so training does not ignore rare classes. It adds volume, not
diversity, and only fires on classes below the target -- so on a signature-rich
dataset (e.g. ROAD) it does nothing. Applied to TRAIN only, after the split.
"""
from __future__ import annotations

import pandas as pd

from .. import config
from ..contract import LABEL_COLUMN


def duplicate_train_classes(train_df, target_per_class=None, benign_class="benign", random_seed=None):
    """Duplicate under-target attack classes up to target_per_class rows; benign untouched."""
    target_per_class = config.DUP_TARGET if target_per_class is None else target_per_class
    seed = config.RANDOM_SEED if random_seed is None else random_seed

    pieces = []
    for class_name in sorted(train_df[LABEL_COLUMN].unique()):
        class_rows = train_df[train_df[LABEL_COLUMN] == class_name]
        n = len(class_rows)
        if class_name == benign_class or n >= target_per_class:
            pieces.append(class_rows)
            continue
        n_extra = target_per_class - n
        extra = class_rows.sample(n=n_extra, replace=True, random_state=seed)
        pieces.append(pd.concat([class_rows, extra], ignore_index=True))

    duplicated = pd.concat(pieces, ignore_index=True)
    return duplicated.sample(frac=1.0, random_state=seed).reset_index(drop=True)
