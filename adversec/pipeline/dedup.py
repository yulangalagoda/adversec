"""
dedup.py
========
De-duplication and duplication auditing.

strict_dedup keeps one row per unique (features + class) signature. This is the
honest analysis set: it isolates the genuinely distinct CAN frames and discards
the redundant copies that inflate accuracy. audit_duplication measures how
redundant a table is, over any chosen signature.
"""
from __future__ import annotations

import pandas as pd

from ..contract import FEATURES, LABEL_COLUMN


def audit_duplication(df: pd.DataFrame, subset) -> dict:
    """Measure redundancy over the given signature columns. Returns citable counts."""
    total = len(df)
    n_unique = len(df.drop_duplicates(subset=subset))
    n_duplicate = total - n_unique
    rate = (n_duplicate / total) if total else 0.0
    return {
        "total_rows": total,
        "unique_signatures": n_unique,
        "duplicate_rows": n_duplicate,
        "duplication_rate": round(rate, 6),
        "duplication_rate_pct": round(rate * 100, 4),
    }


def strict_dedup(df: pd.DataFrame, feature_columns=FEATURES) -> pd.DataFrame:
    """Keep the first row of each unique (features + class) signature."""
    signature = list(feature_columns) + [LABEL_COLUMN]
    return df.drop_duplicates(subset=signature, keep="first").reset_index(drop=True)
