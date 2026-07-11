"""
contract.py
===========
The canonical data contract every dataset adapter must satisfy.

Downstream code (dedup, split, encode, models, experiments) only ever receives a
canonical CAN table. It cannot tell CICIoV2024 from ROAD -- which is exactly the
study's design: one pipeline, only the data differs. This module is the single
source of truth for what "canonical" means, so that promise is enforced, not
assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# The nine CAN features, in fixed order: arbitration ID + eight payload bytes.
ID_COLUMN = "ID"
DATA_COLUMNS = [f"DATA_{i}" for i in range(8)]
FEATURES = [ID_COLUMN] + DATA_COLUMNS

# The label column every adapter must attach.
LABEL_COLUMN = "true_class"

# The full canonical column set, in order.
CANONICAL_COLUMNS = FEATURES + [LABEL_COLUMN]

# A CAN payload byte is a byte, in both datasets.
BYTE_MIN = 0
BYTE_MAX = 255


@dataclass(frozen=True)
class DatasetMeta:
    """
    Everything downstream needs to know about a dataset without touching its raw
    form.

    name         : short identifier, e.g. "ciciov2024" or "road".
    class_names  : every class present, including benign, sorted.
    benign_label : which class is the normal/benign one.
    id_max       : maximum legal arbitration ID (2047 for 11-bit standard CAN).
                   This is where the ID range lives, so nothing downstream (e.g.
                   the realism integer-clip) has to hard-code a magic number.
    """

    name: str
    class_names: list[str]
    benign_label: str
    id_max: int


class ContractError(ValueError):
    """Raised when a dataframe does not satisfy the canonical contract."""


def validate(df: pd.DataFrame, meta: DatasetMeta | None = None) -> pd.DataFrame:
    """
    Check that df is a legal canonical CAN table. Returns df unchanged on success,
    raises ContractError otherwise.

    Without meta: structural checks only (columns, integer features, non-null
    labels). With meta: also value ranges (bytes, ID) and that every label is
    within the declared class set.
    """
    # 1. Exactly the canonical columns, no more, no fewer.
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ContractError(f"missing canonical columns: {missing}")
    extra = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    if extra:
        raise ContractError(f"unexpected columns (the adapter must drop these): {extra}")

    # 2. Feature columns: integer-typed and non-null.
    for col in FEATURES:
        if df[col].isnull().any():
            raise ContractError(f"feature column {col} has nulls")
        if not pd.api.types.is_integer_dtype(df[col]):
            raise ContractError(f"feature column {col} must be integer, got {df[col].dtype}")

    # 3. Labels: non-null.
    if df[LABEL_COLUMN].isnull().any():
        raise ContractError(f"{LABEL_COLUMN} has nulls")

    if meta is not None:
        # 4. Payload bytes within [0, 255].
        for col in DATA_COLUMNS:
            lo, hi = int(df[col].min()), int(df[col].max())
            if lo < BYTE_MIN or hi > BYTE_MAX:
                raise ContractError(f"{col} outside byte range [0,255]: found [{lo},{hi}]")
        # 5. Arbitration ID within [0, id_max].
        id_lo, id_hi = int(df[ID_COLUMN].min()), int(df[ID_COLUMN].max())
        if id_lo < 0 or id_hi > meta.id_max:
            raise ContractError(f"ID outside range [0,{meta.id_max}]: found [{id_lo},{id_hi}]")
        # 6. Labels within the declared class set.
        seen = set(df[LABEL_COLUMN].unique())
        declared = set(meta.class_names)
        if not seen.issubset(declared):
            raise ContractError(f"labels not in declared class set: {sorted(seen - declared)}")

    return df
