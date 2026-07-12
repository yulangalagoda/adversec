"""
Dataset-agnostic pipeline stages.

Every function here consumes a canonical CAN table (or arrays derived from one)
and knows nothing about which dataset produced it. Order of use:
    strict_dedup -> split_train_test -> duplicate_train_classes -> encode/scale
"""

from .augment import duplicate_train_classes
from .dedup import audit_duplication, strict_dedup
from .encode import encode_labels, scale_features
from .split import split_train_test

__all__ = [
    "audit_duplication",
    "strict_dedup",
    "split_train_test",
    "duplicate_train_classes",
    "encode_labels",
    "scale_features",
]
