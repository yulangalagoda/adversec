"""Dataset adapters — the only dataset-aware code in the package."""

from .base import CANDataset
from .registry import available, get_dataset

__all__ = ["CANDataset", "get_dataset", "available"]
