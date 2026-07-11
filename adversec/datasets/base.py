"""
base.py
=======
The one interface every dataset adapter implements.

This is the only object-oriented surface in the codebase, and the only code
allowed to know a dataset's raw format. Downstream code consumes load() + meta()
and is therefore dataset-agnostic -- it cannot tell CICIoV2024 from ROAD. That is
the study's design expressed as architecture: one pipeline, only the data differs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..contract import DatasetMeta


class CANDataset(ABC):
    """Adapter from one raw CAN dataset to the canonical contract."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return a validated canonical CAN table (see contract.CANONICAL_COLUMNS)."""
        raise NotImplementedError

    @abstractmethod
    def meta(self) -> DatasetMeta:
        """Return the dataset's metadata bundle (name, classes, benign label, id_max)."""
        raise NotImplementedError
